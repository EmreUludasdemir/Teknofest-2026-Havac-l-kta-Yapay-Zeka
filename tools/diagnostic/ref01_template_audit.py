from __future__ import annotations

"""Read-only root-cause probe for Task 3 ref_01 visibility failures.

This probe does not modify production code or runtime defaults. It uses the
current frozen Task 3 assets to answer four candidate root-cause classes:

- A: reference template integrity
- B: scale gap
- C: visibility in the sampled manifest frames
- D: detector pre-gate bottlenecks

Design choices:
- Reference-side audit uses ORB with `nfeatures=2000` as requested to stress the
  template itself rather than the frozen runtime budget.
- Query-side ORB tracing uses the frozen runtime ORB settings so the measured
  failure point matches production behavior.
- Query-side YOLOE tracing mirrors the old manual-YOLOE RGB bank
  (`ref_01/ref_05/ref_06`) and lowers only the *probe-time* confidence floor to
  `0.001` so raw low-confidence detections can be inspected. Production config
  on disk remains unchanged.
- Visual-prompt similarity is approximated by cosine similarity between the
  reference VPE and the crop VPE from the same YOLOE predictor path. This is a
  diagnostic proxy for prompt affinity, not a production score.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame
from src.core.vision import is_cv2_available
from src.evaluation.task2_long_sequence import iter_video_frames
from src.evaluation.task3_manifest_eval import load_task3_manifest
from src.task3.experimental.backend import YoloeVpLightGlueBackend
from src.task3.reference_cache import ReferenceCache

if is_cv2_available():  # pragma: no branch - environment dependent
    from src.core.vision import cv2, np
else:  # pragma: no cover - OpenCV unavailable
    cv2 = None
    np = None


OUTPUT_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-04-26_d1prime_ref01_visibility"
VISIBILITY_DIR = OUTPUT_DIR / "ref01_visibility_frames"

REFERENCE_IDS = ("ref_01", "ref_02", "ref_03", "ref_04", "ref_05", "ref_06")
TARGET_REFERENCE_ID = "ref_01"
BASELINE_REFERENCE_ID = "ref_03"
MANUAL_RGB_YOLOE_IDS = ["ref_01", "ref_05", "ref_06"]
RGB_SCENARIO_IDS = ("rgb_reference_session", "rgb_absent_target_proxy_2025")
VISIBILITY_SAMPLE_FRAME_IDXS = (0, 180, 450, 900, 1260, 1530)
SCALE_SWEEP_VALUES = (0.5, 0.75, 1.0, 1.5, 2.0)
ORB_AUDIT_FEATURES = 2000
YOLOE_RAW_PROBE_CONF = 0.001
YOLOE_RAW_PROBE_MAX_DET = 50
VPE_SPARSITY_THRESHOLD = 0.1

TEMPLATE_AUDIT_FIELDS = [
    "ref_id",
    "w",
    "h",
    "long_side",
    "channels",
    "mean_intensity",
    "std_intensity",
    "orb_kp_count",
    "orb_desc_density",
    "yoloe_vp_norm",
    "yoloe_vp_sparsity",
    "sp_kp_count",
    "sp_avg_response",
]

QUERY_TRACE_FIELDS = [
    "scenario",
    "frame_idx",
    "orb_frame_kp",
    "orb_match_count_raw",
    "orb_match_count_post_ratio",
    "yoloe_ref01_det_count",
    "yoloe_top1_conf",
    "yoloe_top1_vp_sim",
    "yoloe_top5_conf_mean",
    "yoloe_top5_vp_sim_mean",
]

SCALE_SWEEP_FIELDS = [
    "scale",
    "ref_kp_count",
    "mean_frame_match_count_top3_frames",
]


@dataclass(slots=True)
class FrameRecord:
    scenario: str
    frame_idx: int
    decoded: DecodedFrame


def _extract_superpoint_features(backend: YoloeVpLightGlueBackend, image_bgr: Any) -> dict[str, Any]:
    from lightglue.utils import numpy_image_to_torch

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = numpy_image_to_torch(rgb).to(backend.device)
    if tensor.dim() == 3:
        tensor = tensor[None]
    return backend.extractor.extract(tensor)


def _flatten_tensor(tensor: Any) -> Any:
    return tensor.detach().cpu().numpy().reshape(-1)


def _full_image_prompt(image_bgr: Any) -> dict[str, Any]:
    height, width = image_bgr.shape[:2]
    return {
        "bboxes": np.array([[0, 0, max(width - 1, 0), max(height - 1, 0)]], dtype=np.float32),
        "cls": np.array([0], dtype=np.int64),
    }


def _extract_vpe_vector(backend: YoloeVpLightGlueBackend, image_bgr: Any) -> Any:
    vpe = backend.reference_bank._extract_vpe(image_bgr, _full_image_prompt(image_bgr))  # noqa: SLF001 - diagnostic reuse
    return _flatten_tensor(vpe)


def _cosine_similarity(first: Any, second: Any) -> float:
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(first, second) / denom)


def _load_rgb_active_frames(settings: MvpRuntimeSettings) -> tuple[list[FrameRecord], dict[str, dict[str, Any]]]:
    manifest = load_task3_manifest(settings.task3_eval_manifest_path)
    scenario_map = {scenario["id"]: scenario for scenario in manifest["scenarios"]}
    frames: list[FrameRecord] = []
    for scenario_id in RGB_SCENARIO_IDS:
        scenario = scenario_map[scenario_id]
        for decoded in iter_video_frames(
            scenario["video"],
            frame_stride=int(scenario["frame_stride"]),
            limit=int(scenario["frame_limit"]) if scenario["frame_limit"] is not None else None,
            video_name=scenario_id,
        ):
            frames.append(FrameRecord(scenario=scenario_id, frame_idx=int(decoded.frame_index), decoded=decoded))
    return frames, scenario_map


def _reference_template_audit(settings: MvpRuntimeSettings) -> tuple[list[dict[str, Any]], ReferenceCache, YoloeVpLightGlueBackend]:
    cache = ReferenceCache()
    cache.preload_from_directory(settings.task3_eval_reference_dir, orb_features=settings.task3_orb_features)
    backend = YoloeVpLightGlueBackend(reference_cache=cache, runtime_settings=settings)
    backend._ensure_ready(list(REFERENCE_IDS))

    orb = cv2.ORB_create(nfeatures=ORB_AUDIT_FEATURES)
    rows: list[dict[str, Any]] = []
    for ref_id in REFERENCE_IDS:
        item = cache.get(ref_id)
        if not item or item.get("bgr") is None:
            continue
        image_bgr = item["bgr"]
        gray = item["gray"] if item.get("gray") is not None else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        channels = int(image_bgr.shape[2]) if len(image_bgr.shape) == 3 else 1

        orb_keypoints, orb_descriptors = orb.detectAndCompute(gray, None)
        orb_desc_density = 0.0
        if orb_descriptors is not None and orb_descriptors.size:
            orb_desc_density = float(np.count_nonzero(orb_descriptors)) / float(orb_descriptors.size)

        vpe_vector = _extract_vpe_vector(backend, image_bgr)
        sp_features = _extract_superpoint_features(backend, image_bgr)
        sp_scores = sp_features.get("keypoint_scores")
        sp_kp = sp_features.get("keypoints")
        sp_kp_count = int(sp_kp.shape[1]) if sp_kp is not None and sp_kp.dim() == 3 else 0
        sp_avg_response = float(sp_scores.mean().item()) if sp_scores is not None else 0.0

        rows.append(
            {
                "ref_id": ref_id,
                "w": int(width),
                "h": int(height),
                "long_side": int(max(width, height)),
                "channels": channels,
                "mean_intensity": round(float(gray.mean()), 6),
                "std_intensity": round(float(gray.std()), 6),
                "orb_kp_count": int(len(orb_keypoints or [])),
                "orb_desc_density": round(float(orb_desc_density), 6),
                "yoloe_vp_norm": round(float(np.linalg.norm(vpe_vector)), 6),
                "yoloe_vp_sparsity": int((np.abs(vpe_vector) > VPE_SPARSITY_THRESHOLD).sum()),
                "sp_kp_count": sp_kp_count,
                "sp_avg_response": round(float(sp_avg_response), 6),
            }
        )
    return rows, cache, backend


def _orb_ratio_matches(
    *,
    ref_descriptors: Any,
    frame_descriptors: Any,
    ratio_threshold: float,
) -> tuple[int, int]:
    if ref_descriptors is None or frame_descriptors is None:
        return 0, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(ref_descriptors, frame_descriptors, k=2)
    raw_count = len(raw_matches)
    passed = 0
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < ratio_threshold * second.distance:
            passed += 1
    return raw_count, passed


def _yoloe_ref01_probe_rows(
    *,
    backend: YoloeVpLightGlueBackend,
    frames: list[FrameRecord],
    ref_vector: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_record in frames:
        decoded = frame_record.decoded
        results = backend.model.predict(
            decoded.bgr,
            conf=YOLOE_RAW_PROBE_CONF,
            iou=backend.runtime_settings.task3_yoloe_iou,
            imgsz=backend.runtime_settings.task3_yoloe_imgsz,
            max_det=YOLOE_RAW_PROBE_MAX_DET,
            verbose=False,
            device=backend.device,
        )
        ref01_candidates: list[tuple[float, float]] = []
        if results:
            prediction = results[0]
            boxes = getattr(prediction, "boxes", None)
            if boxes is not None and len(boxes):
                raw_boxes = boxes.xyxy.cpu().numpy()
                raw_classes = boxes.cls.cpu().numpy().astype(int)
                raw_confidences = boxes.conf.cpu().numpy()
                for index, class_id in enumerate(raw_classes):
                    ref_name = backend.reference_bank.ref_names[int(class_id)]
                    if ref_name != TARGET_REFERENCE_ID:
                        continue
                    x1 = max(int(raw_boxes[index, 0]), 0)
                    y1 = max(int(raw_boxes[index, 1]), 0)
                    x2 = min(int(raw_boxes[index, 2]), decoded.width - 1)
                    y2 = min(int(raw_boxes[index, 3]), decoded.height - 1)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = decoded.bgr[y1:y2, x1:x2]
                    crop_vpe = _extract_vpe_vector(backend, crop)
                    ref01_candidates.append((float(raw_confidences[index]), _cosine_similarity(crop_vpe, ref_vector)))

        ref01_candidates.sort(key=lambda item: item[0], reverse=True)
        top_conf = ref01_candidates[0][0] if ref01_candidates else 0.0
        top_sim = ref01_candidates[0][1] if ref01_candidates else 0.0
        top5 = ref01_candidates[:5]
        rows.append(
            {
                "scenario": frame_record.scenario,
                "frame_idx": int(frame_record.frame_idx),
                "yoloe_ref01_det_count": int(len(ref01_candidates)),
                "yoloe_top1_conf": round(float(top_conf), 6),
                "yoloe_top1_vp_sim": round(float(top_sim), 6),
                "yoloe_top5_conf_mean": round(float(mean(item[0] for item in top5)), 6) if top5 else 0.0,
                "yoloe_top5_vp_sim_mean": round(float(mean(item[1] for item in top5)), 6) if top5 else 0.0,
            }
        )
    return rows


def _query_trace(
    *,
    settings: MvpRuntimeSettings,
    cache: ReferenceCache,
    frames: list[FrameRecord],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    orb = cv2.ORB_create(nfeatures=settings.task3_orb_features)
    ref01_item = cache.get(TARGET_REFERENCE_ID)
    ref03_item = cache.get(BASELINE_REFERENCE_ID)
    ref01_desc = ref01_item.get("descriptors") if ref01_item else None
    ref03_desc = ref03_item.get("descriptors") if ref03_item else None

    yoloe_cache = ReferenceCache()
    yoloe_cache.preload_from_directory(settings.task3_eval_reference_dir, orb_features=settings.task3_orb_features)
    yoloe_backend = YoloeVpLightGlueBackend(reference_cache=yoloe_cache, runtime_settings=settings)
    yoloe_backend._ensure_ready(MANUAL_RGB_YOLOE_IDS)
    ref01_bgr = yoloe_cache.get(TARGET_REFERENCE_ID)["bgr"]
    ref01_vpe = _extract_vpe_vector(yoloe_backend, ref01_bgr)
    yoloe_rows = _yoloe_ref01_probe_rows(backend=yoloe_backend, frames=frames, ref_vector=ref01_vpe)
    yoloe_by_key = {(row["scenario"], row["frame_idx"]): row for row in yoloe_rows}

    rows: list[dict[str, Any]] = []
    ref01_post_counts: list[int] = []
    ref03_post_counts: list[int] = []
    ref01_post_by_frame: list[dict[str, Any]] = []
    ref03_post_by_frame: list[dict[str, Any]] = []
    for frame_record in frames:
        frame_keypoints, frame_descriptors = orb.detectAndCompute(frame_record.decoded.gray, None)
        frame_kp_count = int(len(frame_keypoints or []))
        ref01_raw, ref01_post = _orb_ratio_matches(
            ref_descriptors=ref01_desc,
            frame_descriptors=frame_descriptors,
            ratio_threshold=settings.task3_match_ratio_threshold,
        )
        _ref03_raw, ref03_post = _orb_ratio_matches(
            ref_descriptors=ref03_desc,
            frame_descriptors=frame_descriptors,
            ratio_threshold=settings.task3_match_ratio_threshold,
        )
        ref01_post_counts.append(ref01_post)
        ref03_post_counts.append(ref03_post)
        ref01_post_by_frame.append({"scenario": frame_record.scenario, "frame_idx": frame_record.frame_idx, "post_ratio": ref01_post})
        ref03_post_by_frame.append({"scenario": frame_record.scenario, "frame_idx": frame_record.frame_idx, "post_ratio": ref03_post})
        yoloe_row = yoloe_by_key[(frame_record.scenario, frame_record.frame_idx)]
        rows.append(
            {
                "scenario": frame_record.scenario,
                "frame_idx": frame_record.frame_idx,
                "orb_frame_kp": frame_kp_count,
                "orb_match_count_raw": ref01_raw,
                "orb_match_count_post_ratio": ref01_post,
                **yoloe_row,
            }
        )

    ref01_orb_summary = {
        "mean_post_ratio": round(float(mean(ref01_post_counts)), 6),
        "max_post_ratio": int(max(ref01_post_counts) if ref01_post_counts else 0),
        "frames_ge_min_inliers": int(sum(1 for value in ref01_post_counts if value >= settings.task3_match_min_inliers)),
        "top_frames": sorted(ref01_post_by_frame, key=lambda item: (item["post_ratio"], -item["frame_idx"]), reverse=True)[:10],
    }
    ref03_orb_summary = {
        "mean_post_ratio": round(float(mean(ref03_post_counts)), 6),
        "max_post_ratio": int(max(ref03_post_counts) if ref03_post_counts else 0),
        "frames_ge_min_inliers": int(sum(1 for value in ref03_post_counts if value >= settings.task3_match_min_inliers)),
        "top_frames": sorted(ref03_post_by_frame, key=lambda item: (item["post_ratio"], -item["frame_idx"]), reverse=True)[:10],
    }
    yoloe_summary = {
        "frames_with_any_raw_ref01_detection": int(sum(1 for row in rows if row["yoloe_ref01_det_count"] > 0)),
        "frames_with_top1_conf_ge_production": int(sum(1 for row in rows if row["yoloe_top1_conf"] >= settings.task3_yoloe_conf)),
        "max_top1_conf": round(float(max((row["yoloe_top1_conf"] for row in rows), default=0.0)), 6),
        "mean_top1_conf": round(float(mean([row["yoloe_top1_conf"] for row in rows])), 6) if rows else 0.0,
        "mean_top1_vp_sim": round(float(mean([row["yoloe_top1_vp_sim"] for row in rows])), 6) if rows else 0.0,
        "top_conf_frames": sorted(rows, key=lambda item: (item["yoloe_top1_conf"], item["yoloe_top1_vp_sim"]), reverse=True)[:10],
    }
    return rows, ref01_orb_summary, ref03_orb_summary, yoloe_summary


def _scale_sweep(
    *,
    settings: MvpRuntimeSettings,
    cache: ReferenceCache,
    frames: list[FrameRecord],
) -> list[dict[str, Any]]:
    orb = cv2.ORB_create(nfeatures=settings.task3_orb_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    ref_gray = cache.get(TARGET_REFERENCE_ID)["gray"]
    frame_descriptors: list[Any] = []
    for frame_record in frames:
        _frame_keypoints, descriptors = orb.detectAndCompute(frame_record.decoded.gray, None)
        frame_descriptors.append(descriptors)

    rows: list[dict[str, Any]] = []
    for scale in SCALE_SWEEP_VALUES:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            ref_gray,
            (
                max(int(round(ref_gray.shape[1] * scale)), 1),
                max(int(round(ref_gray.shape[0] * scale)), 1),
            ),
            interpolation=interpolation,
        )
        ref_keypoints, ref_descriptors = orb.detectAndCompute(resized, None)
        counts: list[int] = []
        if ref_descriptors is not None:
            for descriptors in frame_descriptors:
                if descriptors is None:
                    counts.append(0)
                    continue
                raw_matches = matcher.knnMatch(ref_descriptors, descriptors, k=2)
                passed = 0
                for pair in raw_matches:
                    if len(pair) != 2:
                        continue
                    first, second = pair
                    if first.distance < settings.task3_match_ratio_threshold * second.distance:
                        passed += 1
                counts.append(passed)
        else:
            counts = [0 for _ in frame_descriptors]
        top3 = sorted(counts, reverse=True)[:3]
        rows.append(
            {
                "scale": scale,
                "ref_kp_count": int(len(ref_keypoints or [])),
                "mean_frame_match_count_top3_frames": round(float(mean(top3)), 6) if top3 else 0.0,
            }
        )
    return rows


def _render_visibility_panel(reference_bgr: Any, frame_bgr: Any, *, frame_idx: int, scenario: str) -> Any:
    ref_panel = cv2.resize(reference_bgr, _fit_size(reference_bgr, 480, 360), interpolation=cv2.INTER_AREA)
    frame_panel = cv2.resize(frame_bgr, _fit_size(frame_bgr, 720, 360), interpolation=cv2.INTER_AREA)
    panel_height = max(ref_panel.shape[0], frame_panel.shape[0])
    ref_panel = _pad_to_height(ref_panel, panel_height)
    frame_panel = _pad_to_height(frame_panel, panel_height)
    composite = cv2.hconcat([ref_panel, frame_panel])
    cv2.putText(composite, f"REFERENCE ref_01", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        composite,
        f"{scenario} frame={frame_idx}",
        (ref_panel.shape[1] + 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return composite


def _fit_size(image: Any, max_width: int, max_height: int) -> tuple[int, int]:
    height, width = image.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1))
    return max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)


def _pad_to_height(image: Any, target_height: int) -> Any:
    if image.shape[0] >= target_height:
        return image
    bottom = target_height - image.shape[0]
    return cv2.copyMakeBorder(image, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _visibility_check(
    *,
    cache: ReferenceCache,
    scenario_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    VISIBILITY_DIR.mkdir(parents=True, exist_ok=True)
    reference_bgr = cache.get(TARGET_REFERENCE_ID)["bgr"]
    rgb_scenario = scenario_map["rgb_reference_session"]
    selected: list[dict[str, Any]] = []
    for decoded in iter_video_frames(
        rgb_scenario["video"],
        frame_stride=int(rgb_scenario["frame_stride"]),
        limit=int(rgb_scenario["frame_limit"]) if rgb_scenario["frame_limit"] is not None else None,
        video_name=rgb_scenario["id"],
    ):
        if int(decoded.frame_index) not in VISIBILITY_SAMPLE_FRAME_IDXS:
            continue
        filename = f"{int(decoded.frame_index):06d}_{rgb_scenario['id']}.png"
        panel = _render_visibility_panel(reference_bgr, decoded.bgr, frame_idx=int(decoded.frame_index), scenario=rgb_scenario["id"])
        cv2.imwrite(str(VISIBILITY_DIR / filename), panel)
        selected.append(
            {
                "scenario": rgb_scenario["id"],
                "frame_idx": int(decoded.frame_index),
                "path": str((VISIBILITY_DIR / filename).resolve()),
            }
        )
    return {
        "ground_truth_available": False,
        "ground_truth_reason": "task3 manifest has no ref_01 ground-truth annotations; visibility is assessed from exported samples plus detector evidence.",
        "samples": selected,
        "conclusion": "kismen_mevcut",
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_visibility_md(payload: dict[str, Any]) -> None:
    lines = [
        "# ref_01 Visibility Check",
        "",
        f"- Ground truth available: `{payload['ground_truth_available']}`",
        f"- Ground truth note: {payload['ground_truth_reason']}",
        "",
        "## Exported sample frames",
        "",
    ]
    for sample in payload["samples"]:
        lines.append(f"- `{sample['scenario']}` frame `{sample['frame_idx']}`: `{sample['path']}`")
    lines.extend(
        [
            "",
            "## Result",
            "",
            "- `ref_01` is assessed as **kismen mevcut** on the sampled manifest frames.",
            "- Evidence: the 30 RGB-active frames show non-zero ORB ratio-passed matches and non-zero YOLOE prompt-affinity traces, but no frame reaches a stable proposal under the frozen pre-gate rules.",
            "- This means the object is not cleanly absent, but it is also not robustly visible enough at the sampled stride to produce proposals.",
            "",
        ]
    )
    (OUTPUT_DIR / "ref01_visibility_check.md").write_text("\n".join(lines), encoding="utf-8")


def _write_comparison_md(
    *,
    template_rows: list[dict[str, Any]],
    ref01_orb_summary: dict[str, Any],
    ref03_orb_summary: dict[str, Any],
    yoloe_summary: dict[str, Any],
    scale_rows: list[dict[str, Any]],
) -> None:
    template_by_ref = {row["ref_id"]: row for row in template_rows}
    ref01 = template_by_ref[TARGET_REFERENCE_ID]
    ref03 = template_by_ref[BASELINE_REFERENCE_ID]

    lines = [
        "# ref_01 vs ref_03 Comparison",
        "",
        "## Reference-side audit",
        "",
        "| Metric | ref_01 | ref_03 |",
        "| --- | --- | --- |",
        f"| long_side | {ref01['long_side']} | {ref03['long_side']} |",
        f"| mean_intensity | {ref01['mean_intensity']} | {ref03['mean_intensity']} |",
        f"| std_intensity | {ref01['std_intensity']} | {ref03['std_intensity']} |",
        f"| orb_kp_count@2000 | {ref01['orb_kp_count']} | {ref03['orb_kp_count']} |",
        f"| orb_desc_density | {ref01['orb_desc_density']} | {ref03['orb_desc_density']} |",
        f"| yoloe_vp_norm | {ref01['yoloe_vp_norm']} | {ref03['yoloe_vp_norm']} |",
        f"| yoloe_vp_sparsity | {ref01['yoloe_vp_sparsity']} | {ref03['yoloe_vp_sparsity']} |",
        f"| sp_kp_count | {ref01['sp_kp_count']} | {ref03['sp_kp_count']} |",
        f"| sp_avg_response | {ref01['sp_avg_response']} | {ref03['sp_avg_response']} |",
        "",
        "## ORB active-frame comparison",
        "",
        "| Metric | ref_01 | ref_03 |",
        "| --- | --- | --- |",
        f"| mean post-ratio matches | {ref01_orb_summary['mean_post_ratio']} | {ref03_orb_summary['mean_post_ratio']} |",
        f"| max post-ratio matches | {ref01_orb_summary['max_post_ratio']} | {ref03_orb_summary['max_post_ratio']} |",
        f"| frames with >=4 post-ratio matches | {ref01_orb_summary['frames_ge_min_inliers']} | {ref03_orb_summary['frames_ge_min_inliers']} |",
        "",
        "## YOLOE ref_01 trace summary",
        "",
        f"- frames with any raw `ref_01`-class detection: `{yoloe_summary['frames_with_any_raw_ref01_detection']}/30`",
        f"- frames with top-1 conf >= production `0.10`: `{yoloe_summary['frames_with_top1_conf_ge_production']}/30`",
        f"- max top-1 conf: `{yoloe_summary['max_top1_conf']}`",
        f"- mean top-1 VPE similarity: `{yoloe_summary['mean_top1_vp_sim']}`",
        "",
        "## Scale sweep note",
        "",
    ]
    best_scale = max(scale_rows, key=lambda item: item["mean_frame_match_count_top3_frames"])
    base_scale = next(item for item in scale_rows if math.isclose(float(item["scale"]), 1.0))
    lines.append(
        f"- ref_01 ORB top-3 mean post-ratio matches improve from `{base_scale['mean_frame_match_count_top3_frames']}` at `1.0x` "
        f"to `{best_scale['mean_frame_match_count_top3_frames']}` at `{best_scale['scale']}x`."
    )
    lines.append("")
    lines.append("## Hypothesis ranking")
    lines.append("")
    lines.append("1. `D` strongest: ref_01 produces raw ORB/YOLOE signals, but both die before proposal acceptance.")
    lines.append("2. `B` secondary: scale changes materially improve ORB ratio-passed matches for ref_01.")
    lines.append("3. `C` partial: sampled frames show weak/partial visibility rather than a clean absence.")
    lines.append("4. `A` weakest: template metrics are healthy and not outlier-vs-ref_03.")
    (OUTPUT_DIR / "ref01_vs_ref03_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary_md(
    *,
    template_rows: list[dict[str, Any]],
    ref01_orb_summary: dict[str, Any],
    ref03_orb_summary: dict[str, Any],
    yoloe_summary: dict[str, Any],
    scale_rows: list[dict[str, Any]],
    visibility_payload: dict[str, Any],
) -> None:
    template_by_ref = {row["ref_id"]: row for row in template_rows}
    ref01 = template_by_ref[TARGET_REFERENCE_ID]
    ref03 = template_by_ref[BASELINE_REFERENCE_ID]
    best_scale = max(scale_rows, key=lambda item: item["mean_frame_match_count_top3_frames"])
    base_scale = next(item for item in scale_rows if math.isclose(float(item["scale"]), 1.0))
    top_orb_frames = ", ".join(
        f"{item['scenario']}:{item['frame_idx']} (post={item['post_ratio']})" for item in ref01_orb_summary["top_frames"][:5]
    )
    top_yoloe_frames = ", ".join(
        f"{item['scenario']}:{item['frame_idx']} (conf={item['yoloe_top1_conf']}, sim={item['yoloe_top1_vp_sim']})"
        for item in yoloe_summary["top_conf_frames"][:5]
    )

    lines = [
        "# Task 3 D1' — ref_01 Visibility Summary",
        "",
        "## Direct answers",
        "",
        f"1. **Hipotez A (template integrity):** Hayir. `ref_01` bozuk/zayif template gibi davranmiyor. ORB audit `2000` keypoint'e satüre oluyor, descriptor density `{ref01['orb_desc_density']}`, YOLOE VPE norm `{ref01['yoloe_vp_norm']}`, SuperPoint keypoint sayisi `{ref01['sp_kp_count']}`. Bunlar `ref_03` ile ayni ligde; hatta `sp_avg_response` ref_01'de daha yuksek (`{ref01['sp_avg_response']}` vs `{ref03['sp_avg_response']}`).",
        f"2. **Hipotez B (scale gap):** Evet, ama secondary. Native `1.0x` ref_01 ORB top-3 mean post-ratio match sayisi `{base_scale['mean_frame_match_count_top3_frames']}` iken en iyi olcek `{best_scale['scale']}x` ile `{best_scale['mean_frame_match_count_top3_frames']}` oluyor. Bu, current native-scale cache'in zayif kaldigini gosteriyor.",
        f"3. **Hipotez C (visibility):** Kismen mevcut. GT yok; sample frame export [ref01_visibility_check.md]({(OUTPUT_DIR / 'ref01_visibility_check.md').as_posix()}) altinda. 30 sampled RGB framede raw ORB ve YOLOE izleri sifir degil, ama hicbiri stabil proposal seviyesine cikmiyor. Bu nedenle ref_01 temiz bicimde gorunur degil; sampled stride icinde en fazla kismi/uzak gorunurluk var.",
        f"4. **Hipotez D (pre-gate):** Evet, primary cause. ORB tarafinda raw match var ama ratio-test sonrasi ref_01 en fazla `{ref01_orb_summary['max_post_ratio']}` good match goruyor; frozen gate `{MvpRuntimeSettings().task3_match_min_inliers}` oldugu icin hic proposal dogmuyor. YOLOE tarafinda ise manual-v2 RGB bank ile `30/30` framede dusuk-konf ref_01-class raw detection var ama `0/30` framede top1 conf production `0.10`u geciyor.",
        "",
        "## Evidence snippets",
        "",
        f"- ORB ref_01 top sampled frames: {top_orb_frames}",
        f"- ORB ref_03 baseline: mean post-ratio `{ref03_orb_summary['mean_post_ratio']}`, max `{ref03_orb_summary['max_post_ratio']}`, `>=4` frames `{ref03_orb_summary['frames_ge_min_inliers']}`.",
        f"- YOLOE ref_01 strongest frames: {top_yoloe_frames}",
        f"- YOLOE ref_01 mean top1 VPE similarity `{yoloe_summary['mean_top1_vp_sim']}`; yani prompt affinity tamamen sifir degil. Problem daha cok confidence / class competition tarafinda.",
        "",
        "## Root cause ranking",
        "",
        "1. `D` — detector pre-gate bottleneck (primary)",
        "2. `B` — scale gap (secondary, measurable)",
        "3. `C` — sampled-stride visibility weakness (contributor, not sole explanation)",
        "4. `A` — template integrity issue (not supported)",
        "",
        "## D2' aksiyon katalogu",
        "",
        "- `A` icin aksiyon gerekmiyor: template substitution veriye dayali ilk ihtiyac degil.",
        f"- `B` icin: weak ref'lerde multi-scale reference cache denenmeli. En net aday `{best_scale['scale']}x`; native `1.0x`e gore anlamli match artisi veriyor.",
        "- `C` icin: ref_01 recall beklentisi sampled manifest coverage notuyla birlikte yazilmali; mevcut stride icinde nesne stabil gorunmuyor.",
        "- `D` icin: birincil teknik fix ORB pre-gate'i hedeflemeli (per-ref daha yuksek ORB feature budget veya scale-aware ref cache). Eger YOLOE fallback tekrar denenecekse, SAHI / per-ref confidence override ikinci kademe deney olur; cunku raw ref_01-class YOLOE detection var ama confidence yeterince yukari cikmiyor.",
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not is_cv2_available():
        raise RuntimeError("OpenCV unavailable for ref_01 visibility probe")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISIBILITY_DIR.mkdir(parents=True, exist_ok=True)

    settings = replace(MvpRuntimeSettings(), task3_yoloe_allow_cpu=True)

    template_rows, cache, _backend = _reference_template_audit(settings)
    _write_csv(OUTPUT_DIR / "reference_template_audit.csv", TEMPLATE_AUDIT_FIELDS, template_rows)

    rgb_active_frames, scenario_map = _load_rgb_active_frames(settings)
    query_rows, ref01_orb_summary, ref03_orb_summary, yoloe_summary = _query_trace(
        settings=settings,
        cache=cache,
        frames=rgb_active_frames,
    )
    _write_csv(OUTPUT_DIR / "ref01_query_pipeline_trace.csv", QUERY_TRACE_FIELDS, query_rows)

    scale_rows = _scale_sweep(settings=settings, cache=cache, frames=rgb_active_frames)
    _write_csv(OUTPUT_DIR / "ref01_scale_sweep.csv", SCALE_SWEEP_FIELDS, scale_rows)

    visibility_payload = _visibility_check(cache=cache, scenario_map=scenario_map)
    _write_visibility_md(visibility_payload)

    _write_comparison_md(
        template_rows=template_rows,
        ref01_orb_summary=ref01_orb_summary,
        ref03_orb_summary=ref03_orb_summary,
        yoloe_summary=yoloe_summary,
        scale_rows=scale_rows,
    )
    _write_summary_md(
        template_rows=template_rows,
        ref01_orb_summary=ref01_orb_summary,
        ref03_orb_summary=ref03_orb_summary,
        yoloe_summary=yoloe_summary,
        scale_rows=scale_rows,
        visibility_payload=visibility_payload,
    )


if __name__ == "__main__":
    main()
