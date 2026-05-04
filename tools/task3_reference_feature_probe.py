"""Probe script for Task 3 official-reference feature and visibility diagnostics.

This is a read-only diagnostic tool. It reuses the YOLOE + SuperPoint + LightGlue
backend code path to inspect why selected references produce zero gate-passing
candidates. It does not modify runtime settings on disk or production behavior.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.core.vision import is_cv2_available
from src.core.video_io import iter_video_frames
from src.evaluation.task3_manifest_eval import load_task3_manifest
from src.task3.experimental.backend import YoloeVpLightGlueBackend
from src.task3.reference_cache import ReferenceCache

if is_cv2_available():  # pragma: no branch
    from src.core.vision import cv2
else:  # pragma: no cover
    cv2 = None


OUTPUT_DIR = Path("_logs") / "reference_feature_probe"
RGB_SCENARIO_ID = "rgb_reference_session"
THERMAL_SCENARIO_ID = "thermal_cross_sensor_proxy"
TARGET_REFS = ("ref_01", "ref_02", "ref_03")
REFERENCE_TABLE_REFS = ("ref_01", "ref_02", "ref_03", "ref_04", "ref_05", "ref_06")


def _long_side_resize(image: Any, long_side: int) -> Any:
    height, width = image.shape[:2]
    current_long_side = max(height, width)
    if current_long_side <= long_side:
        return image.copy()
    scale = float(long_side) / float(current_long_side)
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _feature_keypoint_count(features: dict[str, Any]) -> int:
    keypoints = features.get("keypoints")
    if keypoints is None:
        return 0
    if hasattr(keypoints, "dim"):
        if keypoints.dim() == 3:
            return int(keypoints.shape[1])
        return int(keypoints.shape[0])
    return len(keypoints)


def _extract_features(backend: YoloeVpLightGlueBackend, image_bgr: Any) -> dict[str, Any]:
    from lightglue.utils import numpy_image_to_torch

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = numpy_image_to_torch(rgb).to(backend.device)
    if tensor.dim() == 3:
        tensor = tensor[None]
    return backend.extractor.extract(tensor)


def _probe_crop_match(
    backend: YoloeVpLightGlueBackend,
    crop_bgr: Any,
    ref_features: dict[str, Any],
) -> dict[str, Any]:
    from lightglue.utils import rbd

    crop_tensor = backend.verifier._prepare_crop(crop_bgr)  # noqa: SLF001 - diagnostic reuse
    if crop_tensor is None:
        return {
            "crop_kpts": 0,
            "match_count": 0,
            "gate_pass": False,
        }
    crop_features = backend.extractor.extract(crop_tensor)
    result = backend.matcher({"image0": ref_features, "image1": crop_features})
    result = rbd(result)
    matches = result.get("matches")
    match_count = int(matches.shape[0]) if matches is not None else 0
    return {
        "crop_kpts": _feature_keypoint_count(crop_features),
        "match_count": match_count,
        "gate_pass": match_count >= backend.runtime_settings.task3_lightglue_min_matches,
    }


def _collect_pre_gate_candidates(
    backend: YoloeVpLightGlueBackend,
    *,
    scenario: dict[str, Any],
    reference_ids: list[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ref_name_to_index = {name: index for index, name in enumerate(backend.reference_bank.ref_names or [])}
    allowed_ids = set(reference_ids)
    for decoded_frame in iter_video_frames(
        scenario["video"],
        frame_stride=int(scenario["frame_stride"]),
        limit=int(scenario["frame_limit"]) if scenario["frame_limit"] is not None else None,
        video_name=scenario["id"],
    ):
        results = backend.model.predict(
            decoded_frame.bgr,
            conf=backend.runtime_settings.task3_yoloe_conf,
            iou=backend.runtime_settings.task3_yoloe_iou,
            imgsz=backend.runtime_settings.task3_yoloe_imgsz,
            max_det=max(backend.runtime_settings.task3_yoloe_max_det_per_class * max(len(reference_ids), 1), 1),
            verbose=False,
            device=backend.device,
        )
        if not results:
            continue
        prediction = results[0]
        boxes = getattr(prediction, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        raw_boxes = boxes.xyxy.cpu().numpy()
        raw_classes = boxes.cls.cpu().numpy().astype(int)
        raw_confidences = boxes.conf.cpu().numpy()
        grouped: dict[int, list[int]] = {}
        for index, class_id in enumerate(raw_classes):
            if not 0 <= int(class_id) < len(backend.reference_bank.ref_names or []):
                continue
            ref_name = backend.reference_bank.ref_names[int(class_id)]
            if ref_name not in allowed_ids:
                continue
            grouped.setdefault(int(class_id), []).append(index)

        keep_indices: list[int] = []
        for class_id, indices in grouped.items():
            del class_id
            sorted_indices = sorted(indices, key=lambda idx: -float(raw_confidences[idx]))
            keep_indices.extend(sorted_indices[: backend.runtime_settings.task3_yoloe_max_det_per_class])

        for candidate_idx, index in enumerate(keep_indices):
            class_id = int(raw_classes[index])
            object_id = backend.reference_bank.ref_names[class_id]
            x1 = max(int(raw_boxes[index, 0]), 0)
            y1 = max(int(raw_boxes[index, 1]), 0)
            x2 = min(int(raw_boxes[index, 2]), decoded_frame.width - 1)
            y2 = min(int(raw_boxes[index, 3]), decoded_frame.height - 1)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_bgr = decoded_frame.bgr[y1:y2, x1:x2]
            probe = _probe_crop_match(
                backend,
                crop_bgr,
                backend.reference_bank.sp_features[ref_name_to_index[object_id]],
            )
            candidates.append(
                {
                    "scenario_id": scenario["id"],
                    "frame_idx": int(decoded_frame.frame_index),
                    "candidate_idx": int(candidate_idx),
                    "object_id": object_id,
                    "bbox": [x1, y1, x2, y2],
                    "yoloe_confidence": round(float(raw_confidences[index]), 6),
                    "crop_dims": [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])],
                    "crop_kpts": int(probe["crop_kpts"]),
                    "match_count": int(probe["match_count"]),
                    "gate_pass": bool(probe["gate_pass"]),
                }
            )
    return candidates


def _measure_reference_features(backend: YoloeVpLightGlueBackend, cache: ReferenceCache) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference_id in REFERENCE_TABLE_REFS:
        item = cache.get(reference_id)
        if not item or item.get("bgr") is None:
            rows.append(
                {
                    "ref": reference_id,
                    "orig_dims": None,
                    "kpts_raw": None,
                    "kpts_1280": None,
                    "kpts_640": None,
                }
            )
            continue
        image = item["bgr"]
        raw_features = _extract_features(backend, image)
        features_1280 = _extract_features(backend, _long_side_resize(image, 1280))
        features_640 = _extract_features(backend, _long_side_resize(image, 640))
        rows.append(
            {
                "ref": reference_id,
                "orig_dims": [int(image.shape[0]), int(image.shape[1])],
                "kpts_raw": _feature_keypoint_count(raw_features),
                "kpts_1280": _feature_keypoint_count(features_1280),
                "kpts_640": _feature_keypoint_count(features_640),
            }
        )
    return rows


def _load_scenarios(settings: MvpRuntimeSettings) -> dict[str, dict[str, Any]]:
    manifest = load_task3_manifest(settings.task3_eval_manifest_path)
    return {scenario["id"]: scenario for scenario in manifest["scenarios"]}


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Task 3 Reference Feature Probe")
    lines.append("")
    lines.append("## Table 1 — Reference-side feature quality")
    lines.append("")
    lines.append("| ref | orig dims (HxW) | kpts raw | kpts @ 1280 | kpts @ 640 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["reference_feature_rows"]:
        dims = "-" if row["orig_dims"] is None else f"{row['orig_dims'][0]}x{row['orig_dims'][1]}"
        lines.append(f"| {row['ref']} | {dims} | {row['kpts_raw']} | {row['kpts_1280']} | {row['kpts_640']} |")

    lines.append("")
    lines.append("## Table 2 — Candidate-side feature quality for ref_01/ref_02/ref_03")
    lines.append("")
    lines.append("| ref | frame | bbox | crop dims (HxW) | crop kpts | matches | gate pass |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["candidate_probe_rows"]:
        bbox = "[" + ", ".join(str(item) for item in row["bbox"]) + "]" if row["bbox"] else "-"
        dims = "-" if row["crop_dims"] is None else f"{row['crop_dims'][0]}x{row['crop_dims'][1]}"
        lines.append(f"| {row['ref']} | {row['frame']} | {bbox} | {dims} | {row['crop_kpts']} | {row['matches']} | {row['gate_pass']} |")

    lines.append("")
    lines.append("## Visibility summary")
    lines.append("")
    for ref_id, summary in sorted(payload["visibility_summary"].items()):
        lines.append(f"- `{ref_id}`: rgb pre-gate={summary['rgb_reference_session']}, thermal pre-gate={summary['thermal_cross_sensor_proxy']}")

    lines.append("")
    lines.append("## CPU timing sample")
    lines.append("")
    timing = payload.get("cpu_timing_sample", {})
    if timing:
        for key, value in timing.items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- unavailable")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not is_cv2_available():
        raise RuntimeError("OpenCV unavailable")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = MvpRuntimeSettings()
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("torch unavailable for probe") from exc

    probe_settings = replace(
        settings,
        task3_reference_dir=settings.task3_eval_reference_dir,
        task3_yoloe_allow_cpu=settings.task3_yoloe_allow_cpu or not bool(torch.cuda.is_available()),
    )
    cache = ReferenceCache()
    cache.preload_from_directory(probe_settings.task3_eval_reference_dir, orb_features=probe_settings.task3_orb_features)
    backend = YoloeVpLightGlueBackend(reference_cache=cache, runtime_settings=probe_settings)
    reference_ids = cache.list_ids()
    backend._ensure_ready(reference_ids)  # noqa: SLF001 - diagnostic reuse

    scenarios = _load_scenarios(probe_settings)
    rgb_scenario = scenarios[RGB_SCENARIO_ID]
    thermal_scenario = scenarios[THERMAL_SCENARIO_ID]

    reference_feature_rows = _measure_reference_features(backend, cache)
    rgb_candidates = _collect_pre_gate_candidates(backend, scenario=rgb_scenario, reference_ids=reference_ids)
    thermal_candidates = _collect_pre_gate_candidates(backend, scenario=thermal_scenario, reference_ids=reference_ids)

    selected_rows: list[dict[str, Any]] = []
    rgb_counts = Counter(item["object_id"] for item in rgb_candidates)
    thermal_counts = Counter(item["object_id"] for item in thermal_candidates)
    for reference_id in TARGET_REFS:
        failed_candidates = [item for item in rgb_candidates if item["object_id"] == reference_id and not item["gate_pass"]]
        if not failed_candidates:
            selected_rows.append(
                {
                    "ref": reference_id,
                    "frame": "NONE",
                    "bbox": None,
                    "crop_dims": None,
                    "crop_kpts": 0,
                    "matches": 0,
                    "gate_pass": "no_candidate",
                }
            )
            continue
        for item in failed_candidates[:3]:
            selected_rows.append(
                {
                    "ref": reference_id,
                    "frame": item["frame_idx"],
                    "bbox": item["bbox"],
                    "crop_dims": item["crop_dims"],
                    "crop_kpts": item["crop_kpts"],
                    "matches": item["match_count"],
                    "gate_pass": item["gate_pass"],
                }
            )

    cpu_timing_sample: dict[str, Any] = {}
    if backend.device == "cpu":
        sample_frame = next(
            iter_video_frames(
                rgb_scenario["video"],
                frame_stride=int(rgb_scenario["frame_stride"]),
                limit=1,
                video_name=rgb_scenario["id"],
            )
        )
        for imgsz in (1280, 1920, 2560):
            start = time.perf_counter()
            backend.model.predict(
                sample_frame.bgr,
                conf=backend.runtime_settings.task3_yoloe_conf,
                iou=backend.runtime_settings.task3_yoloe_iou,
                imgsz=imgsz,
                max_det=max(backend.runtime_settings.task3_yoloe_max_det_per_class * max(len(reference_ids), 1), 1),
                verbose=False,
                device=backend.device,
            )
            cpu_timing_sample[f"imgsz_{imgsz}_ms_one_frame"] = round((time.perf_counter() - start) * 1000.0, 4)

    payload = {
        "reference_feature_rows": reference_feature_rows,
        "candidate_probe_rows": selected_rows,
        "visibility_summary": {
            reference_id: {
                "rgb_reference_session": int(rgb_counts.get(reference_id, 0)),
                "thermal_cross_sensor_proxy": int(thermal_counts.get(reference_id, 0)),
            }
            for reference_id in TARGET_REFS
        },
        "rgb_candidate_totals_by_ref": dict(sorted(rgb_counts.items())),
        "thermal_candidate_totals_by_ref": dict(sorted(thermal_counts.items())),
        "device": backend.device,
        "weight_path": str(probe_settings.task3_yoloe_weight_path),
        "lightglue_min_matches": probe_settings.task3_lightglue_min_matches,
        "cpu_timing_sample": cpu_timing_sample,
    }

    (OUTPUT_DIR / "reference_feature_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "reference_feature_probe.md").write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
