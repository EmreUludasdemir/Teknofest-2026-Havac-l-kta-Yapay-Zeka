from __future__ import annotations

"""Read-only SAHI feasibility probe for Task 3 D3.

Goal: for `ref_01` and `ref_02` (both flagged `large_rgb_reference` by
auto-routing) on `rgb_reference_session` and `rgb_absent_target_proxy_2025`,
compare two configurations:

- baseline: full-frame YOLOE inference at imgsz=1280
- sahi:     2x2 grid manual slicing with overlap_ratio=0.2 at imgsz=640 per slice

For each frame, ref, and config, the probe writes a row capturing the YOLOE
top-1 confidence, gate pass status, candidate count, LightGlue match count,
RANSAC inlier statistics, final score, threshold pass status, and per-frame
latency components. No production code is modified; all state changes are
local to this script.

Usage:
  py -3.12 tools/diagnostic/sahi_probe.py
"""

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame
from src.core.vision import cv2, is_cv2_available, np
from src.evaluation.task2_long_sequence import iter_video_frames
from src.task3.experimental.backend import (
    YoloeVpLightGlueBackend,
    _compute_homography_consistency,
)
from src.task3.no_match_logic import compute_mode_candidate_score, normalize_yoloe_match_count
from src.task3.reference_cache import ReferenceCache

MANIFEST_PATH = PROJECT_ROOT / "data" / "task3_eval_manifest.json"
REFERENCE_DIR = PROJECT_ROOT / "data" / "references" / "2026_baseline"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-04-27_d3_sahi_feasibility"
)

TARGET_REFERENCE_IDS = ("ref_01", "ref_02")
TARGET_SCENARIO_IDS = ("rgb_reference_session", "rgb_absent_target_proxy_2025")
SAHI_GRID_ROWS = 2
SAHI_GRID_COLS = 2
SAHI_OVERLAP_RATIO = 0.2
SAHI_IMGSZ = 640
BASELINE_IMGSZ = 1280
SAHI_NMS_IOU = 0.5

CSV_FIELDS = [
    "scenario",
    "frame_idx",
    "ref_id",
    "config",
    "slice_count",
    "yoloe_top1_conf",
    "yoloe_top1_passes_gate",
    "candidate_count_post_gate",
    "lightglue_max_matches",
    "ransac_max_inlier_count",
    "ransac_max_inlier_ratio",
    "final_score_max",
    "final_score_passes_threshold",
    "yoloe_inference_ms",
    "lightglue_verify_ms",
    "total_per_frame_ms",
]


@dataclass(slots=True)
class CandidateMeasurement:
    ref_id: str
    yoloe_confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    match_count: int
    inlier_count: int
    inlier_ratio: float
    homography_found: bool
    final_score: float


def _slice_rects(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Return SAHI 2x2 slice rectangles (x1, y1, x2, y2) in absolute pixels."""

    slice_w = int(round(width * (1.0 + SAHI_OVERLAP_RATIO) / SAHI_GRID_COLS))
    slice_h = int(round(height * (1.0 + SAHI_OVERLAP_RATIO) / SAHI_GRID_ROWS))
    slice_w = max(1, min(slice_w, width))
    slice_h = max(1, min(slice_h, height))

    xs = (0, max(0, width - slice_w))
    ys = (0, max(0, height - slice_h))

    rects: list[tuple[int, int, int, int]] = []
    for y_start in ys:
        for x_start in xs:
            x_end = min(width, x_start + slice_w)
            y_end = min(height, y_start + slice_h)
            rects.append((x_start, y_start, x_end, y_end))
    return rects


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = max(1, area_a + area_b - inter)
    return inter / union


def _per_ref_nms(
    boxes_per_ref: dict[str, list[tuple[float, tuple[int, int, int, int]]]],
    iou_threshold: float,
) -> dict[str, list[tuple[float, tuple[int, int, int, int]]]]:
    kept: dict[str, list[tuple[float, tuple[int, int, int, int]]]] = {}
    for ref_id, candidates in boxes_per_ref.items():
        sorted_candidates = sorted(candidates, key=lambda item: -float(item[0]))
        survivors: list[tuple[float, tuple[int, int, int, int]]] = []
        for conf, bbox in sorted_candidates:
            if any(_iou(bbox, prev_bbox) >= iou_threshold for _prev_conf, prev_bbox in survivors):
                continue
            survivors.append((conf, bbox))
        kept[ref_id] = survivors
    return kept


def _verify_and_score(
    backend: YoloeVpLightGlueBackend,
    full_bgr: Any,
    bbox: tuple[int, int, int, int],
    ref_class_index: int,
    confidence: float,
) -> tuple[CandidateMeasurement | None, float]:
    """Run LightGlue verify + RANSAC + score on a candidate. Returns (measurement, verify_ms)."""

    assert backend.verifier is not None
    assert backend.reference_bank is not None
    assert backend.reference_bank.sp_features is not None
    assert backend.reference_bank.ref_names is not None

    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None, 0.0
    crop = full_bgr[y1:y2, x1:x2]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None, 0.0

    verify_start = time.perf_counter()
    verify_passed, match_count, _internal_verify_ms, kpts_data = backend.verifier.verify(
        crop, backend.reference_bank.sp_features[ref_class_index]
    )
    verify_ms = (time.perf_counter() - verify_start) * 1000.0

    inlier_count, inlier_ratio, homography_found, _homography_ms = _compute_homography_consistency(
        kpts_data,
        reproj_threshold=backend.runtime_settings.task3_yoloe_homography_ransac_reproj_threshold,
    )

    if not verify_passed:
        return (
            CandidateMeasurement(
                ref_id=backend.reference_bank.ref_names[ref_class_index],
                yoloe_confidence=float(confidence),
                bbox_xyxy=bbox,
                match_count=int(match_count),
                inlier_count=int(inlier_count),
                inlier_ratio=float(inlier_ratio),
                homography_found=bool(homography_found),
                final_score=0.0,
            ),
            verify_ms,
        )

    normalized_matches = normalize_yoloe_match_count(
        match_count=int(match_count),
        normalization_scale=backend.runtime_settings.task3_yoloe_match_normalization_scale,
    )
    score = compute_mode_candidate_score(
        confidence=float(confidence),
        normalized_matches=normalized_matches,
        inlier_ratio=float(inlier_ratio),
        mode="yoloe_vp_lightglue",
        yoloe_confidence_weight=backend.runtime_settings.task3_yoloe_score_confidence_weight,
        yoloe_matches_weight=backend.runtime_settings.task3_yoloe_score_matches_weight,
        yoloe_inlier_weight=backend.runtime_settings.task3_yoloe_score_inlier_weight,
    )
    return (
        CandidateMeasurement(
            ref_id=backend.reference_bank.ref_names[ref_class_index],
            yoloe_confidence=float(confidence),
            bbox_xyxy=bbox,
            match_count=int(match_count),
            inlier_count=int(inlier_count),
            inlier_ratio=float(inlier_ratio),
            homography_found=bool(homography_found),
            final_score=float(score),
        ),
        verify_ms,
    )


def _yoloe_predict_boxes(
    backend: YoloeVpLightGlueBackend,
    image_bgr: Any,
    *,
    imgsz: int,
) -> tuple[list[tuple[int, float, tuple[int, int, int, int]]], float]:
    """Return (class_idx, confidence, bbox_xyxy) tuples and total inference ms.

    bbox is in image_bgr's local coordinates.
    """

    assert backend.model is not None
    assert backend.reference_bank is not None
    assert backend.reference_bank.ref_names is not None

    height, width = image_bgr.shape[:2]
    ref_count = len(backend.reference_bank.ref_names)
    max_det = max(backend.runtime_settings.task3_yoloe_max_det_per_class * max(ref_count, 1), 1)

    start = time.perf_counter()
    results = backend.model.predict(
        image_bgr,
        conf=backend.runtime_settings.task3_yoloe_conf,
        iou=backend.runtime_settings.task3_yoloe_iou,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
        device=backend.device,
    )
    inference_ms = (time.perf_counter() - start) * 1000.0

    boxes_out: list[tuple[int, float, tuple[int, int, int, int]]] = []
    if not results:
        return boxes_out, inference_ms
    prediction = results[0]
    boxes = getattr(prediction, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return boxes_out, inference_ms
    raw_boxes = boxes.xyxy.cpu().numpy()
    raw_classes = boxes.cls.cpu().numpy().astype(int)
    raw_confidences = boxes.conf.cpu().numpy()
    for index, class_id in enumerate(raw_classes):
        if not 0 <= int(class_id) < ref_count:
            continue
        x1 = max(int(raw_boxes[index, 0]), 0)
        y1 = max(int(raw_boxes[index, 1]), 0)
        x2 = min(int(raw_boxes[index, 2]), width - 1)
        y2 = min(int(raw_boxes[index, 3]), height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes_out.append((int(class_id), float(raw_confidences[index]), (x1, y1, x2, y2)))
    return boxes_out, inference_ms


def _measure_baseline(
    backend: YoloeVpLightGlueBackend, decoded_frame: DecodedFrame
) -> tuple[dict[str, list[CandidateMeasurement]], float, float, float]:
    full_bgr = decoded_frame.bgr
    boxes, inference_ms = _yoloe_predict_boxes(backend, full_bgr, imgsz=BASELINE_IMGSZ)
    per_ref_boxes: dict[str, list[tuple[float, tuple[int, int, int, int]]]] = {}
    box_to_class_index: dict[tuple[int, int, int, int], int] = {}
    for class_idx, conf, bbox in boxes:
        ref_id = backend.reference_bank.ref_names[class_idx]
        per_ref_boxes.setdefault(ref_id, []).append((conf, bbox))
        box_to_class_index[bbox] = class_idx
    deduped = _per_ref_nms(per_ref_boxes, iou_threshold=SAHI_NMS_IOU)

    measurements: dict[str, list[CandidateMeasurement]] = {}
    verify_ms_total = 0.0
    for ref_id, kept in deduped.items():
        for conf, bbox in kept:
            class_idx = box_to_class_index[bbox]
            measurement, verify_ms = _verify_and_score(backend, full_bgr, bbox, class_idx, conf)
            verify_ms_total += verify_ms
            if measurement is not None:
                measurements.setdefault(ref_id, []).append(measurement)
    total_ms = inference_ms + verify_ms_total
    return measurements, inference_ms, verify_ms_total, total_ms


def _measure_sahi(
    backend: YoloeVpLightGlueBackend, decoded_frame: DecodedFrame
) -> tuple[dict[str, list[CandidateMeasurement]], float, float, float, int]:
    full_bgr = decoded_frame.bgr
    height, width = full_bgr.shape[:2]
    rects = _slice_rects(width, height)
    inference_ms_total = 0.0
    per_ref_boxes: dict[str, list[tuple[float, tuple[int, int, int, int]]]] = {}
    box_to_class_index: dict[tuple[int, int, int, int], int] = {}
    for x1, y1, x2, y2 in rects:
        slice_bgr = full_bgr[y1:y2, x1:x2]
        if slice_bgr is None or getattr(slice_bgr, "size", 0) == 0:
            continue
        boxes, inference_ms = _yoloe_predict_boxes(backend, slice_bgr, imgsz=SAHI_IMGSZ)
        inference_ms_total += inference_ms
        for class_idx, conf, (lx1, ly1, lx2, ly2) in boxes:
            absolute_bbox = (
                int(x1 + lx1),
                int(y1 + ly1),
                int(x1 + lx2),
                int(y1 + ly2),
            )
            ref_id = backend.reference_bank.ref_names[class_idx]
            per_ref_boxes.setdefault(ref_id, []).append((conf, absolute_bbox))
            box_to_class_index[absolute_bbox] = class_idx
    deduped = _per_ref_nms(per_ref_boxes, iou_threshold=SAHI_NMS_IOU)

    measurements: dict[str, list[CandidateMeasurement]] = {}
    verify_ms_total = 0.0
    for ref_id, kept in deduped.items():
        for conf, bbox in kept:
            class_idx = box_to_class_index[bbox]
            measurement, verify_ms = _verify_and_score(backend, full_bgr, bbox, class_idx, conf)
            verify_ms_total += verify_ms
            if measurement is not None:
                measurements.setdefault(ref_id, []).append(measurement)
    total_ms = inference_ms_total + verify_ms_total
    return measurements, inference_ms_total, verify_ms_total, total_ms, len(rects)


def _build_row(
    *,
    scenario_id: str,
    frame_idx: int,
    ref_id: str,
    config: str,
    slice_count: int,
    measurements: list[CandidateMeasurement],
    inference_ms: float,
    verify_ms: float,
    total_ms: float,
    score_threshold: float,
    yoloe_conf_threshold: float,
) -> dict[str, Any]:
    if not measurements:
        return {
            "scenario": scenario_id,
            "frame_idx": frame_idx,
            "ref_id": ref_id,
            "config": config,
            "slice_count": slice_count,
            "yoloe_top1_conf": "",
            "yoloe_top1_passes_gate": False,
            "candidate_count_post_gate": 0,
            "lightglue_max_matches": "",
            "ransac_max_inlier_count": "",
            "ransac_max_inlier_ratio": "",
            "final_score_max": "",
            "final_score_passes_threshold": False,
            "yoloe_inference_ms": f"{inference_ms:.4f}",
            "lightglue_verify_ms": f"{verify_ms:.4f}",
            "total_per_frame_ms": f"{total_ms:.4f}",
        }
    top_conf = max(item.yoloe_confidence for item in measurements)
    top1_passes = top_conf >= float(yoloe_conf_threshold)
    max_matches = max(item.match_count for item in measurements)
    max_inliers = max(item.inlier_count for item in measurements)
    max_inlier_ratio = max(item.inlier_ratio for item in measurements)
    max_score = max(item.final_score for item in measurements)
    score_passes = max_score >= float(score_threshold)
    return {
        "scenario": scenario_id,
        "frame_idx": frame_idx,
        "ref_id": ref_id,
        "config": config,
        "slice_count": slice_count,
        "yoloe_top1_conf": f"{top_conf:.6f}",
        "yoloe_top1_passes_gate": top1_passes,
        "candidate_count_post_gate": len(measurements),
        "lightglue_max_matches": int(max_matches),
        "ransac_max_inlier_count": int(max_inliers),
        "ransac_max_inlier_ratio": f"{max_inlier_ratio:.6f}",
        "final_score_max": f"{max_score:.6f}",
        "final_score_passes_threshold": score_passes,
        "yoloe_inference_ms": f"{inference_ms:.4f}",
        "lightglue_verify_ms": f"{verify_ms:.4f}",
        "total_per_frame_ms": f"{total_ms:.4f}",
    }


def _empty_row(
    *,
    scenario_id: str,
    frame_idx: int,
    ref_id: str,
    config: str,
    slice_count: int,
    inference_ms: float,
    verify_ms: float,
    total_ms: float,
) -> dict[str, Any]:
    return {
        "scenario": scenario_id,
        "frame_idx": frame_idx,
        "ref_id": ref_id,
        "config": config,
        "slice_count": slice_count,
        "yoloe_top1_conf": "",
        "yoloe_top1_passes_gate": False,
        "candidate_count_post_gate": 0,
        "lightglue_max_matches": "",
        "ransac_max_inlier_count": "",
        "ransac_max_inlier_ratio": "",
        "final_score_max": "",
        "final_score_passes_threshold": False,
        "yoloe_inference_ms": f"{inference_ms:.4f}",
        "lightglue_verify_ms": f"{verify_ms:.4f}",
        "total_per_frame_ms": f"{total_ms:.4f}",
    }


def _make_settings() -> MvpRuntimeSettings:
    return MvpRuntimeSettings(
        task3_mode="yoloe_vp_lightglue",
        task3_reference_dir=REFERENCE_DIR,
        task3_eval_reference_dir=REFERENCE_DIR,
        task3_yoloe_allow_cpu=True,
    )


def _load_manifest_scenarios() -> list[dict[str, Any]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    scenarios = [
        scenario for scenario in payload.get("scenarios", [])
        if str(scenario.get("id")) in TARGET_SCENARIO_IDS
    ]
    return scenarios


def main() -> int:
    if not is_cv2_available():
        raise RuntimeError("OpenCV gerekli")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = _make_settings()

    cache = ReferenceCache()
    cache.preload_from_directory(REFERENCE_DIR, orb_features=settings.task3_orb_features)
    backend = YoloeVpLightGlueBackend(reference_cache=cache, runtime_settings=settings)
    backend._ensure_ready(list(TARGET_REFERENCE_IDS))
    assert backend.reference_bank is not None and backend.reference_bank.ref_names is not None
    if backend.reference_bank.ref_names != list(TARGET_REFERENCE_IDS):
        raise RuntimeError(
            f"reference bank mismatch: expected {TARGET_REFERENCE_IDS}, got {backend.reference_bank.ref_names}"
        )

    score_threshold = float(settings.task3_yoloe_min_score)
    yoloe_conf_threshold = float(settings.task3_yoloe_conf)

    rows: list[dict[str, Any]] = []
    scenarios = _load_manifest_scenarios()
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        frame_iter = iter_video_frames(
            scenario["video"],
            frame_stride=int(scenario["frame_stride"]),
            limit=int(scenario["frame_limit"]) if scenario.get("frame_limit") is not None else None,
            video_name=scenario_id,
        )
        for decoded_frame in frame_iter:
            if decoded_frame.bgr is None:
                continue

            baseline_measurements, baseline_inf, baseline_ver, baseline_total = _measure_baseline(
                backend, decoded_frame
            )
            (
                sahi_measurements,
                sahi_inf,
                sahi_ver,
                sahi_total,
                sahi_slice_count,
            ) = _measure_sahi(backend, decoded_frame)

            for ref_id in TARGET_REFERENCE_IDS:
                rows.append(
                    _build_row(
                        scenario_id=scenario_id,
                        frame_idx=int(decoded_frame.frame_index),
                        ref_id=ref_id,
                        config="baseline",
                        slice_count=1,
                        measurements=baseline_measurements.get(ref_id, []),
                        inference_ms=baseline_inf,
                        verify_ms=baseline_ver,
                        total_ms=baseline_total,
                        score_threshold=score_threshold,
                        yoloe_conf_threshold=yoloe_conf_threshold,
                    )
                )
                rows.append(
                    _build_row(
                        scenario_id=scenario_id,
                        frame_idx=int(decoded_frame.frame_index),
                        ref_id=ref_id,
                        config="sahi",
                        slice_count=sahi_slice_count,
                        measurements=sahi_measurements.get(ref_id, []),
                        inference_ms=sahi_inf,
                        verify_ms=sahi_ver,
                        total_ms=sahi_total,
                        score_threshold=score_threshold,
                        yoloe_conf_threshold=yoloe_conf_threshold,
                    )
                )

    csv_path = OUTPUT_DIR / "sahi_per_frame_trace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
