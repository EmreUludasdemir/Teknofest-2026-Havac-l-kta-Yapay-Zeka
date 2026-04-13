from __future__ import annotations

import json
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalDetection, DecodedFrame, FrameEnvelope
from src.core.utils import percentile
from src.core.vision import is_cv2_available
from src.evaluation.task1_onnx_validation import (
    discover_task1_validation_video,
    sample_task1_validation_frames,
)
from src.task1.detector import Task1Detector
from src.task1.experimental.rfdetr_support import (
    DEFAULT_MODEL_ROOT,
    ArtifactProbe,
    DatasetConnection,
    TileWindow,
    build_tile_windows,
    discover_local_task1_artifacts,
    discover_python_runtime,
    discover_task1_datasets,
    offset_detection_to_frame,
    probe_optional_dependencies,
)
from src.task1.landing_logic import assign_landing_status
from src.task1.motion_logic import assign_motion_status
from src.task1.postprocess import deduplicate_detections
from src.task1.tracker import Task1Tracker
from src.tools.profiling_harness import evaluate_candidates
from src.tools.vram_monitor import query_vram, recovery_mb

if is_cv2_available():  # pragma: no branch - environment dependent
    from src.core.vision import cv2
else:  # pragma: no cover - cv2 unavailable
    cv2 = None


DEFAULT_VARIANTS = ("baseline_yolo26n", "control_yolo11n", "tiled_yolo26n_probe")
CUSTOM_VARIANT_NAME = "candidate_probe"
CUSTOM_TILED_VARIANT_NAME = "candidate_tiled_probe"
LABEL_AWARE_LABEL_FORMATS = {"coco", "yolo"}
SMALL_OBJECT_AREA_LIMIT = 32.0 * 32.0


@dataclass(slots=True)
class Task1GroundTruth:
    frame_id: str
    class_id: int
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float
    motion_status: int | None = None
    landing_status: int | None = None


@dataclass(slots=True)
class Task1Prediction:
    frame_id: str
    class_id: int
    score: float
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float
    motion_status: int | None = None
    landing_status: int | None = None


@dataclass(frozen=True, slots=True)
class VariantSpec:
    name: str
    mode: str
    candidate_name: str | None
    artifact_path: str | None
    description: str
    sahi_status: str
    tile_aware_training_required: bool
    notes: list[str]


def compute_task1_label_aware_metrics(
    ground_truth: list[Task1GroundTruth],
    predictions: list[Task1Prediction],
    *,
    iou_threshold: float = 0.5,
    small_object_area_limit: float = SMALL_OBJECT_AREA_LIMIT,
) -> dict[str, Any]:
    present_classes = sorted({item.class_id for item in ground_truth})
    per_class_ap: dict[str, float | None] = {}
    class_values: list[float] = []
    for class_id in range(4):
        ap = _compute_average_precision(
            ground_truth,
            predictions,
            iou_threshold=iou_threshold,
            class_filter=class_id,
            area_limit=None,
        )
        per_class_ap[str(class_id)] = ap
        if ap is not None:
            class_values.append(ap)

    small_object_ap = _compute_average_precision(
        ground_truth,
        predictions,
        iou_threshold=iou_threshold,
        class_filter=None,
        area_limit=small_object_area_limit,
    )
    motion_status_ap = _compute_average_precision(
        ground_truth,
        predictions,
        iou_threshold=iou_threshold,
        class_filter=0,
        area_limit=None,
    )
    landing_values = [
        value
        for value in (
            _compute_average_precision(ground_truth, predictions, iou_threshold=iou_threshold, class_filter=2, area_limit=None),
            _compute_average_precision(ground_truth, predictions, iou_threshold=iou_threshold, class_filter=3, area_limit=None),
        )
        if value is not None
    ]
    landing_status_ap = round(sum(landing_values) / len(landing_values), 6) if landing_values else None
    overall = round(sum(class_values) / len(class_values), 6) if class_values else None
    return {
        "overall_mAP": overall,
        "small_object_AP": small_object_ap,
        "per_class_AP": per_class_ap,
        "motion_status_sensitive_performance": motion_status_ap,
        "landing_status_sensitive_performance": landing_status_ap,
        "measured_class_ids": present_classes,
        "unmeasured_class_ids": [class_id for class_id in range(4) if class_id not in present_classes],
    }


def run_task1_rfdetr_comparison(
    *,
    output_dir: str | Path = "reports",
    runtime_settings: MvpRuntimeSettings | None = None,
    sample_count: int = 4,
    variants: list[str] | None = None,
    tile_size: int = 960,
    tile_overlap: float = 0.25,
    dataset_root: str | Path | None = None,
    labels_format: str | None = None,
    candidate_name: str | None = None,
    candidate_path: str | Path | None = None,
    candidate_variant_name: str = CUSTOM_VARIANT_NAME,
) -> dict[str, Any]:
    settings = runtime_settings or MvpRuntimeSettings()
    runtime_probe = discover_python_runtime()
    optional_dependencies = probe_optional_dependencies()
    artifact_probe = discover_local_task1_artifacts(DEFAULT_MODEL_ROOT)
    datasets = discover_task1_datasets(
        explicit_root=Path(dataset_root) if dataset_root else None,
        explicit_labels_format=labels_format,
    )
    requested_variants = list(dict.fromkeys(variants or list(DEFAULT_VARIANTS)))
    if candidate_name and candidate_path and candidate_variant_name not in requested_variants:
        requested_variants.append(candidate_variant_name)
    if candidate_name and candidate_path and CUSTOM_TILED_VARIANT_NAME not in requested_variants and "candidate_tiled_probe" in (variants or []):
        requested_variants.append(CUSTOM_TILED_VARIANT_NAME)
    variant_specs = _resolve_variant_specs(
        requested_variants,
        artifact_probe,
        candidate_name=candidate_name,
        candidate_path=str(candidate_path) if candidate_path else None,
        candidate_variant_name=candidate_variant_name,
    )
    blockers: list[str] = []

    if not runtime_probe.is_repo_compatible:
        blockers.append("no_python_3_10_plus_runtime_found")

    samples = []
    validation_video: str | None = None
    if runtime_probe.is_repo_compatible and any(spec.mode in {"single_yolo", "tiled_yolo"} for spec in variant_specs):
        validation_path = discover_task1_validation_video()
        validation_video = str(validation_path)
        samples = sample_task1_validation_frames(validation_path, sample_count=sample_count)

    variant_payloads: dict[str, Any] = {}
    for spec in variant_specs:
        variant_payloads[spec.name] = _run_variant(
            spec,
            runtime_settings=settings,
            samples=samples,
            validation_video=validation_video,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            runtime_compatible=runtime_probe.is_repo_compatible,
        )

    label_aware_status = _build_label_aware_status(
        datasets=datasets,
        labels_format=labels_format,
        dataset_root=dataset_root,
    )
    if label_aware_status["status"] != "ready":
        blockers.extend(label_aware_status["blockers"])

    payload = _build_payload(
        runtime_probe=runtime_probe.to_dict(),
        optional_dependencies=optional_dependencies,
        artifact_probe=artifact_probe,
        datasets=datasets,
        requested_variants=requested_variants,
        variant_payloads=variant_payloads,
        label_aware_status=label_aware_status,
        blockers=blockers,
    )
    _write_reports(Path(output_dir), payload)
    return payload


def _resolve_variant_specs(
    requested_variants: list[str],
    artifact_probe: ArtifactProbe,
    *,
    candidate_name: str | None = None,
    candidate_path: str | None = None,
    candidate_variant_name: str = CUSTOM_VARIANT_NAME,
) -> list[VariantSpec]:
    variants: list[VariantSpec] = []
    for variant_name in requested_variants:
        if variant_name == CUSTOM_TILED_VARIANT_NAME and candidate_name and candidate_path:
            variants.append(
                VariantSpec(
                    name=variant_name,
                    mode="tiled_yolo",
                    candidate_name=candidate_name,
                    artifact_path=candidate_path,
                    description="Branch-local trained candidate tiled replay.",
                    sahi_status="manual_tiling_probe_only_not_sahi",
                    tile_aware_training_required=False,
                    notes=["Candidate path supplied explicitly by CLI.", "Tile-aware training remains unverified."],
                )
            )
            continue
        if variant_name == candidate_variant_name and candidate_name and candidate_path:
            variants.append(
                VariantSpec(
                    name=variant_name,
                    mode="single_yolo",
                    candidate_name=candidate_name,
                    artifact_path=candidate_path,
                    description="Branch-local trained candidate replay.",
                    sahi_status="not_applicable",
                    tile_aware_training_required=False,
                    notes=["Candidate path supplied explicitly by CLI."],
                )
            )
            continue
        if variant_name == "baseline_yolo26n":
            variants.append(
                VariantSpec(
                    name=variant_name,
                    mode="single_yolo",
                    candidate_name="yolo26n",
                    artifact_path=_first_or_none(artifact_probe.baseline_yolo26n_paths),
                    description="Current production-aligned local YOLO26n control.",
                    sahi_status="not_applicable",
                    tile_aware_training_required=False,
                    notes=["Production defaults are unchanged; this is evaluator-only replay."],
                )
            )
            continue
        if variant_name == "control_yolo11n":
            variants.append(
                VariantSpec(
                    name=variant_name,
                    mode="single_yolo",
                    candidate_name="yolo11n",
                    artifact_path=_first_or_none(artifact_probe.control_yolo11n_paths),
                    description="Alternative YOLO family control on the same sample slices.",
                    sahi_status="not_applicable",
                    tile_aware_training_required=False,
                    notes=["Control-only; no production swap is implied."],
                )
            )
            continue
        if variant_name == "tiled_yolo26n_probe":
            variants.append(
                VariantSpec(
                    name=variant_name,
                    mode="tiled_yolo",
                    candidate_name="yolo26n",
                    artifact_path=_first_or_none(artifact_probe.baseline_yolo26n_paths),
                    description="Manual tiled YOLO26n probe for small-object recall pressure testing.",
                    sahi_status="manual_tiling_probe_only_not_sahi",
                    tile_aware_training_required=False,
                    notes=[
                        "This is not a SAHI success claim.",
                        "Tile-aware training remains unverified.",
                    ],
                )
            )
            continue
        if variant_name == "rtdetr_probe":
            variants.append(
                VariantSpec(
                    name=variant_name,
                    mode="transformer_gate",
                    candidate_name="rtdetr",
                    artifact_path=_first_or_none(artifact_probe.rtdetr_candidate_paths),
                    description="Future transformer detector gate only.",
                    sahi_status="planned_not_verified",
                    tile_aware_training_required=True,
                    notes=["No comparison should be claimed until local artifact and labeled data are connected."],
                )
            )
            continue
        if variant_name == "rfdetr_base":
            variants.append(
                VariantSpec(
                    name=variant_name,
                    mode="transformer_gate",
                    candidate_name="rfdetr_base",
                    artifact_path=_first_or_none(artifact_probe.rfdetr_candidate_paths),
                    description="Future RF-DETR-Base gate only.",
                    sahi_status="planned_not_verified",
                    tile_aware_training_required=True,
                    notes=["No comparison should be claimed until local artifact and labeled data are connected."],
                )
            )
            continue
        raise ValueError(f"unsupported_variant:{variant_name}")
    return variants


def _run_variant(
    spec: VariantSpec,
    *,
    runtime_settings: MvpRuntimeSettings,
    samples: list[Any],
    validation_video: str | None,
    tile_size: int,
    tile_overlap: float,
    runtime_compatible: bool,
) -> dict[str, Any]:
    payload = {
        "description": spec.description,
        "mode": spec.mode,
        "candidate_name": spec.candidate_name,
        "artifact_path": spec.artifact_path,
        "status": "not_run",
        "available": bool(spec.artifact_path),
        "sahi_status": spec.sahi_status,
        "tile_aware_training_required": spec.tile_aware_training_required,
        "notes": list(spec.notes),
        "issues": [],
        "runtime_profile": None,
        "proxy_metrics": None,
        "label_aware_metrics": None,
    }

    if not runtime_compatible:
        payload["status"] = "blocked_runtime"
        payload["issues"].append("no_python_3_10_plus_runtime_found")
        return payload

    if spec.mode == "transformer_gate":
        payload["status"] = "blocked_missing_artifact" if not spec.artifact_path else "blocked_missing_label_aware_dataset"
        if not spec.artifact_path:
            payload["issues"].append(f"no_local_{spec.name}_artifact_found")
        else:
            payload["issues"].append("no_supported_labeled_dataset_found")
        return payload

    if not spec.artifact_path:
        payload["status"] = "blocked_missing_artifact"
        payload["issues"].append(f"no_local_{spec.name}_artifact_found")
        return payload

    if spec.mode == "single_yolo":
        payload["runtime_profile"] = _run_variant_profile(
            runtime_settings,
            candidate_name=spec.candidate_name or "unknown",
            model_path=spec.artifact_path,
        )
    payload["proxy_metrics"] = _run_variant_proxy_replay(
        runtime_settings,
        spec=spec,
        samples=samples,
        validation_video=validation_video,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    payload["issues"].extend(_derive_variant_issues(payload["proxy_metrics"]))
    payload["status"] = "proxy_only"
    return payload


def _run_variant_profile(
    runtime_settings: MvpRuntimeSettings,
    *,
    candidate_name: str,
    model_path: str,
) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory() as temp_dir:
        payload = evaluate_candidates(
            runtime_settings=replace(
                runtime_settings,
                task1_model_runtime="ultralytics",
                task1_detector_backend=candidate_name,
                task1_env_name="py312",
            ),
            output_dir=temp_dir,
            mode="smoke",
            task="task1",
            candidate=candidate_name,
            model_path=model_path,
        )
    results = payload.get("results", [])
    return results[0] if results else None


def _run_variant_proxy_replay(
    runtime_settings: MvpRuntimeSettings,
    *,
    spec: VariantSpec,
    samples: list[Any],
    validation_video: str | None,
    tile_size: int,
    tile_overlap: float,
) -> dict[str, Any]:
    detector_settings = replace(
        runtime_settings,
        task1_detector_backend=spec.candidate_name or "synthetic",
        task1_model_runtime="ultralytics",
        task1_candidate_paths={
            **dict(runtime_settings.task1_candidate_paths),
            **({spec.candidate_name: spec.artifact_path} if spec.candidate_name and spec.artifact_path else {}),
        },
    )
    detector = Task1Detector(runtime_settings=detector_settings)
    tracker = Task1Tracker()
    latencies_ms: list[float] = []
    aggregate_histogram: Counter[int] = Counter()
    runtime_identifiers: Counter[str] = Counter()
    bbox_areas: list[float] = []
    zero_detection_frames = 0
    total_pre_dedup = 0
    total_post_dedup = 0
    tile_raw_total = 0
    tile_merged_total = 0
    tile_count_total = 0
    used_fallback = False
    frame_rows: list[dict[str, Any]] = []

    before_run = query_vram(runtime_settings.profiling_gpu_query_cmd)
    peak_vram_mb = before_run.used_mb
    try:
        for sample in samples:
            started_at = perf_counter()
            if spec.mode == "tiled_yolo":
                detections, tile_stats = _run_tiled_detector_on_frame(
                    detector,
                    sample.frame,
                    sample.decoded_frame,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    runtime_settings=runtime_settings,
                )
                tile_raw_total += int(tile_stats["raw_detection_count"])
                tile_merged_total += int(tile_stats["merged_detection_count"])
                tile_count_total += int(tile_stats["tile_count"])
            else:
                detections = detector.detect(sample.frame, sample.image_bytes, decoded_frame=sample.decoded_frame)
                tile_stats = None

            staged = tracker.update(sample.frame, detections)
            staged = assign_motion_status(
                staged,
                sample.frame,
                threshold_px=runtime_settings.task1_motion_threshold_px,
            )
            staged = assign_landing_status(
                staged,
                sample.frame,
                decoded_frame=sample.decoded_frame,
                margin_px=runtime_settings.task1_landing_margin_px,
            )
            total_pre_dedup += len(staged)
            final = deduplicate_detections(
                staged,
                max_objects_per_frame=runtime_settings.max_objects_per_frame,
                iou_threshold=runtime_settings.task1_iou_threshold,
            )
            total_post_dedup += len(final)
            latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
            latencies_ms.append(latency_ms)

            histogram = Counter(int(item.class_id) for item in final)
            aggregate_histogram.update(histogram)
            if not final:
                zero_detection_frames += 1
            for detection in final:
                bbox_areas.append(_area(detection))
                runtime_identifier = str(
                    detection.metadata.get("runtime_device")
                    or detection.metadata.get("active_backend_name")
                    or detection.metadata.get("backend_name")
                    or "unknown"
                )
                runtime_identifiers.update([runtime_identifier])
                if detection.metadata.get("backend_unavailable"):
                    used_fallback = True
            frame_rows.append(
                {
                    "frame_index": sample.decoded_frame.frame_index,
                    "bbox_count": len(final),
                    "class_histogram": {str(key): int(value) for key, value in sorted(histogram.items())},
                    "latency_ms": latency_ms,
                    "tile_stats": tile_stats,
                }
            )
            current_snapshot = query_vram(runtime_settings.profiling_gpu_query_cmd)
            if current_snapshot.used_mb is not None:
                peak_vram_mb = max(peak_vram_mb or current_snapshot.used_mb, current_snapshot.used_mb)
    finally:
        before_unload = query_vram(runtime_settings.profiling_gpu_query_cmd)
        detector.unload()
        after_unload = query_vram(runtime_settings.profiling_gpu_query_cmd)

    small_box_count = sum(1 for value in bbox_areas if value <= SMALL_OBJECT_AREA_LIMIT)
    return {
        "validation_video": validation_video,
        "sample_count": len(samples),
        "runtime_identifier": runtime_identifiers.most_common(1)[0][0] if runtime_identifiers else "unknown",
        "latency_mean_ms": round(sum(latencies_ms) / len(latencies_ms), 6) if latencies_ms else None,
        "latency_p50_ms": round(percentile(latencies_ms, 50.0), 6) if latencies_ms else None,
        "latency_p95_ms": round(percentile(latencies_ms, 95.0), 6) if latencies_ms else None,
        "peak_vram_mb": peak_vram_mb,
        "unload_recovery_mb": recovery_mb(before_unload, after_unload),
        "aggregate_class_histogram": {str(key): int(value) for key, value in sorted(aggregate_histogram.items())},
        "zero_detection_frames": zero_detection_frames,
        "zero_detection_rate": round(zero_detection_frames / max(len(samples), 1), 6),
        "detections_before_dedup": total_pre_dedup,
        "detections_after_dedup": total_post_dedup,
        "duplicate_ratio": round(max(total_pre_dedup - total_post_dedup, 0) / max(total_pre_dedup, 1), 6),
        "bbox_area_distribution": _summarize_areas(bbox_areas),
        "small_box_count": small_box_count,
        "small_box_ratio": round(small_box_count / max(len(bbox_areas), 1), 6),
        "used_fallback": used_fallback,
        "tile_probe": {
            "tile_count_total": tile_count_total,
            "raw_detection_count": tile_raw_total,
            "merged_detection_count": tile_merged_total,
            "tile_merge_duplicate_ratio": round(max(tile_raw_total - tile_merged_total, 0) / max(tile_raw_total, 1), 6),
        }
        if spec.mode == "tiled_yolo"
        else None,
        "frames": frame_rows,
    }


def _run_tiled_detector_on_frame(
    detector: Task1Detector,
    frame: FrameEnvelope,
    decoded_frame: DecodedFrame,
    *,
    tile_size: int,
    tile_overlap: float,
    runtime_settings: MvpRuntimeSettings,
) -> tuple[list[CanonicalDetection], dict[str, Any]]:
    if cv2 is None or decoded_frame.bgr is None:
        raise RuntimeError("cv2_and_decoded_frame_required_for_tiled_probe")

    tile_windows = build_tile_windows(
        decoded_frame.width,
        decoded_frame.height,
        tile_size=tile_size,
        overlap=tile_overlap,
    )
    tiled_detections: list[CanonicalDetection] = []
    for index, tile_window in enumerate(tile_windows):
        tile_frame, tile_decoded = _slice_tile(frame, decoded_frame, tile_window, index=index)
        tile_outputs = detector.detect(tile_frame, b"", decoded_frame=tile_decoded)
        tiled_detections.extend(
            offset_detection_to_frame(
                item,
                x_offset=tile_window.left,
                y_offset=tile_window.top,
                frame_width=decoded_frame.width,
                frame_height=decoded_frame.height,
            )
            for item in tile_outputs
        )

    merged = deduplicate_detections(
        tiled_detections,
        max_objects_per_frame=max(runtime_settings.max_objects_per_frame * 4, 64),
        iou_threshold=runtime_settings.task1_iou_threshold,
    )
    return merged, {
        "tile_count": len(tile_windows),
        "raw_detection_count": len(tiled_detections),
        "merged_detection_count": len(merged),
    }


def _slice_tile(
    frame: FrameEnvelope,
    decoded_frame: DecodedFrame,
    tile_window: TileWindow,
    *,
    index: int,
) -> tuple[FrameEnvelope, DecodedFrame]:
    bgr = decoded_frame.bgr[tile_window.top:tile_window.bottom, tile_window.left:tile_window.right].copy()
    gray = None
    if decoded_frame.gray is not None:
        gray = decoded_frame.gray[tile_window.top:tile_window.bottom, tile_window.left:tile_window.right].copy()
    tile_frame = FrameEnvelope(
        frame_url=f"{frame.frame_url}#tile-{index}",
        image_url=frame.image_url,
        video_name=frame.video_name,
        translation_x=frame.translation_x,
        translation_y=frame.translation_y,
        translation_z=frame.translation_z,
        health_status=frame.health_status,
        metadata={
            **dict(frame.metadata),
            "image_width": tile_window.width,
            "image_height": tile_window.height,
            "tile_window": tile_window.to_dict(),
        },
    )
    tile_decoded = DecodedFrame(
        bgr=bgr,
        gray=gray,
        width=int(tile_window.width),
        height=int(tile_window.height),
        channel_count=int(decoded_frame.channel_count),
        modality=decoded_frame.modality,
        frame_index=int(decoded_frame.frame_index),
        metadata={**dict(decoded_frame.metadata), "tile_window": tile_window.to_dict()},
    )
    return tile_frame, tile_decoded


def _build_label_aware_status(
    *,
    datasets: list[DatasetConnection],
    labels_format: str | None,
    dataset_root: str | Path | None,
) -> dict[str, Any]:
    connected_parsable = [
        item
        for item in datasets
        if item.supports_label_parsing and (item.labels_format in LABEL_AWARE_LABEL_FORMATS or item.labels_format == "visdrone")
    ]
    if not datasets:
        return {
            "status": "blocked_by_labels",
            "dataset_root": None,
            "labels_format": labels_format,
            "results": None,
            "notes": [
                "No local labeled aerial dataset is connected.",
                "Proxy comparison is the only honest result currently available.",
            ],
            "blockers": ["no_supported_labeled_dataset_found"],
        }
    if dataset_root and labels_format and labels_format.lower() in LABEL_AWARE_LABEL_FORMATS:
        return {
            "status": "adapter_ready_dataset_unrun",
            "dataset_root": str(dataset_root),
            "labels_format": labels_format.lower(),
            "results": None,
            "notes": [
                "Explicit labeled dataset root is connected.",
                "Real label-aware replay remains disabled until a truthful side-by-side detector comparison is available.",
            ],
            "blockers": [],
        }
    if connected_parsable:
        return {
            "status": "adapter_ready_dataset_unrun",
            "dataset_root": connected_parsable[0].root,
            "labels_format": connected_parsable[0].labels_format,
            "results": None,
            "notes": [
                "A parsable dataset root was discovered.",
                "This branch still lacks the truthful detector-side conditions for a real label-aware comparison run.",
            ],
            "blockers": [],
        }
    return {
        "status": "blocked_by_labels",
        "dataset_root": str(dataset_root) if dataset_root else None,
        "labels_format": labels_format,
        "results": None,
        "notes": [
            "A dataset root may exist, but no supported label parser was confirmed.",
        ],
        "blockers": ["no_supported_labeled_dataset_found"],
    }


def _build_proxy_summary(variants: dict[str, Any]) -> dict[str, Any]:
    proxy_ready = {
        name: payload
        for name, payload in variants.items()
        if payload.get("status") == "proxy_only" and payload.get("proxy_metrics") is not None
    }
    if not proxy_ready:
        return {
            "status": "no_proxy_runs",
            "best_latency_variant": None,
            "lowest_zero_detection_variant": None,
            "highest_small_box_variant": None,
            "comparisons": {},
        }

    best_latency_variant = min(
        proxy_ready.items(),
        key=lambda item: float(item[1]["proxy_metrics"].get("latency_p95_ms") or 10**9),
    )[0]
    lowest_zero_variant = min(
        proxy_ready.items(),
        key=lambda item: float(item[1]["proxy_metrics"].get("zero_detection_rate") or 10**9),
    )[0]
    highest_small_box_variant = max(
        proxy_ready.items(),
        key=lambda item: int(item[1]["proxy_metrics"].get("small_box_count") or 0),
    )[0]
    highest_small_box_count = int(proxy_ready[highest_small_box_variant]["proxy_metrics"].get("small_box_count") or 0)
    if highest_small_box_count <= 0:
        highest_small_box_variant = None
    comparisons: dict[str, Any] = {}
    baseline = proxy_ready.get("baseline_yolo26n")
    control_outcome = None
    tiled_probe_outcome = None
    if baseline is not None:
        baseline_metrics = baseline["proxy_metrics"]
        for name, payload in proxy_ready.items():
            if name == "baseline_yolo26n":
                continue
            metrics = payload["proxy_metrics"]
            comparison = {
                "zero_detection_rate_delta": _delta(metrics.get("zero_detection_rate"), baseline_metrics.get("zero_detection_rate")),
                "small_box_count_delta": _delta(metrics.get("small_box_count"), baseline_metrics.get("small_box_count")),
                "duplicate_ratio_delta": _delta(metrics.get("duplicate_ratio"), baseline_metrics.get("duplicate_ratio")),
                "latency_p95_ratio": _ratio(metrics.get("latency_p95_ms"), baseline_metrics.get("latency_p95_ms")),
            }
            comparisons[f"{name}_vs_baseline_yolo26n"] = comparison
            if name == "control_yolo11n":
                if (
                    comparison["zero_detection_rate_delta"] is not None
                    and comparison["zero_detection_rate_delta"] < 0.0
                    and (comparison["latency_p95_ratio"] or 10.0) <= 1.0
                ):
                    control_outcome = "proxy_better_than_baseline_on_this_slice"
                else:
                    control_outcome = "no_clear_proxy_advantage"
            if name == "tiled_yolo26n_probe":
                if (
                    comparison["zero_detection_rate_delta"] is not None
                    and comparison["zero_detection_rate_delta"] <= 0.0
                    and comparison["small_box_count_delta"] is not None
                    and comparison["small_box_count_delta"] > 0.0
                    and (comparison["latency_p95_ratio"] or 10.0) <= 2.0
                ):
                    tiled_probe_outcome = "proxy_promising"
                else:
                    tiled_probe_outcome = "no_proxy_gain_or_latency_regression"
    return {
        "status": "proxy_only",
        "best_latency_variant": best_latency_variant,
        "lowest_zero_detection_variant": lowest_zero_variant,
        "highest_small_box_variant": highest_small_box_variant,
        "control_outcome": control_outcome,
        "tiled_probe_outcome": tiled_probe_outcome,
        "comparisons": comparisons,
    }


def _build_payload(
    *,
    runtime_probe: dict[str, Any],
    optional_dependencies: dict[str, bool],
    artifact_probe: ArtifactProbe,
    datasets: list[DatasetConnection],
    requested_variants: list[str],
    variant_payloads: dict[str, Any],
    label_aware_status: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    proxy_summary = _build_proxy_summary(variant_payloads)
    return {
        "date": date.today().isoformat(),
        "branch": "feature/task1-rfdetr-sahi",
        "decision": "EXPERIMENTAL ONLY",
        "truth_backbone": {
            "task1_status": "active_score_up_branch",
            "task2_status": "FROZEN_NO_MERGE_CANDIDATE",
            "task3_status": "WEIGHT_MISSING",
            "stale_reports": [
                "reports/scoreup_priority_summary.md",
                "reports/task3_weight_watchpoint.md",
            ],
        },
        "environment": {
            "python_runtime": runtime_probe,
            "optional_dependencies": optional_dependencies,
            "teknofest_gpu_venv": _probe_teknofest_gpu_venv(),
        },
        "artifacts": artifact_probe.to_dict(),
        "datasets": [item.to_dict() for item in datasets],
        "requested_variants": requested_variants,
        "variants": variant_payloads,
        "proxy_summary": proxy_summary,
        "label_aware_comparison": label_aware_status,
        "comparison_path": {
            "baseline_runtime_harness": "src/tools/profiling_harness.py::evaluate_candidates",
            "sample_replay": "src/evaluation/task1_rfdetr_comparison.py::_run_variant_proxy_replay",
            "post_detector_logic": [
                "src/task1/tracker.py::Task1Tracker",
                "src/task1/motion_logic.py::assign_motion_status",
                "src/task1/landing_logic.py::assign_landing_status",
                "src/task1/postprocess.py::deduplicate_detections",
            ],
            "manual_tiling_probe": "src/evaluation/task1_rfdetr_comparison.py::_run_tiled_detector_on_frame",
            "label_aware_metric_function": "src/evaluation/task1_rfdetr_comparison.py::compute_task1_label_aware_metrics",
        },
        "recommended_paths": {
            "short_term": "YOLO-first proxy comparison and tiled small-object probe",
            "medium_term": "Labeled aerial dataset hookup, then label-aware YOLO tiled evaluation before transformer detector retry",
            "risk_now": "RF-DETR/RT-DETR-first pivot remains blocked by missing local artifact and labeled dataset",
        },
        "blockers": _dedupe(_variant_blockers(variant_payloads) + blockers),
    }


def _variant_blockers(variants: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for payload in variants.values():
        blockers.extend(payload.get("issues", []))
    return blockers


def _derive_variant_issues(proxy_metrics: dict[str, Any] | None) -> list[str]:
    if proxy_metrics is None:
        return []
    issues: list[str] = []
    if proxy_metrics.get("used_fallback"):
        issues.append("detector_fallback_used")
    if bool(proxy_metrics.get("zero_detection_rate") == 1.0):
        issues.append("all_sample_frames_zero_detection")
    histogram = proxy_metrics.get("aggregate_class_histogram") or {}
    if "2" not in histogram and "3" not in histogram:
        issues.append("no_uap_uai_predictions_seen_in_proxy_run")
    return issues


def _write_reports(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task1_rfdetr_vs_yolo26n.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "task1_rfdetr_vs_yolo26n.md").write_text(render_comparison_report(payload), encoding="utf-8")
    (output_dir / "task1_rfdetr_risks.md").write_text(render_risks_report(payload), encoding="utf-8")


def render_comparison_report(payload: dict[str, Any]) -> str:
    runtime = payload["environment"]["python_runtime"]
    deps = payload["environment"]["optional_dependencies"]
    venv_state = payload["environment"]["teknofest_gpu_venv"]
    variants = payload["variants"]
    datasets = payload["datasets"]
    proxy_summary = payload["proxy_summary"]
    lines = [
        "# Task 1 YOLO-First Experimental Comparison",
        "",
        f"**Date:** {payload['date']}",
        f"**Branch:** `{payload['branch']}`",
        f"**Decision:** `{payload['decision']}`",
        "",
        "## Current Truth",
        "",
        "- Task 1 is the only active score-up branch.",
        "- Task 2 remains frozen with no merge candidate.",
        "- Task 3 remains blocked by missing real YOLOE weight.",
        "- Short-term direction on this branch is YOLO-first proxy comparison, not RF-DETR-first swap.",
        "",
        "## Runtime and Dependency State",
        "",
        f"- Selected Python: `{runtime.get('selected_path')}`",
        f"- Version: `{runtime.get('version')}`",
        f"- Repo compatible: `{runtime.get('is_repo_compatible')}`",
        f"- `teknofest-gpu` venv status: `{venv_state.get('status')}`",
        f"- {venv_state.get('note')}",
        f"- Optional dependencies: `{deps}`",
        "",
        "## Local Artefact Truth",
        "",
        f"- `yolo26n` count: `{len(payload['artifacts'].get('baseline_yolo26n_paths', []))}`",
        f"- `yolo11n` count: `{len(payload['artifacts'].get('control_yolo11n_paths', []))}`",
        f"- `rtdetr` count: `{len(payload['artifacts'].get('rtdetr_candidate_paths', []))}`",
        f"- `rfdetr` count: `{len(payload['artifacts'].get('rfdetr_candidate_paths', []))}`",
        "",
        "## Dataset Truth",
        "",
        f"- Connected dataset roots: `{len(datasets)}`",
    ]
    if datasets:
        lines.extend(
            f"- `{dataset['name']}` via `{dataset['adapter_name']}` covers classes {dataset['canonical_class_coverage']}: {dataset['notes']}"
            for dataset in datasets
        )
    else:
        lines.append("- No local labeled aerial dataset is connected. Real AP comparison remains blocked.")

    lines.extend(["", "## Variant Results", ""])
    for variant_name, variant in variants.items():
        proxy = variant.get("proxy_metrics") or {}
        profile = variant.get("runtime_profile") or {}
        lines.extend(
            [
                f"### `{variant_name}`",
                "",
                f"- status: `{variant.get('status')}`",
                f"- candidate: `{variant.get('candidate_name')}`",
                f"- artifact: `{variant.get('artifact_path')}`",
                f"- mode: `{variant.get('mode')}`",
                f"- sahi status: `{variant.get('sahi_status')}`",
                f"- runtime profile p50 ms: `{profile.get('warm_p50_latency_ms')}`",
                f"- runtime profile p95 ms: `{profile.get('warm_p95_latency_ms')}`",
                f"- proxy replay p50 ms: `{proxy.get('latency_p50_ms')}`",
                f"- proxy replay p95 ms: `{proxy.get('latency_p95_ms')}`",
                f"- zero-detection rate: `{proxy.get('zero_detection_rate')}`",
                f"- duplicate ratio: `{proxy.get('duplicate_ratio')}`",
                f"- small box count: `{proxy.get('small_box_count')}`",
                f"- class histogram: `{proxy.get('aggregate_class_histogram')}`",
            ]
        )
        if proxy.get("tile_probe") is not None:
            lines.append(f"- tile probe: `{proxy.get('tile_probe')}`")
        if variant.get("issues"):
            lines.append(f"- issues: `{variant.get('issues')}`")
        lines.append("")

    lines.extend(
        [
            "## Proxy Summary",
            "",
            f"- best latency variant: `{proxy_summary.get('best_latency_variant')}`",
            f"- lowest zero-detection variant: `{proxy_summary.get('lowest_zero_detection_variant')}`",
            f"- highest small-box variant: `{proxy_summary.get('highest_small_box_variant')}`",
            f"- control outcome: `{proxy_summary.get('control_outcome')}`",
            f"- tiled probe outcome: `{proxy_summary.get('tiled_probe_outcome')}`",
            f"- comparisons: `{proxy_summary.get('comparisons')}`",
            "",
            "## Label-Aware State",
            "",
            f"- status: `{payload['label_aware_comparison']['status']}`",
            f"- dataset root: `{payload['label_aware_comparison']['dataset_root']}`",
            f"- labels format: `{payload['label_aware_comparison']['labels_format']}`",
        ]
    )
    lines.extend(f"- {item}" for item in payload["label_aware_comparison"]["notes"])
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "`EXPERIMENTAL ONLY`",
            "",
            "Reason:",
            "",
            "- YOLO-first proxy comparison is now runnable on real local artefacts.",
            "- Manual tiling is available as an evaluator-only small-object probe.",
            "- Real label-aware AP comparison is still blocked by missing labeled aerial data.",
            "- RT-DETR / RF-DETR remain gate-only because local artefacts are absent.",
            "",
        ]
    )
    return "\n".join(lines)


def render_risks_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Task 1 Experimental Risks",
        "",
        f"**Date:** {payload['date']}",
        f"**Branch:** `{payload['branch']}`",
        "",
        "## Active Risks",
        "",
        "### 1. Proxy metrics are not AP evidence",
        "",
        "- Zero-detection rate, duplicate ratio and small-box count are only directional probes on unlabeled sample video.",
        "- They cannot justify a merge decision on their own.",
        "",
        "### 2. Labeled aerial dataset is still missing",
        "",
    ]
    if payload["datasets"]:
        lines.extend(
            f"- `{item['name']}` exists but current branch still lacks a truthful label-aware run: {item['notes']}"
            for item in payload["datasets"]
        )
    else:
        lines.append("- No local VisDrone/UAVDT or explicit labeled dataset is connected.")
    lines.extend(
        [
            "",
            "### 3. Tiled probe may improve recall proxy at a latency / duplicate cost",
            "",
            "- Manual tiling is intentionally branch-local and evaluator-only.",
            "- It must not be sold as SAHI success or production-ready behavior.",
            "",
            "### 4. Transformer path remains blocked",
            "",
            "- RT-DETR / RF-DETR artefacts are not locally connected.",
            "- Any transformer-first claim would still be fabricated.",
            "",
            "### 5. UAP/UAI coverage is still weakly evidenced",
            "",
            "- Current local smoke path and sample replay only validate that classes 0/1 are exercised.",
            "- UAP/UAI performance remains an open risk until labeled data is connected.",
            "",
            "## Current Blocker List",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload.get("blockers", []))
    lines.append("")
    return "\n".join(lines)


def _probe_teknofest_gpu_venv() -> dict[str, Any]:
    candidate = Path(r"C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe")
    if not candidate.exists():
        return {
            "status": "missing",
            "note": "The teknofest-gpu venv entry point is not present.",
        }
    try:
        completed = subprocess.run(
            [str(candidate), "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10.0,
        )
        return {
            "status": "callable",
            "path": completed.stdout.strip(),
            "note": "The teknofest-gpu venv entry point is callable from the repaired Python 3.12 environment.",
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "status": "broken",
            "note": f"The teknofest-gpu venv entry point exists but failed from this shell: {exc.__class__.__name__}.",
        }


def _compute_average_precision(
    ground_truth: list[Task1GroundTruth],
    predictions: list[Task1Prediction],
    *,
    iou_threshold: float,
    class_filter: int | None,
    area_limit: float | None,
) -> float | None:
    filtered_gt = [
        item
        for item in ground_truth
        if (class_filter is None or item.class_id == class_filter) and (area_limit is None or _area(item) <= area_limit)
    ]
    filtered_predictions = [item for item in predictions if class_filter is None or item.class_id == class_filter]
    if not filtered_gt:
        return None
    if not filtered_predictions:
        return 0.0

    gt_by_frame: dict[str, list[Task1GroundTruth]] = {}
    for item in filtered_gt:
        gt_by_frame.setdefault(item.frame_id, []).append(item)
    matched: set[tuple[str, int]] = set()
    sorted_predictions = sorted(filtered_predictions, key=lambda item: item.score, reverse=True)
    true_positive: list[int] = []
    false_positive: list[int] = []

    for prediction in sorted_predictions:
        frame_ground_truth = gt_by_frame.get(prediction.frame_id, [])
        best_index = -1
        best_iou = 0.0
        for index, gt_item in enumerate(frame_ground_truth):
            if (prediction.frame_id, index) in matched:
                continue
            if gt_item.class_id != prediction.class_id:
                continue
            current_iou = _iou(prediction, gt_item)
            if current_iou < iou_threshold:
                continue
            if not _status_matches(prediction, gt_item):
                continue
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index
        if best_index >= 0:
            matched.add((prediction.frame_id, best_index))
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)

    tp_cumulative: list[int] = []
    fp_cumulative: list[int] = []
    tp_running = 0
    fp_running = 0
    for tp_item, fp_item in zip(true_positive, false_positive):
        tp_running += tp_item
        fp_running += fp_item
        tp_cumulative.append(tp_running)
        fp_cumulative.append(fp_running)
    recall = [value / len(filtered_gt) for value in tp_cumulative]
    precision = [tp / max(tp + fp, 1) for tp, fp in zip(tp_cumulative, fp_cumulative)]
    return _area_under_pr_curve(recall, precision)


def _status_matches(prediction: Task1Prediction, ground_truth: Task1GroundTruth) -> bool:
    if ground_truth.class_id == 0 and ground_truth.motion_status is not None:
        return prediction.motion_status == ground_truth.motion_status
    if ground_truth.class_id in (2, 3) and ground_truth.landing_status is not None:
        return prediction.landing_status == ground_truth.landing_status
    return True


def _iou(prediction: Task1Prediction, ground_truth: Task1GroundTruth) -> float:
    ix1 = max(prediction.top_left_x, ground_truth.top_left_x)
    iy1 = max(prediction.top_left_y, ground_truth.top_left_y)
    ix2 = min(prediction.bottom_right_x, ground_truth.bottom_right_x)
    iy2 = min(prediction.bottom_right_y, ground_truth.bottom_right_y)
    width = max(ix2 - ix1, 0.0)
    height = max(iy2 - iy1, 0.0)
    intersection = width * height
    if intersection <= 0.0:
        return 0.0
    prediction_area = _area(prediction)
    ground_truth_area = _area(ground_truth)
    union = max(prediction_area + ground_truth_area - intersection, 1e-9)
    return float(intersection / union)


def _area(item: Any) -> float:
    return max(float(item.bottom_right_x) - float(item.top_left_x), 0.0) * max(
        float(item.bottom_right_y) - float(item.top_left_y),
        0.0,
    )


def _area_under_pr_curve(recall: list[float], precision: list[float]) -> float:
    if not recall or not precision:
        return 0.0
    padded_recall = [0.0, *recall, 1.0]
    padded_precision = [precision[0], *precision, 0.0]
    for index in range(len(padded_precision) - 2, -1, -1):
        padded_precision[index] = max(padded_precision[index], padded_precision[index + 1])
    area = 0.0
    for index in range(1, len(padded_recall)):
        delta = padded_recall[index] - padded_recall[index - 1]
        if delta <= 0:
            continue
        area += delta * padded_precision[index]
    return round(area, 6)


def _summarize_areas(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(float(ordered[0]), 6),
        "p25": round(percentile(ordered, 25.0), 6),
        "p50": round(percentile(ordered, 50.0), 6),
        "p75": round(percentile(ordered, 75.0), 6),
        "max": round(float(ordered[-1]), 6),
    }


def _delta(current: Any, baseline: Any) -> float | None:
    if current is None or baseline is None:
        return None
    return round(float(current) - float(baseline), 6)


def _ratio(current: Any, baseline: Any) -> float | None:
    if current is None or baseline in (None, 0, 0.0):
        return None
    return round(float(current) / float(baseline), 6)


def _dedupe(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _first_or_none(values: list[str]) -> str | None:
    return values[0] if values else None
