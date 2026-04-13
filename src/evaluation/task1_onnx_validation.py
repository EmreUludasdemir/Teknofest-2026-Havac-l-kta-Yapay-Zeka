from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.core.logger import StructuredLogger
from src.core.utils import percentile
from src.core.vision import is_cv2_available
from src.exports.export_onnx import export_task1_candidate_to_onnx
from src.task1.detector import Task1Detector
from src.task1.landing_logic import assign_landing_status
from src.task1.motion_logic import assign_motion_status
from src.task1.postprocess import deduplicate_detections
from src.task1.tracker import Task1Tracker
from src.tools.report_paths import EXPORT_REPORTS_DIR
from src.tools.vram_monitor import query_vram, recovery_mb

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2
else:  # pragma: no cover - cv2 yoksa
    cv2 = None


@dataclass(slots=True)
class ValidationFrameSample:
    frame: FrameEnvelope
    image_bytes: bytes
    decoded_frame: DecodedFrame


def discover_task1_validation_video(root_dir: str | Path | None = None) -> Path:
    root = Path(root_dir) if root_dir is not None else Path("data")
    explicit = next(iter(root.rglob("THYZ_2026_Ornek_Veri_1.MP4")), None)
    if explicit is not None:
        return explicit
    candidates = sorted(root.rglob("*.MP4"))
    if not candidates:
        raise FileNotFoundError("task1_validation_video_missing")
    return candidates[0]


def sample_task1_validation_frames(
    video_path: str | Path,
    *,
    sample_count: int,
) -> list[ValidationFrameSample]:
    if not is_cv2_available():
        raise RuntimeError("cv2_required_for_task1_onnx_validation")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"video_open_failed:{video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"video_frame_count_invalid:{video_path}")

    if sample_count <= 1:
        target_indices = [0]
    else:
        target_indices = sorted(
            {
                min(int(round(step * (total_frames - 1) / float(sample_count - 1))), max(total_frames - 1, 0))
                for step in range(sample_count)
            }
        )

    samples: list[ValidationFrameSample] = []
    try:
        frame_index = 0
        target_pointer = 0
        current_target = target_indices[target_pointer] if target_indices else None
        while current_target is not None:
            ok, image = capture.read()
            if not ok:
                break
            if frame_index != current_target:
                frame_index += 1
                continue
            success, encoded = cv2.imencode(".png", image)
            if success:
                height, width = image.shape[:2]
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                frame = FrameEnvelope(
                    frame_url=f"http://task1-export-validation/frames/{frame_index + 1}/",
                    image_url=f"/task1-export/{frame_index + 1}.png",
                    video_name=Path(video_path).stem,
                    translation_x=0.0,
                    translation_y=0.0,
                    translation_z=0.0,
                    health_status="1",
                    metadata={"frame_index": frame_index, "image_width": width, "image_height": height},
                )
                samples.append(
                    ValidationFrameSample(
                        frame=frame,
                        image_bytes=encoded.tobytes(),
                        decoded_frame=DecodedFrame(
                            bgr=image,
                            gray=gray,
                            width=int(width),
                            height=int(height),
                            channel_count=int(image.shape[2]) if len(image.shape) == 3 else 1,
                            modality="rgb",
                            frame_index=frame_index,
                        ),
                    )
                )
            target_pointer += 1
            current_target = target_indices[target_pointer] if target_pointer < len(target_indices) else None
            frame_index += 1
    finally:
        capture.release()
    return samples


def run_task1_pipeline_on_samples(
    runtime_settings: MvpRuntimeSettings,
    samples: list[ValidationFrameSample],
) -> dict[str, Any]:
    detector = Task1Detector(runtime_settings=runtime_settings)
    tracker = Task1Tracker()
    frame_results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    aggregate_histogram: Counter[int] = Counter()
    before_run = query_vram(runtime_settings.profiling_gpu_query_cmd)
    peak_vram_mb = before_run.used_mb

    for sample in samples:
        started_at = perf_counter()
        detections = detector.detect(
            sample.frame,
            sample.image_bytes,
            decoded_frame=sample.decoded_frame,
        )
        detections = tracker.update(sample.frame, detections)
        detections = assign_motion_status(
            detections,
            sample.frame,
            threshold_px=runtime_settings.task1_motion_threshold_px,
        )
        detections = assign_landing_status(
            detections,
            sample.frame,
            decoded_frame=sample.decoded_frame,
            margin_px=runtime_settings.task1_landing_margin_px,
        )
        detections = deduplicate_detections(
            detections,
            max_objects_per_frame=runtime_settings.max_objects_per_frame,
            iou_threshold=runtime_settings.task1_iou_threshold,
        )
        latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
        latencies_ms.append(latency_ms)

        histogram = Counter(int(item.class_id) for item in detections)
        aggregate_histogram.update(histogram)
        metadata = detections[0].metadata if detections else {}
        frame_results.append(
            {
                "frame_url": sample.frame.frame_url,
                "frame_index": sample.decoded_frame.frame_index,
                "bbox_count": len(detections),
                "class_histogram": {str(key): int(value) for key, value in sorted(histogram.items())},
                "latency_ms": latency_ms,
                "runtime_identifier": metadata.get("runtime_device") or metadata.get("active_backend_name") or metadata.get("backend_name") or "unknown",
                "provider": metadata.get("trt_provider") or metadata.get("onnx_provider"),
            }
        )
        current_snapshot = query_vram(runtime_settings.profiling_gpu_query_cmd)
        if current_snapshot.used_mb is not None:
            peak_vram_mb = max(peak_vram_mb or current_snapshot.used_mb, current_snapshot.used_mb)

    before_unload = query_vram(runtime_settings.profiling_gpu_query_cmd)
    detector.unload()
    after_unload = query_vram(runtime_settings.profiling_gpu_query_cmd)

    return {
        "frame_results": frame_results,
        "aggregate_class_histogram": {str(key): int(value) for key, value in sorted(aggregate_histogram.items())},
        "latency_p50_ms": round(percentile(latencies_ms, 50.0), 6) if latencies_ms else 0.0,
        "latency_p95_ms": round(percentile(latencies_ms, 95.0), 6) if latencies_ms else 0.0,
        "runtime_identifier": frame_results[0]["runtime_identifier"] if frame_results else "unknown",
        "provider": frame_results[0].get("provider") if frame_results else None,
        "all_zero_detections": all(item["bbox_count"] == 0 for item in frame_results),
        "peak_vram_mb": peak_vram_mb,
        "unload_recovery_mb": recovery_mb(before_unload, after_unload),
    }


def compare_detection_runs(
    native_run: dict[str, Any],
    onnx_run: dict[str, Any],
    *,
    latency_factor_threshold: float = 1.5,
    bbox_count_ratio: float = 0.20,
    class_delta_limit: int = 1,
) -> dict[str, Any]:
    frame_comparisons: list[dict[str, Any]] = []
    count_mismatches = 0
    class_mismatches = 0
    native_frames = native_run.get("frame_results", [])
    onnx_frames = onnx_run.get("frame_results", [])
    for native_frame, onnx_frame in zip(native_frames, onnx_frames):
        native_count = int(native_frame.get("bbox_count", 0))
        onnx_count = int(onnx_frame.get("bbox_count", 0))
        allowed_delta = 0 if native_count == 0 else max(1, ceil(native_count * bbox_count_ratio))
        count_ok = abs(native_count - onnx_count) <= allowed_delta
        if native_count == 0 and onnx_count != 0:
            count_ok = False
        native_hist = Counter({int(key): int(value) for key, value in native_frame.get("class_histogram", {}).items()})
        onnx_hist = Counter({int(key): int(value) for key, value in onnx_frame.get("class_histogram", {}).items()})
        histogram_ok = True
        for class_id in sorted(set(native_hist) | set(onnx_hist)):
            if abs(native_hist.get(class_id, 0) - onnx_hist.get(class_id, 0)) > class_delta_limit:
                histogram_ok = False
                break
        if not count_ok:
            count_mismatches += 1
        if not histogram_ok:
            class_mismatches += 1
        frame_comparisons.append(
            {
                "frame_index": native_frame.get("frame_index"),
                "native_bbox_count": native_count,
                "onnx_bbox_count": onnx_count,
                "allowed_bbox_delta": allowed_delta,
                "bbox_count_ok": count_ok,
                "class_histogram_ok": histogram_ok,
                "native_latency_ms": native_frame.get("latency_ms"),
                "onnx_latency_ms": onnx_frame.get("latency_ms"),
            }
        )

    native_histogram = Counter({int(key): int(value) for key, value in native_run.get("aggregate_class_histogram", {}).items()})
    onnx_histogram = Counter({int(key): int(value) for key, value in onnx_run.get("aggregate_class_histogram", {}).items()})
    aggregate_class_delta_ok = True
    aggregate_class_deltas: dict[str, int] = {}
    for class_id in sorted(set(native_histogram) | set(onnx_histogram)):
        delta = abs(native_histogram.get(class_id, 0) - onnx_histogram.get(class_id, 0))
        aggregate_class_deltas[str(class_id)] = int(delta)
        if delta > class_delta_limit:
            aggregate_class_delta_ok = False

    native_p50 = float(native_run.get("latency_p50_ms", 0.0))
    onnx_p50 = float(onnx_run.get("latency_p50_ms", 0.0))
    latency_ok = True if native_p50 <= 0 else onnx_p50 <= native_p50 * latency_factor_threshold
    onnx_smoke_passed = bool(onnx_frames)
    accepted = (
        onnx_smoke_passed
        and count_mismatches == 0
        and class_mismatches == 0
        and aggregate_class_delta_ok
        and latency_ok
    )
    return {
        "accepted": accepted,
        "onnx_smoke_passed": onnx_smoke_passed,
        "latency_ok": latency_ok,
        "native_latency_p50_ms": native_p50,
        "onnx_latency_p50_ms": onnx_p50,
        "bbox_count_mismatch_frames": count_mismatches,
        "class_distribution_mismatch_frames": class_mismatches,
        "aggregate_class_delta_ok": aggregate_class_delta_ok,
        "aggregate_class_deltas": aggregate_class_deltas,
        "frame_comparisons": frame_comparisons,
    }


def validate_task1_onnx_export(
    runtime_settings: MvpRuntimeSettings,
    *,
    candidate_name: str = "yolo26n",
    output_dir: str | Path = EXPORT_REPORTS_DIR,
    logger: StructuredLogger | None = None,
) -> dict[str, Any]:
    logger = logger or StructuredLogger()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    export_payload = export_task1_candidate_to_onnx(
        runtime_settings,
        candidate_name=candidate_name,
        logger=logger,
    )
    summary_name = f"task1_{candidate_name}_onnx_export_summary.json"
    report_json_name = f"task1_{candidate_name}_native_vs_onnx.json"
    report_md_name = f"task1_{candidate_name}_native_vs_onnx.md"
    export_summary_path = output_path / summary_name
    if not export_payload.get("success"):
        export_summary_path.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
        if candidate_name == "yolo26n":
            (output_path / "task1_onnx_export_summary.json").write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
        return {
            "export_summary": export_payload,
            "comparison": {"accepted": False, "onnx_smoke_passed": False},
        }

    validation_video = discover_task1_validation_video()
    samples = sample_task1_validation_frames(validation_video, sample_count=10)

    native_settings = replace(
        runtime_settings,
        task1_detector_backend=candidate_name,
        task1_model_runtime="ultralytics",
    )
    onnx_candidate_paths = dict(runtime_settings.task1_onnx_candidate_paths)
    onnx_candidate_paths[candidate_name] = str(export_payload["exported_path"])
    onnx_settings = replace(
        runtime_settings,
        task1_detector_backend=candidate_name,
        task1_model_runtime="onnxruntime",
        task1_onnx_candidate_paths=onnx_candidate_paths,
    )

    native_run = run_task1_pipeline_on_samples(native_settings, samples)
    if native_run.get("all_zero_detections"):
        samples = sample_task1_validation_frames(validation_video, sample_count=20)
        native_run = run_task1_pipeline_on_samples(native_settings, samples)
    onnx_run = run_task1_pipeline_on_samples(onnx_settings, samples)
    comparison = compare_detection_runs(native_run, onnx_run)

    export_summary = {
        **export_payload,
        "validation_video": str(validation_video),
        "sample_count": len(samples),
        "onnx_provider": onnx_run.get("provider"),
        "onnx_runtime_identifier": onnx_run.get("runtime_identifier"),
        "onnx_smoke_passed": comparison["onnx_smoke_passed"],
        "validation_passed": comparison["accepted"],
    }
    export_summary_path.write_text(json.dumps(export_summary, indent=2), encoding="utf-8")
    if candidate_name == "yolo26n":
        (output_path / "task1_onnx_export_summary.json").write_text(json.dumps(export_summary, indent=2), encoding="utf-8")

    native_vs_onnx_payload = {
        "candidate_name": candidate_name,
        "validation_video": str(validation_video),
        "sample_count": len(samples),
        "native": native_run,
        "onnx": onnx_run,
        "comparison": comparison,
    }
    (output_path / report_json_name).write_text(
        json.dumps(native_vs_onnx_payload, indent=2),
        encoding="utf-8",
    )
    (output_path / report_md_name).write_text(
        render_task1_native_vs_onnx_report(native_vs_onnx_payload),
        encoding="utf-8",
    )
    if candidate_name == "yolo26n":
        (output_path / "task1_native_vs_onnx.json").write_text(
            json.dumps(native_vs_onnx_payload, indent=2),
            encoding="utf-8",
        )
        (output_path / "task1_native_vs_onnx.md").write_text(
            render_task1_native_vs_onnx_report(native_vs_onnx_payload),
            encoding="utf-8",
        )
    logger.log_runtime(
        event="task1_onnx_validation_completed",
        adapter="Task1OnnxValidation",
        diagnostics={
            "candidate_name": candidate_name,
            "onnx_provider": onnx_run.get("provider"),
            "sample_count": len(samples),
            "validation_passed": comparison["accepted"],
            "onnx_smoke_passed": comparison["onnx_smoke_passed"],
        },
    )
    return {
        "export_summary": export_summary,
        "native": native_run,
        "onnx": onnx_run,
        "comparison": comparison,
    }


def render_task1_native_vs_onnx_report(payload: dict[str, Any]) -> str:
    native = payload.get("native", {})
    onnx = payload.get("onnx", {})
    comparison = payload.get("comparison", {})
    lines = [
        f"Candidate: {payload.get('candidate_name')}",
        f"Validation Video: {payload.get('validation_video')}",
        f"Sample Count: {payload.get('sample_count')}",
        "",
        "| Metric | Native | ONNX |",
        "| --- | --- | --- |",
        f"| Runtime | {native.get('runtime_identifier')} | {onnx.get('runtime_identifier')} |",
        f"| Provider | - | {onnx.get('provider')} |",
        f"| P50 Latency ms | {native.get('latency_p50_ms')} | {onnx.get('latency_p50_ms')} |",
        f"| P95 Latency ms | {native.get('latency_p95_ms')} | {onnx.get('latency_p95_ms')} |",
        f"| Aggregate Classes | {json.dumps(native.get('aggregate_class_histogram', {}), ensure_ascii=False)} | {json.dumps(onnx.get('aggregate_class_histogram', {}), ensure_ascii=False)} |",
        "",
        f"Accepted: {comparison.get('accepted')}",
        f"ONNX Smoke Passed: {comparison.get('onnx_smoke_passed')}",
        "",
        "| Frame | Native Count | ONNX Count | Allowed Delta | Count OK | Class OK |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in comparison.get("frame_comparisons", []):
        lines.append(
            "| {frame} | {native_count} | {onnx_count} | {allowed} | {count_ok} | {class_ok} |".format(
                frame=item.get("frame_index"),
                native_count=item.get("native_bbox_count"),
                onnx_count=item.get("onnx_bbox_count"),
                allowed=item.get("allowed_bbox_delta"),
                count_ok=item.get("bbox_count_ok"),
                class_ok=item.get("class_histogram_ok"),
            )
        )
    return "\n".join(lines) + "\n"
