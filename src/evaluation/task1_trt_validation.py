from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.logger import StructuredLogger
from src.evaluation.task1_onnx_validation import (
    compare_detection_runs,
    discover_task1_validation_video,
    render_task1_native_vs_onnx_report,
    run_task1_pipeline_on_samples,
    sample_task1_validation_frames,
    validate_task1_onnx_export,
)
from src.exports.export_trt import export_task1_candidate_to_trt


def validate_task1_trt_export(
    runtime_settings: MvpRuntimeSettings,
    *,
    primary_candidate: str = "yolo26n",
    fallback_candidate: str = "yolo11n",
    output_dir: str | Path = "reports/export",
    logger: StructuredLogger | None = None,
) -> dict[str, Any]:
    logger = logger or StructuredLogger()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_export_settings = replace(runtime_settings, task1_model_runtime="ultralytics")
    primary_onnx_payload = validate_task1_onnx_export(
        onnx_export_settings,
        candidate_name=primary_candidate,
        output_dir=output_path,
        logger=logger,
    )
    fallback_onnx_payload = validate_task1_onnx_export(
        onnx_export_settings,
        candidate_name=fallback_candidate,
        output_dir=output_path,
        logger=logger,
    )

    trt_export_payload = export_task1_candidate_to_trt(
        runtime_settings,
        candidate_name=primary_candidate,
        logger=logger,
    )
    validation_video = discover_task1_validation_video()
    samples = sample_task1_validation_frames(validation_video, sample_count=10)

    native_run = primary_onnx_payload.get("native")
    onnx_run = primary_onnx_payload.get("onnx")
    trt_run: dict[str, Any] | None = None
    comparison = _build_failed_trt_comparison(
        native_run=native_run,
        onnx_run=onnx_run,
        onnx_accepted=bool(primary_onnx_payload.get("comparison", {}).get("accepted")),
    )

    if trt_export_payload.get("success"):
        trt_candidate_paths = dict(runtime_settings.task1_trt_candidate_paths)
        trt_candidate_paths[primary_candidate] = str(trt_export_payload["engine_path"])
        trt_settings = _runtime_settings_for_stage(
            runtime_settings,
            runtime_name="tensorrt",
            candidate_name=primary_candidate,
            trt_candidate_paths=trt_candidate_paths,
        )
        trt_run = run_task1_pipeline_on_samples(trt_settings, samples)
        comparison = compare_task1_native_onnx_trt_runs(
            native_run=native_run,
            onnx_run=onnx_run,
            trt_run=trt_run,
            runtime_settings=runtime_settings,
        )

    summary_payload = {
        **trt_export_payload,
        "primary_candidate": primary_candidate,
        "fallback_candidate": fallback_candidate,
        "validation_video": str(validation_video),
        "sample_count": len(samples),
        "trt_smoke_passed": comparison.get("trt_smoke_passed", False),
        "validation_passed": comparison.get("accepted", False),
        "fallback_onnx_validated": bool(
            fallback_onnx_payload.get("comparison", {}).get("accepted")
        ),
    }
    (output_path / "task1_trt_export_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    fallback_tree_payload = build_task1_fallback_tree(
        runtime_settings=runtime_settings,
        reports_dir=output_path,
    )
    triple_payload = {
        "primary_candidate": primary_candidate,
        "fallback_candidate": fallback_candidate,
        "validation_video": str(validation_video),
        "sample_count": len(samples),
        "native": native_run,
        "onnx": onnx_run,
        "trt": trt_run,
        "comparison": comparison,
        "fallback_tree": fallback_tree_payload,
        "fallback_onnx_summary": fallback_onnx_payload.get("export_summary"),
    }
    (output_path / "task1_native_vs_onnx_vs_trt.json").write_text(
        json.dumps(triple_payload, indent=2),
        encoding="utf-8",
    )
    (output_path / "task1_native_vs_onnx_vs_trt.md").write_text(
        render_task1_native_vs_onnx_vs_trt_report(triple_payload),
        encoding="utf-8",
    )
    (output_path / "task1_fallback_tree.json").write_text(
        json.dumps(fallback_tree_payload, indent=2),
        encoding="utf-8",
    )
    (output_path / "task1_fallback_tree.md").write_text(
        render_task1_fallback_tree_report(fallback_tree_payload),
        encoding="utf-8",
    )

    logger.log_runtime(
        event="task1_trt_validation_completed",
        adapter="Task1TrtValidation",
        diagnostics={
            "primary_candidate": primary_candidate,
            "fallback_candidate": fallback_candidate,
            "validation_passed": comparison.get("accepted", False),
            "trt_smoke_passed": comparison.get("trt_smoke_passed", False),
        },
    )
    return {
        "summary": summary_payload,
        "native": native_run,
        "onnx": onnx_run,
        "trt": trt_run,
        "comparison": comparison,
        "fallback_tree": fallback_tree_payload,
    }


def compare_task1_native_onnx_trt_runs(
    *,
    native_run: dict[str, Any],
    onnx_run: dict[str, Any],
    trt_run: dict[str, Any],
    runtime_settings: MvpRuntimeSettings,
) -> dict[str, Any]:
    trt_vs_native = compare_detection_runs(
        native_run,
        trt_run,
        latency_factor_threshold=999.0,
    )
    onnx_vs_native = compare_detection_runs(native_run, onnx_run)

    onnx_p50 = float(onnx_run.get("latency_p50_ms", 0.0))
    trt_p50 = float(trt_run.get("latency_p50_ms", 0.0))
    trt_latency_ok = True if onnx_p50 <= 0 else trt_p50 <= onnx_p50 * 1.10
    trt_peak_vram = float(trt_run.get("peak_vram_mb") or 0.0)
    trt_vram_ok = trt_peak_vram <= float(runtime_settings.task1_trt_max_vram_mb)
    accepted = (
        bool(onnx_vs_native.get("accepted"))
        and
        bool(trt_run.get("frame_results"))
        and trt_vs_native.get("bbox_count_mismatch_frames", 1) == 0
        and trt_vs_native.get("class_distribution_mismatch_frames", 1) == 0
        and bool(trt_vs_native.get("aggregate_class_delta_ok"))
        and trt_latency_ok
        and trt_vram_ok
    )
    return {
        "accepted": accepted,
        "trt_smoke_passed": bool(trt_run.get("frame_results")),
        "onnx_reference_accepted": onnx_vs_native.get("accepted"),
        "trt_latency_ok": trt_latency_ok,
        "trt_vram_ok": trt_vram_ok,
        "trt_peak_vram_mb": trt_peak_vram,
        "onnx_latency_p50_ms": onnx_p50,
        "trt_latency_p50_ms": trt_p50,
        "onnx_vs_native": onnx_vs_native,
        "trt_vs_native": trt_vs_native,
    }


def build_task1_fallback_tree(
    *,
    runtime_settings: MvpRuntimeSettings,
    reports_dir: str | Path = "reports/export",
) -> dict[str, Any]:
    base = Path(reports_dir)
    validation_map = {
        "onnxruntime:yolo26n": _load_validation_status(base / "task1_yolo26n_onnx_export_summary.json"),
        "onnxruntime:yolo11n": _load_validation_status(base / "task1_yolo11n_onnx_export_summary.json"),
        "tensorrt:yolo26n": _load_validation_status(base / "task1_trt_export_summary.json"),
    }

    stages: list[dict[str, Any]] = []
    for order_index, stage_name in enumerate(runtime_settings.task1_runtime_order, start=1):
        runtime_name, candidate_name = _parse_stage_name(stage_name)
        production_enabled = not (
            runtime_name == "onnxruntime"
            and candidate_name == "yolo11n"
        )
        stage_payload = {
            "order": order_index,
            "stage": stage_name,
            "runtime": runtime_name,
            "candidate": candidate_name,
            "validation_passed": validation_map.get(stage_name),
            "production_enabled": production_enabled,
            "required_artifact": _required_artifact_name(runtime_name, candidate_name),
        }
        stages.append(stage_payload)
    return {"stages": stages}


def render_task1_native_vs_onnx_vs_trt_report(payload: dict[str, Any]) -> str:
    native = payload.get("native") or {}
    onnx = payload.get("onnx") or {}
    trt_run = payload.get("trt") or {}
    comparison = payload.get("comparison") or {}
    trt_vs_native = comparison.get("trt_vs_native", {})
    lines = [
        f"Primary Candidate: {payload.get('primary_candidate')}",
        f"Fallback Candidate: {payload.get('fallback_candidate')}",
        f"Validation Video: {payload.get('validation_video')}",
        f"Sample Count: {payload.get('sample_count')}",
        "",
        "| Metric | Native | ONNX | TRT |",
        "| --- | --- | --- | --- |",
        f"| Runtime | {native.get('runtime_identifier')} | {onnx.get('runtime_identifier')} | {trt_run.get('runtime_identifier')} |",
        f"| Provider | - | {onnx.get('provider')} | {trt_run.get('provider')} |",
        f"| P50 Latency ms | {native.get('latency_p50_ms')} | {onnx.get('latency_p50_ms')} | {trt_run.get('latency_p50_ms')} |",
        f"| P95 Latency ms | {native.get('latency_p95_ms')} | {onnx.get('latency_p95_ms')} | {trt_run.get('latency_p95_ms')} |",
        f"| Peak VRAM MB | {native.get('peak_vram_mb')} | {onnx.get('peak_vram_mb')} | {trt_run.get('peak_vram_mb')} |",
        f"| Aggregate Classes | {json.dumps(native.get('aggregate_class_histogram', {}), ensure_ascii=False)} | {json.dumps(onnx.get('aggregate_class_histogram', {}), ensure_ascii=False)} | {json.dumps(trt_run.get('aggregate_class_histogram', {}), ensure_ascii=False)} |",
        "",
        f"Accepted: {comparison.get('accepted')}",
        f"TRT Smoke Passed: {comparison.get('trt_smoke_passed')}",
        f"TRT Latency OK: {comparison.get('trt_latency_ok')}",
        f"TRT VRAM OK: {comparison.get('trt_vram_ok')}",
        "",
        "| Frame | Native Count | TRT Count | Allowed Delta | Count OK | Class OK |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in trt_vs_native.get("frame_comparisons", []):
        lines.append(
            "| {frame} | {native_count} | {trt_count} | {allowed} | {count_ok} | {class_ok} |".format(
                frame=item.get("frame_index"),
                native_count=item.get("native_bbox_count"),
                trt_count=item.get("onnx_bbox_count"),
                allowed=item.get("allowed_bbox_delta"),
                count_ok=item.get("bbox_count_ok"),
                class_ok=item.get("class_histogram_ok"),
            )
        )
    return "\n".join(lines) + "\n"


def render_task1_fallback_tree_report(payload: dict[str, Any]) -> str:
    lines = [
        "| Order | Stage | Runtime | Candidate | Validation Passed | Production Enabled | Required Artifact |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for stage in payload.get("stages", []):
        lines.append(
            "| {order} | {stage} | {runtime} | {candidate} | {validation} | {enabled} | {artifact} |".format(
                order=stage.get("order"),
                stage=stage.get("stage"),
                runtime=stage.get("runtime"),
                candidate=stage.get("candidate"),
                validation=stage.get("validation_passed"),
                enabled=stage.get("production_enabled"),
                artifact=stage.get("required_artifact"),
            )
        )
    return "\n".join(lines) + "\n"


def _build_failed_trt_comparison(
    *,
    native_run: dict[str, Any] | None,
    onnx_run: dict[str, Any] | None,
    onnx_accepted: bool,
) -> dict[str, Any]:
    return {
        "accepted": False,
        "trt_smoke_passed": False,
        "onnx_reference_accepted": onnx_accepted,
        "trt_latency_ok": False,
        "trt_vram_ok": False,
        "trt_peak_vram_mb": None,
        "onnx_latency_p50_ms": float((onnx_run or {}).get("latency_p50_ms", 0.0)) if onnx_run else 0.0,
        "trt_latency_p50_ms": None,
        "onnx_vs_native": compare_detection_runs(native_run or {"frame_results": [], "aggregate_class_histogram": {}, "latency_p50_ms": 0.0}, onnx_run or {"frame_results": [], "aggregate_class_histogram": {}, "latency_p50_ms": 0.0}) if native_run and onnx_run else {},
        "trt_vs_native": {
            "bbox_count_mismatch_frames": 0,
            "class_distribution_mismatch_frames": 0,
            "aggregate_class_delta_ok": False,
            "aggregate_class_deltas": {},
            "frame_comparisons": [],
        },
    }


def _runtime_settings_for_stage(
    runtime_settings: MvpRuntimeSettings,
    *,
    runtime_name: str,
    candidate_name: str,
    trt_candidate_paths: dict[str, str] | None = None,
) -> MvpRuntimeSettings:
    return replace(
        runtime_settings,
        task1_detector_backend=candidate_name,
        task1_model_runtime=runtime_name,
        task1_trt_candidate_paths=trt_candidate_paths or dict(runtime_settings.task1_trt_candidate_paths),
    )


def _load_validation_status(path: Path) -> bool | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return bool(payload.get("validation_passed"))


def _parse_stage_name(stage_name: str) -> tuple[str, str | None]:
    if stage_name == "synthetic":
        return "synthetic", None
    runtime_name, candidate_name = stage_name.split(":", maxsplit=1)
    return runtime_name, candidate_name


def _required_artifact_name(runtime_name: str, candidate_name: str | None) -> str | None:
    if runtime_name == "synthetic" or candidate_name is None:
        return None
    if runtime_name == "tensorrt":
        return f"{candidate_name}.engine"
    if runtime_name == "onnxruntime":
        return f"{candidate_name}.onnx"
    return f"{candidate_name}.pt"
