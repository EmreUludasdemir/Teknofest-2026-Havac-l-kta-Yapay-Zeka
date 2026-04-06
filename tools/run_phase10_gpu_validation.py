from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.core.logger import StructuredLogger
from src.data.validators import SchemaValidator
from src.exports.export_trt import export_task1_candidate_to_trt, resolve_gpu_name
from src.pipeline.mvp_processor import MvpFrameProcessor
from src.server.final_sequential_adapter import FinalSequentialAdapter
from src.tools.mock_server import OfficialRepoMockServer
from src.tools.runtime_bootstrap import execute_runtime_warmup
from src.tools.runtime_package import load_runtime_bootstrap
from src.tools.vram_monitor import query_vram


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 10 GPU production validation")
    parser.add_argument("--config", default="final_runtime/config/runtime.toml")
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--output-json", default="reports/phase10_gpu_validation.json")
    parser.add_argument("--output-md", default="reports/phase10_gpu_validation.md")
    parser.add_argument("--skip-fallback-probes", action="store_true")
    args = parser.parse_args(argv)

    bootstrap = load_runtime_bootstrap(
        args.config,
        production=True,
        base_url_override=args.base_url,
        username_override=args.username,
        password_override=args.password,
    )
    logger = StructuredLogger(log_dir=Path(bootstrap.runtime_config.get("paths", {}).get("log_dir", "final_runtime/logs")))
    preflight = collect_gpu_preflight(bootstrap.runtime_settings)
    rebuild_info = ensure_trt_engine_ready(bootstrap.runtime_settings, logger=logger)

    primary_run = run_validation_with_optional_mock(
        runtime_settings=bootstrap.runtime_settings,
        sequential_settings=bootstrap.sequential_settings,
        logger=logger,
        max_frames=args.max_frames,
        session_name="phase10_gpu_validation",
        base_url=args.base_url,
    )

    fallback_probes: list[dict[str, Any]] = []
    if not args.skip_fallback_probes:
        fallback_probes.append(
            run_fallback_probe(
                runtime_settings=bootstrap.runtime_settings,
                sequential_settings=bootstrap.sequential_settings,
                logger=logger,
                disabled_stages={"tensorrt:yolo26n"},
                probe_name="trt_unavailable_probe",
                base_url=args.base_url,
            )
        )
        fallback_probes.append(
            run_fallback_probe(
                runtime_settings=bootstrap.runtime_settings,
                sequential_settings=bootstrap.sequential_settings,
                logger=logger,
                disabled_stages={"tensorrt:yolo26n", "onnxruntime:yolo26n"},
                probe_name="trt_onnx_unavailable_probe",
                base_url=args.base_url,
            )
        )

    selected_backend = str(primary_run.get("active_task1_stage"))
    selected_runtime = selected_backend.split(":", maxsplit=1)[0] if ":" in selected_backend else selected_backend
    selected_provider = str(primary_run.get("active_provider") or "")
    gpu_path_exercised = (
        selected_backend in {"tensorrt:yolo26n", "onnxruntime:yolo26n", "ultralytics:yolo26n"}
        and (
            selected_backend.startswith("tensorrt:")
            or "cuda" in selected_provider.lower()
            or "executionprovider" in selected_provider.lower()
        )
    )
    first_frame_latency = float(primary_run.get("first_frame_total_latency_ms") or 0.0)
    warm_p50 = float(primary_run.get("warm_total_latency_p50_ms") or 0.0)
    first_frame_regression = bool(warm_p50 > 0.0 and first_frame_latency > (2.0 * warm_p50))
    no_go_reason = None
    if selected_backend in {"ultralytics:yolo11n", "synthetic"}:
        no_go_reason = f"selected_backend={selected_backend}"
    elif not gpu_path_exercised:
        no_go_reason = f"gpu_path_not_exercised:{selected_backend}:{selected_provider or 'unknown'}"

    payload = {
        "machine": {
            "platform": platform.platform(),
            "python_executable": sys.executable,
        },
        "preflight": preflight,
        "trt_rebuild": rebuild_info,
        "primary_run": primary_run,
        "fallback_probes": fallback_probes,
        "decision": {
            "gpu_path_exercised": gpu_path_exercised,
            "selected_backend": selected_backend,
            "selected_provider": selected_provider,
            "first_frame_regression": first_frame_regression,
            "no_go_reason": no_go_reason,
            "accepted": gpu_path_exercised and no_go_reason is None,
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_phase10_gpu_validation_report(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2))
    return 0 if bool(payload["decision"]["accepted"]) else 1


def collect_gpu_preflight(runtime_settings: MvpRuntimeSettings) -> dict[str, Any]:
    torch_payload = {"available": False, "cuda_available": False, "version": None}
    try:
        import torch  # type: ignore[import-not-found]

        torch_payload = {
            "available": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "version": getattr(torch, "__version__", None),
            "device_count": int(torch.cuda.device_count()) if bool(torch.cuda.is_available()) else 0,
            "gpu_name": resolve_gpu_name(),
        }
    except Exception as exc:
        torch_payload["error"] = str(exc)

    onnx_payload = {"available": False, "providers": []}
    try:
        import onnxruntime  # type: ignore[import-not-found]

        onnx_payload = {
            "available": True,
            "version": getattr(onnxruntime, "__version__", None),
            "providers": list(onnxruntime.get_available_providers()),
        }
    except Exception as exc:
        onnx_payload["error"] = str(exc)

    trt_payload = {"available": False, "version": None}
    try:
        import tensorrt as trt  # type: ignore[import-not-found]

        trt_payload = {
            "available": True,
            "version": getattr(trt, "__version__", None),
        }
    except Exception as exc:
        trt_payload["error"] = str(exc)

    gpu_query = query_gpu_name_and_total()
    manifest_paths = {
        "yolo26n_pt": str(runtime_settings.resolve_task1_model_path("yolo26n") or ""),
        "yolo26n_onnx": str(runtime_settings.resolve_task1_onnx_path("yolo26n") or ""),
        "yolo26n_trt": str(runtime_settings.resolve_task1_trt_path("yolo26n") or ""),
    }
    return {
        "torch": torch_payload,
        "onnxruntime": onnx_payload,
        "tensorrt": trt_payload,
        "gpu_query": gpu_query,
        "vram_snapshot": asdict(query_vram(runtime_settings.profiling_gpu_query_cmd)),
        "required_paths": {
            key: {
                "path": value,
                "exists": bool(value) and Path(value).exists(),
            }
            for key, value in manifest_paths.items()
        },
    }


def ensure_trt_engine_ready(
    runtime_settings: MvpRuntimeSettings,
    *,
    logger: StructuredLogger,
) -> dict[str, Any]:
    engine_path = runtime_settings.resolve_task1_trt_path("yolo26n")
    metadata_path = runtime_settings.resolve_task1_trt_metadata_path("yolo26n")
    current_gpu_name = resolve_gpu_name()
    current_trt_version = None
    try:
        import tensorrt as trt  # type: ignore[import-not-found]

        current_trt_version = getattr(trt, "__version__", None)
    except Exception:
        current_trt_version = None

    mismatch_reasons: list[str] = []
    metadata_payload: dict[str, Any] = {}
    if engine_path is None or not Path(engine_path).exists():
        mismatch_reasons.append("missing_engine")
    if not metadata_path.exists():
        mismatch_reasons.append("missing_engine_metadata")
    else:
        try:
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            mismatch_reasons.append(f"metadata_read_failed:{exc}")
        else:
            expected_shape = list(runtime_settings.task1_trt_input_shape)
            metadata_shape = list(metadata_payload.get("input_shape", []))
            if metadata_shape != expected_shape:
                mismatch_reasons.append("input_shape_mismatch")
            metadata_precision = str(metadata_payload.get("precision", "")).lower()
            allowed_precisions = {
                str(runtime_settings.task1_trt_precision).lower(),
                str(runtime_settings.task1_trt_fallback_precision).lower(),
            }
            if metadata_precision not in allowed_precisions:
                mismatch_reasons.append("precision_mismatch")
            metadata_trt_version = metadata_payload.get("builder_version")
            if current_trt_version and metadata_trt_version and str(metadata_trt_version) != str(current_trt_version):
                mismatch_reasons.append("tensorrt_version_mismatch")
            metadata_gpu_name = metadata_payload.get("gpu_name")
            if current_gpu_name and metadata_gpu_name and str(metadata_gpu_name) != str(current_gpu_name):
                mismatch_reasons.append("gpu_name_mismatch")
            if current_gpu_name and not metadata_gpu_name:
                mismatch_reasons.append("missing_gpu_name")

    rebuild_payload: dict[str, Any] | None = None
    if mismatch_reasons:
        rebuild_payload = export_task1_candidate_to_trt(
            runtime_settings,
            candidate_name="yolo26n",
            logger=logger,
        )
        if rebuild_payload.get("success"):
            mismatch_reasons = []
            try:
                metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata_payload = {}
    return {
        "engine_path": str(engine_path) if engine_path is not None else None,
        "metadata_path": str(metadata_path),
        "mismatch_reasons": mismatch_reasons,
        "metadata": metadata_payload,
        "rebuild_attempted": rebuild_payload is not None,
        "rebuild_result": rebuild_payload,
    }


def run_validation_sequence(
    runtime_settings: MvpRuntimeSettings,
    sequential_settings,
    *,
    logger: StructuredLogger,
    max_frames: int,
    session_name: str,
) -> dict[str, Any]:
    validator = SchemaValidator()
    adapter = FinalSequentialAdapter(sequential_settings, logger=logger)
    processor = MvpFrameProcessor(runtime_settings=runtime_settings)
    processed_frames: list[dict[str, Any]] = []
    session_payload: dict[str, Any] = {}
    warmup_payload: dict[str, Any] = {}
    initial_vram = query_vram(runtime_settings.profiling_gpu_query_cmd)
    peak_vram = initial_vram.used_mb or 0.0

    try:
        adapter.login()
        session_payload = adapter.open_session()
        warmup_payload = execute_runtime_warmup(
            processor=processor,
            logger=logger,
            session_payload=session_payload,
            session_name=str(session_payload.get("session_name", session_name)),
            adapter=adapter,
        )

        for _ in range(max_frames):
            fetch_started = perf_counter()
            frame = adapter.fetch_next_frame()
            fetch_latency_ms = round((perf_counter() - fetch_started) * 1000.0, 3)
            if frame is None:
                break
            image_started = perf_counter()
            image_bytes = adapter.download_image(frame)
            process_started = perf_counter()
            result = processor(frame, image_bytes)
            validator.validate_canonical_result(result.to_canonical_dict())
            payload = adapter.build_wire_prediction(result)
            validator.validate_sequential_prediction(payload, profile=adapter.settings.wire_profile)
            response = adapter.send_wire_prediction(payload)
            total_latency_ms = round((perf_counter() - image_started) * 1000.0, 3)
            process_latency_ms = round((perf_counter() - process_started) * 1000.0, 3)
            active_stage = processor.task1_detector.active_stage_id
            active_provider = resolve_active_provider(processor, active_stage)
            processed_frames.append(
                {
                    "frame_url": frame.frame_url,
                    "frame_id": frame.metadata.get("frame_id"),
                    "fetch_latency_ms": fetch_latency_ms,
                    "process_latency_ms": process_latency_ms,
                    "total_latency_ms": total_latency_ms,
                    "status_code": response.status_code,
                    "active_task1_stage": active_stage,
                    "active_provider": active_provider,
                    "task2_diagnostics": result.diagnostics.get("task2_info", {}),
                }
            )
            current_vram = query_vram(runtime_settings.profiling_gpu_query_cmd)
            if current_vram.used_mb is not None:
                peak_vram = max(float(peak_vram), float(current_vram.used_mb))
    finally:
        processor.task1_detector.unload()
        try:
            adapter.close_session()
        except Exception:
            pass

    total_latencies = [float(item["total_latency_ms"]) for item in processed_frames]
    warm_latencies = total_latencies[1:] if len(total_latencies) > 1 else total_latencies
    active_stage = str(warmup_payload.get("active_task1_stage", "unknown"))
    active_provider = (
        processed_frames[0]["active_provider"]
        if processed_frames
        else resolve_active_provider(processor, active_stage)
    )
    return {
        "session_id": session_payload.get("session_id"),
        "session_name": session_payload.get("session_name"),
        "warmup": warmup_payload,
        "processed_frames": len(processed_frames),
        "frame_results": processed_frames,
        "active_task1_stage": active_stage,
        "active_provider": active_provider,
        "first_frame_total_latency_ms": total_latencies[0] if total_latencies else None,
        "warm_total_latency_p50_ms": round(percentile50(warm_latencies), 6) if warm_latencies else None,
        "warm_total_latency_p95_ms": round(percentile95(warm_latencies), 6) if warm_latencies else None,
        "peak_vram_mb": round(float(peak_vram), 3) if peak_vram else 0.0,
    }


def run_fallback_probe(
    *,
    runtime_settings: MvpRuntimeSettings,
    sequential_settings,
    logger: StructuredLogger,
    disabled_stages: set[str],
    probe_name: str,
    base_url: str | None,
) -> dict[str, Any]:
    runtime_order = [stage for stage in runtime_settings.task1_runtime_order if stage not in disabled_stages]
    probe_settings = replace(
        runtime_settings,
        task1_runtime_order=runtime_order,
        task1_enabled_stage_ids=[stage for stage in runtime_settings.task1_enabled_stage_ids if stage not in disabled_stages],
    )
    result = run_validation_with_optional_mock(
        runtime_settings=probe_settings,
        sequential_settings=sequential_settings,
        logger=logger,
        max_frames=1,
        session_name=probe_name,
        base_url=base_url,
    )
    return {
        "probe_name": probe_name,
        "disabled_stages": sorted(disabled_stages),
        "selected_stage": result.get("active_task1_stage"),
        "selected_provider": result.get("active_provider"),
        "warmup": result.get("warmup"),
    }


def run_validation_with_optional_mock(
    *,
    runtime_settings: MvpRuntimeSettings,
    sequential_settings,
    logger: StructuredLogger,
    max_frames: int,
    session_name: str,
    base_url: str | None,
) -> dict[str, Any]:
    with maybe_mock_server(base_url) as server_base_url:
        effective_base_url = server_base_url or sequential_settings.base_url
        run_settings = replace(sequential_settings, base_url=effective_base_url)
        return run_validation_sequence(
            runtime_settings,
            run_settings,
            logger=logger,
            max_frames=max_frames,
            session_name=session_name,
        )


def resolve_active_provider(processor: MvpFrameProcessor, active_stage: str) -> str:
    for spec, backend in processor.task1_detector.backend_chain:
        if spec.stage_id != active_stage:
            continue
        provider = getattr(backend, "runtime_provider", None)
        if provider:
            return str(provider)
        runtime_device = getattr(backend, "runtime_device", None)
        if runtime_device is not None:
            return str(runtime_device)
        return spec.runtime_name
    return "unknown"


def percentile50(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(item) for item in values)
    index = max(int(round((len(ordered) - 1) * 0.95)), 0)
    return float(ordered[index])


def query_gpu_name_and_total() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"available": False, "error": completed.stderr.strip() or f"returncode={completed.returncode}"}
    line = completed.stdout.strip().splitlines()[0]
    name_text, total_text = [part.strip() for part in line.split(",", maxsplit=1)]
    return {"available": True, "name": name_text, "memory_total_mb": float(total_text)}


@contextmanager
def maybe_mock_server(base_url: str | None):
    if base_url:
        yield None
        return
    server = OfficialRepoMockServer(mode="sequential", sequential_warmup_delay_s=0.05)
    server.start()
    try:
        yield server.base_url
    finally:
        server.stop()


def render_phase10_gpu_validation_report(payload: dict[str, Any]) -> str:
    primary = payload.get("primary_run", {})
    decision = payload.get("decision", {})
    preflight = payload.get("preflight", {})
    rebuild = payload.get("trt_rebuild", {})
    lines = [
        "# Phase 10 GPU Validation",
        "",
        f"- Python: `{payload.get('machine', {}).get('python_executable')}`",
        f"- Platform: `{payload.get('machine', {}).get('platform')}`",
        f"- GPU: `{preflight.get('gpu_query', {}).get('name')}`",
        f"- CUDA Visible: `{preflight.get('torch', {}).get('cuda_available')}`",
        f"- ONNX Providers: `{preflight.get('onnxruntime', {}).get('providers')}`",
        f"- TensorRT Version: `{preflight.get('tensorrt', {}).get('version')}`",
        "",
        "## Production Run",
        f"- Active backend: `{primary.get('active_task1_stage')}`",
        f"- Active provider: `{primary.get('active_provider')}`",
        f"- Warm-up attempted stages: `{primary.get('warmup', {}).get('attempted_stages')}`",
        f"- Warm-up failed stages: `{primary.get('warmup', {}).get('failed_stages')}`",
        f"- First frame latency ms: `{primary.get('first_frame_total_latency_ms')}`",
        f"- Warm P50 ms: `{primary.get('warm_total_latency_p50_ms')}`",
        f"- Warm P95 ms: `{primary.get('warm_total_latency_p95_ms')}`",
        f"- Peak VRAM MB: `{primary.get('peak_vram_mb')}`",
        "",
        "## TRT Compatibility",
        f"- Rebuild attempted: `{rebuild.get('rebuild_attempted')}`",
        f"- Mismatch reasons: `{rebuild.get('mismatch_reasons')}`",
        "",
        "## Fallback Probes",
    ]
    for probe in payload.get("fallback_probes", []):
        lines.append(
            "- `{name}` -> `{stage}` ({provider})".format(
                name=probe.get("probe_name"),
                stage=probe.get("selected_stage"),
                provider=probe.get("selected_provider"),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            f"- GPU path exercised: `{decision.get('gpu_path_exercised')}`",
            f"- First-frame regression: `{decision.get('first_frame_regression')}`",
            f"- Accepted: `{decision.get('accepted')}`",
            f"- No-go reason: `{decision.get('no_go_reason')}`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
