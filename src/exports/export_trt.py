from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.logger import StructuredLogger


@dataclass(slots=True)
class TrtExportSpec:
    candidate_name: str
    source_onnx_path: Path
    output_engine_path: Path
    metadata_path: Path
    input_shape: tuple[int, int, int, int]
    precision: str
    fallback_precision: str
    workspace_mb: int


def is_tensorrt_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


def build_task1_trt_export_spec(
    runtime_settings: MvpRuntimeSettings,
    *,
    candidate_name: str = "yolo26n",
) -> TrtExportSpec:
    source_onnx_path = runtime_settings.resolve_task1_onnx_path(candidate_name)
    if source_onnx_path is None:
        raise FileNotFoundError(f"task1_source_onnx_missing:{candidate_name}")
    return TrtExportSpec(
        candidate_name=candidate_name,
        source_onnx_path=Path(source_onnx_path),
        output_engine_path=runtime_settings.resolve_task1_trt_path(candidate_name)
        or runtime_settings.resolve_task1_export_engine_path(candidate_name),
        metadata_path=runtime_settings.resolve_task1_trt_metadata_path(candidate_name),
        input_shape=tuple(int(item) for item in runtime_settings.task1_trt_input_shape),
        precision=str(runtime_settings.task1_trt_precision).lower(),
        fallback_precision=str(runtime_settings.task1_trt_fallback_precision).lower(),
        workspace_mb=int(runtime_settings.task1_trt_workspace_mb),
    )


def export_task1_candidate_to_trt(
    runtime_settings: MvpRuntimeSettings,
    *,
    candidate_name: str = "yolo26n",
    logger: StructuredLogger | None = None,
) -> dict[str, Any]:
    logger = logger or StructuredLogger()
    spec = build_task1_trt_export_spec(runtime_settings, candidate_name=candidate_name)
    payload: dict[str, Any] = {
        "success": False,
        "candidate_name": candidate_name,
        "source_onnx_path": str(spec.source_onnx_path),
        "output_engine_path": str(spec.output_engine_path),
        "metadata_path": str(spec.metadata_path),
        "input_shape": list(spec.input_shape),
        "precision": spec.precision,
        "fallback_precision": spec.fallback_precision,
        "workspace_mb": spec.workspace_mb,
    }
    logger.log_runtime(
        event="task1_trt_export_started",
        adapter="Task1TrtExport",
        diagnostics=payload,
    )

    if not spec.source_onnx_path.exists():
        payload["error"] = f"missing_onnx_file:{spec.source_onnx_path}"
        logger.log_error(
            event="task1_trt_export_failed",
            adapter="Task1TrtExport",
            diagnostics=payload,
        )
        return payload
    if not is_tensorrt_available():
        payload["error"] = "missing_dependency:tensorrt"
        logger.log_error(
            event="task1_trt_export_failed",
            adapter="Task1TrtExport",
            diagnostics=payload,
        )
        return payload

    spec.output_engine_path.parent.mkdir(parents=True, exist_ok=True)
    spec.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tensorrt as trt  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - opsiyonel bagimlilik
        payload["error"] = f"missing_dependency:tensorrt:{exc}"
        logger.log_error(
            event="task1_trt_export_failed",
            adapter="Task1TrtExport",
            diagnostics=payload,
        )
        return payload

    build_error: str | None = None
    built_precision: str | None = None
    engine_bytes: bytes | None = None
    version_info = {
        "tensorrt_version": getattr(trt, "__version__", "unknown"),
    }
    for precision in _unique_precisions(spec.precision, spec.fallback_precision):
        try:
            engine_bytes = _build_engine_bytes(
                trt=trt,
                onnx_path=spec.source_onnx_path,
                input_shape=spec.input_shape,
                precision=precision,
                workspace_mb=spec.workspace_mb,
            )
            if engine_bytes:
                built_precision = precision
                break
        except Exception as exc:  # pragma: no cover - TRT ortam bagimli
            build_error = str(exc)

    if not engine_bytes or built_precision is None:
        payload["error"] = build_error or "trt_build_failed"
        logger.log_error(
            event="task1_trt_export_failed",
            adapter="Task1TrtExport",
            diagnostics=payload,
        )
        return payload

    spec.output_engine_path.write_bytes(engine_bytes)
    onnx_metadata_path = runtime_settings.resolve_task1_export_metadata_path(candidate_name)
    class_names: list[str] = []
    if onnx_metadata_path.exists():
        try:
            class_names = list(json.loads(onnx_metadata_path.read_text(encoding="utf-8")).get("class_names", []))
        except Exception:
            class_names = []
    metadata_payload = {
        "candidate_name": candidate_name,
        "source_onnx_path": str(spec.source_onnx_path),
        "output_engine_path": str(spec.output_engine_path),
        "input_shape": list(spec.input_shape),
        "precision": built_precision,
        "workspace_mb": spec.workspace_mb,
        "builder_version": version_info["tensorrt_version"],
        "runtime": "tensorrt",
        "class_names": class_names,
    }
    spec.metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    payload.update(
        {
            "success": True,
            "engine_path": str(spec.output_engine_path),
            "metadata_path": str(spec.metadata_path),
            "precision": built_precision,
            "engine_size_bytes": spec.output_engine_path.stat().st_size,
            **version_info,
        }
    )
    logger.log_runtime(
        event="task1_trt_export_finished",
        adapter="Task1TrtExport",
        diagnostics=payload,
    )
    return payload


def export_trt_spec_to_dict(spec: TrtExportSpec) -> dict[str, Any]:
    data = asdict(spec)
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def _build_engine_bytes(
    *,
    trt: Any,
    onnx_path: Path,
    input_shape: tuple[int, int, int, int],
    precision: str,
    workspace_mb: int,
) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        errors = [parser.get_error(index).desc() for index in range(parser.num_errors)]
        raise RuntimeError("trt_onnx_parse_failed:" + " | ".join(errors))

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * 1024 * 1024)
    if precision == "fp16":
        if not bool(getattr(builder, "platform_has_fast_fp16", False)):
            raise RuntimeError("trt_fp16_not_supported")
        config.set_flag(trt.BuilderFlag.FP16)

    profile = _maybe_create_optimization_profile(trt=trt, builder=builder, network=network, input_shape=input_shape)
    if profile is not None:
        config.add_optimization_profile(profile)

    engine_blob = builder.build_serialized_network(network, config)
    if engine_blob is None:
        raise RuntimeError("trt_serialized_engine_empty")
    return bytes(engine_blob)


def _maybe_create_optimization_profile(
    *,
    trt: Any,
    builder: Any,
    network: Any,
    input_shape: tuple[int, int, int, int],
) -> Any | None:
    input_tensor = network.get_input(0) if network.num_inputs else None
    if input_tensor is None:
        raise RuntimeError("trt_network_has_no_input")
    input_dims = tuple(int(dim) for dim in input_tensor.shape)
    if all(dim > 0 for dim in input_dims):
        return None
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
    return profile


def _unique_precisions(primary: str, fallback: str) -> list[str]:
    ordered: list[str] = []
    for precision in (primary, fallback):
        normalized = str(precision).lower()
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered
