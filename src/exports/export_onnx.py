from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.logger import StructuredLogger


@dataclass(slots=True)
class OnnxExportSpec:
    candidate_name: str
    source_model_path: Path
    output_model_path: Path
    metadata_path: Path
    imgsz: int
    opset: int
    dynamic: bool
    half: bool
    runtime_name: str
    device: str


def build_task1_export_spec(
    runtime_settings: MvpRuntimeSettings,
    *,
    candidate_name: str = "yolo26n",
) -> OnnxExportSpec:
    source_model_path = runtime_settings.resolve_task1_model_path(candidate_name)
    if source_model_path is None:
        raise FileNotFoundError(f"task1_source_model_missing:{candidate_name}")
    output_model_path = runtime_settings.resolve_task1_onnx_path(candidate_name) or runtime_settings.resolve_task1_export_model_path(candidate_name)
    metadata_path = runtime_settings.resolve_task1_export_metadata_path(candidate_name)
    return OnnxExportSpec(
        candidate_name=candidate_name,
        source_model_path=Path(source_model_path),
        output_model_path=Path(output_model_path),
        metadata_path=Path(metadata_path),
        imgsz=int(runtime_settings.task1_model_input_size),
        opset=int(runtime_settings.task1_export_opset),
        dynamic=bool(runtime_settings.task1_export_dynamic),
        half=bool(runtime_settings.task1_export_half),
        runtime_name=str(runtime_settings.task1_model_runtime),
        device=str(runtime_settings.task1_device or "cpu"),
    )


def export_task1_candidate_to_onnx(
    runtime_settings: MvpRuntimeSettings,
    *,
    candidate_name: str = "yolo26n",
    logger: StructuredLogger | None = None,
) -> dict[str, Any]:
    logger = logger or StructuredLogger()
    spec = build_task1_export_spec(runtime_settings, candidate_name=candidate_name)
    payload: dict[str, Any] = {
        "success": False,
        "candidate_name": candidate_name,
        "source_model_path": str(spec.source_model_path),
        "output_model_path": str(spec.output_model_path),
        "metadata_path": str(spec.metadata_path),
        "imgsz": spec.imgsz,
        "opset": spec.opset,
        "dynamic": spec.dynamic,
        "half": spec.half,
        "runtime_name": spec.runtime_name,
        "device": spec.device,
    }
    logger.log_runtime(
        event="task1_onnx_export_started",
        adapter="Task1OnnxExport",
        diagnostics=payload,
    )

    if spec.runtime_name != "ultralytics":
        payload["error"] = f"unsupported_export_runtime:{spec.runtime_name}"
        logger.log_error(
            event="task1_onnx_export_failed",
            adapter="Task1OnnxExport",
            diagnostics=payload,
        )
        return payload
    if importlib.util.find_spec("ultralytics") is None:
        payload["error"] = "missing_dependency:ultralytics"
        logger.log_error(
            event="task1_onnx_export_failed",
            adapter="Task1OnnxExport",
            diagnostics=payload,
        )
        return payload
    if not spec.source_model_path.exists():
        payload["error"] = f"missing_model_file:{spec.source_model_path}"
        logger.log_error(
            event="task1_onnx_export_failed",
            adapter="Task1OnnxExport",
            diagnostics=payload,
        )
        return payload

    spec.output_model_path.parent.mkdir(parents=True, exist_ok=True)
    spec.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]

        model = YOLO(str(spec.source_model_path))
        exported_path = Path(
            model.export(
                format="onnx",
                imgsz=spec.imgsz,
                opset=spec.opset,
                dynamic=spec.dynamic,
                half=spec.half,
                nms=False,
                simplify=False,
                batch=1,
                device=spec.device,
                verbose=False,
            )
        )
        if exported_path.resolve() != spec.output_model_path.resolve():
            spec.output_model_path.write_bytes(exported_path.read_bytes())
        names = getattr(model, "names", {}) or {}
        metadata_payload = {
            "candidate_name": candidate_name,
            "source_model_path": str(spec.source_model_path),
            "output_model_path": str(spec.output_model_path),
            "input_shape": [1, 3, spec.imgsz, spec.imgsz],
            "output_contract": "raw_yolo_predictions",
            "class_names": _normalize_class_names(names),
            "export_runtime": "ultralytics",
            "device": spec.device,
            "opset": spec.opset,
            "dynamic": spec.dynamic,
            "half": spec.half,
        }
        spec.metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
        payload.update(
            {
                "success": True,
                "exported_path": str(spec.output_model_path),
                "metadata_path": str(spec.metadata_path),
                "class_count": len(metadata_payload["class_names"]),
                "input_shape": metadata_payload["input_shape"],
            }
        )
        logger.log_runtime(
            event="task1_onnx_export_finished",
            adapter="Task1OnnxExport",
            diagnostics=payload,
        )
        return payload
    except Exception as exc:
        payload["error"] = str(exc)
        logger.log_error(
            event="task1_onnx_export_failed",
            adapter="Task1OnnxExport",
            diagnostics=payload,
        )
        return payload


def export_spec_to_dict(spec: OnnxExportSpec) -> dict[str, Any]:
    data = asdict(spec)
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def _normalize_class_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(item) for item in names]
    return []
