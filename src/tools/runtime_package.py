from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.tools.report_paths import EXPORT_REPORTS_DIR


def prepare_runtime_package(
    runtime_settings: MvpRuntimeSettings,
    *,
    base_dir: str | Path = "final_runtime",
    reports_dir: str | Path = EXPORT_REPORTS_DIR,
) -> dict[str, Any]:
    base = Path(base_dir)
    reports_base = Path(reports_dir)
    config_dir = base / "config"
    artifacts_onnx_dir = base / "artifacts" / "onnx"
    artifacts_trt_dir = base / "artifacts" / "trt"
    logs_dir = base / "logs"
    cache_dir = base / "cache"
    for directory in (config_dir, artifacts_onnx_dir, artifacts_trt_dir, logs_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    validation_status = _collect_validation_statuses(reports_base)
    copied_artifacts = {
        "yolo26n_onnx": _copy_if_exists(
            runtime_settings.resolve_task1_onnx_path("yolo26n") or reports_base / "models" / "yolo26n.onnx",
            artifacts_onnx_dir / "yolo26n.onnx",
        ),
        "yolo11n_onnx": _copy_if_exists(
            runtime_settings.resolve_task1_onnx_path("yolo11n") or reports_base / "models" / "yolo11n.onnx",
            artifacts_onnx_dir / "yolo11n.onnx",
        ),
        "yolo26n_trt": _copy_if_exists(
            runtime_settings.resolve_task1_trt_path("yolo26n") or reports_base / "models" / "yolo26n.engine",
            artifacts_trt_dir / "yolo26n.engine",
        ),
    }
    _copy_if_exists(
        runtime_settings.resolve_task1_export_metadata_path("yolo26n"),
        artifacts_onnx_dir / "yolo26n.metadata.json",
    )
    _copy_if_exists(
        runtime_settings.resolve_task1_export_metadata_path("yolo11n"),
        artifacts_onnx_dir / "yolo11n.metadata.json",
    )
    _copy_if_exists(
        runtime_settings.resolve_task1_trt_metadata_path("yolo26n"),
        artifacts_trt_dir / "yolo26n.engine.metadata.json",
    )

    manifest = _build_model_manifest(
        runtime_settings=runtime_settings,
        copied_artifacts=copied_artifacts,
        validation_status=validation_status,
    )
    manifest_path = config_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    runtime_toml = _render_runtime_toml(
        runtime_settings=runtime_settings,
        manifest=manifest,
        base_dir=base,
    )
    runtime_toml_path = config_dir / "runtime.toml"
    runtime_toml_path.write_text(runtime_toml, encoding="utf-8")

    readme_path = base / "README.md"
    readme_path.write_text(_render_runtime_readme(), encoding="utf-8")
    return {
        "base_dir": str(base),
        "config_path": str(runtime_toml_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def _collect_validation_statuses(reports_base: Path) -> dict[str, bool]:
    statuses = {
        "onnxruntime:yolo26n": _read_validation_flag(reports_base / "task1_yolo26n_onnx_export_summary.json"),
        "onnxruntime:yolo11n": _read_validation_flag(reports_base / "task1_yolo11n_onnx_export_summary.json"),
        "tensorrt:yolo26n": _read_validation_flag(reports_base / "task1_trt_export_summary.json"),
    }
    return statuses


def _build_model_manifest(
    *,
    runtime_settings: MvpRuntimeSettings,
    copied_artifacts: dict[str, str | None],
    validation_status: dict[str, bool],
) -> dict[str, Any]:
    required = [
        _artifact_entry(
            artifact_id="yolo26n.pt",
            path=_string_or_none(runtime_settings.resolve_task1_model_path("yolo26n")),
            candidate="yolo26n",
            runtime="ultralytics",
            required=True,
            validation_status=True,
        ),
        _artifact_entry(
            artifact_id="yolo26n.onnx",
            path=copied_artifacts.get("yolo26n_onnx"),
            candidate="yolo26n",
            runtime="onnxruntime",
            required=True,
            validation_status=validation_status.get("onnxruntime:yolo26n", False),
        ),
    ]
    optional = [
        _artifact_entry(
            artifact_id="yolo26n.engine",
            path=copied_artifacts.get("yolo26n_trt"),
            candidate="yolo26n",
            runtime="tensorrt",
            required=False,
            validation_status=validation_status.get("tensorrt:yolo26n", False),
        ),
        _artifact_entry(
            artifact_id="yolo11n.pt",
            path=_string_or_none(runtime_settings.resolve_task1_model_path("yolo11n")),
            candidate="yolo11n",
            runtime="ultralytics",
            required=False,
            validation_status=True,
        ),
        _artifact_entry(
            artifact_id="yolo11n.onnx",
            path=copied_artifacts.get("yolo11n_onnx"),
            candidate="yolo11n",
            runtime="onnxruntime",
            required=False,
            validation_status=validation_status.get("onnxruntime:yolo11n", False),
        ),
    ]
    return {
        "required": required,
        "optional": optional,
        "runtime_order": list(runtime_settings.task1_runtime_order),
    }


def _artifact_entry(
    *,
    artifact_id: str,
    path: str | None,
    candidate: str,
    runtime: str,
    required: bool,
    validation_status: bool,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "path": path,
        "candidate": candidate,
        "runtime": runtime,
        "required": required,
        "validation_status": validation_status,
    }


def _render_runtime_toml(
    *,
    runtime_settings: MvpRuntimeSettings,
    manifest: dict[str, Any],
    base_dir: Path,
) -> str:
    default_runtime = "tensorrt" if _manifest_entry_valid(manifest, "yolo26n.engine") else "onnxruntime"
    lines = [
        "[task1]",
        'detector_backend = "yolo26n"',
        f'model_runtime = "{default_runtime}"',
        f'task1_device = "{runtime_settings.task1_device or "cuda:0"}"',
        f'trt_precision = "{runtime_settings.task1_trt_precision}"',
        f'trt_warmup_runs = {int(runtime_settings.task1_trt_warmup_runs)}',
        f'onnx_conf_threshold_offset = {float(runtime_settings.task1_onnx_conf_threshold_offset)}',
        "runtime_order = [",
    ]
    for item in runtime_settings.task1_runtime_order:
        lines.append(f'  "{item}",')
    lines.extend(
        [
            "]",
            "",
            "[paths]",
            f'log_dir = "{(base_dir / "logs").as_posix()}"',
            f'cache_dir = "{(base_dir / "cache").as_posix()}"',
            f'manifest_path = "{(base_dir / "config" / "model_manifest.json").as_posix()}"',
            "",
            "[smoke]",
            'default_mode = "batch"',
            "default_max_frames = 2",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_runtime_readme() -> str:
    return (
        "# Final Runtime\n\n"
        "Bu klasor Faz 8 itibariyla yarismaya yakin runtime artefact ve config duzenini toplar.\n"
        "Model manifesti zorunlu/opsiyonel artefact listesini, runtime.toml ise smoke ayarlarini icerir.\n"
    )


def _copy_if_exists(source: str | Path | None, destination: Path) -> str | None:
    if source is None:
        return None
    source_path = Path(source)
    if not source_path.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(destination)


def _read_validation_flag(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("validation_passed"))


def _manifest_entry_valid(manifest: dict[str, Any], artifact_id: str) -> bool:
    for section in ("required", "optional"):
        for item in manifest.get(section, []):
            if item.get("artifact_id") == artifact_id:
                return bool(item.get("validation_status")) and bool(item.get("path"))
    return False


def _string_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None
