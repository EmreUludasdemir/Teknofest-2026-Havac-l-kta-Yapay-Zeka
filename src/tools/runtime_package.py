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
    del reports_dir
    base = Path(base_dir)
    config_dir = base / "config"
    artifacts_dir = base / "artifacts" / "task3"
    logs_dir = base / "logs"
    cache_dir = base / "cache"
    for directory in (config_dir, artifacts_dir, logs_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copied_weight = _copy_if_exists(
        runtime_settings.task3_yoloe_weight_path,
        artifacts_dir / Path(runtime_settings.task3_yoloe_weight_path).name,
    )
    manifest = _build_runtime_manifest(
        runtime_settings=runtime_settings,
        copied_weight=copied_weight,
    )
    manifest_path = config_dir / "task3_runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    runtime_toml_path = config_dir / "runtime.toml"
    runtime_toml_path.write_text(_render_runtime_toml(runtime_settings, manifest_path, base), encoding="utf-8")

    readme_path = base / "README.md"
    readme_path.write_text(_render_runtime_readme(), encoding="utf-8")
    return {
        "base_dir": str(base),
        "config_path": str(runtime_toml_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def _build_runtime_manifest(
    *,
    runtime_settings: MvpRuntimeSettings,
    copied_weight: str | None,
) -> dict[str, Any]:
    return {
        "task": "task3",
        "required": [
            {
                "artifact_id": Path(runtime_settings.task3_yoloe_weight_path).name,
                "path": copied_weight,
                "runtime": "ultralytics",
                "required": True,
                "validation_status": copied_weight is not None,
            }
        ],
        "references": {
            "reference_dir": str(runtime_settings.task3_reference_dir),
            "eval_reference_dir": str(runtime_settings.task3_eval_reference_dir),
            "eval_manifest_path": str(runtime_settings.task3_eval_manifest_path),
        },
    }


def _render_runtime_toml(runtime_settings: MvpRuntimeSettings, manifest_path: Path, base_dir: Path) -> str:
    lines = [
        "[task3]",
        f'mode = "{runtime_settings.task3_mode}"',
        f'reference_dir = "{Path(runtime_settings.task3_reference_dir).as_posix()}"',
        f'eval_reference_dir = "{Path(runtime_settings.task3_eval_reference_dir).as_posix()}"',
        f'eval_manifest_path = "{Path(runtime_settings.task3_eval_manifest_path).as_posix()}"',
        f'yoloe_weight_path = "{Path(runtime_settings.task3_yoloe_weight_path).as_posix()}"',
        f'yoloe_device = "{runtime_settings.task3_yoloe_device or "auto"}"',
        f"yoloe_allow_cpu = {str(bool(runtime_settings.task3_yoloe_allow_cpu)).lower()}",
        f"orb_features = {int(runtime_settings.task3_orb_features)}",
        f"match_min_inliers = {int(runtime_settings.task3_match_min_inliers)}",
        f"min_score = {float(runtime_settings.task3_min_score)}",
        f"yoloe_min_score = {float(runtime_settings.task3_yoloe_min_score)}",
        f"yoloe_thermal_min_score = {float(runtime_settings.task3_yoloe_thermal_min_score)}",
        "",
        "[paths]",
        f'log_dir = "{(base_dir / "logs").as_posix()}"',
        f'cache_dir = "{(base_dir / "cache").as_posix()}"',
        f'manifest_path = "{manifest_path.as_posix()}"',
        "",
        "[smoke]",
        'default_mode = "batch"',
        "default_max_frames = 2",
    ]
    return "\n".join(lines) + "\n"


def _render_runtime_readme() -> str:
    return (
        "# Final Runtime\n\n"
        "Bu klasor Task 3 batch/sequential adapter smoke calismalari icin gerekli runtime artefact ve config duzenini toplar.\n"
        "Task 1 ve Task 2 artefactlari bu repodan cikarilmistir.\n"
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
