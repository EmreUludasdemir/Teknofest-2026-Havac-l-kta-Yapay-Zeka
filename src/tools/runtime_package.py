from __future__ import annotations

import json
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import MvpRuntimeSettings, SequentialProtocolSettings


@dataclass(slots=True)
class RuntimeBootstrap:
    config_path: Path
    manifest_path: Path
    runtime_config: dict[str, Any]
    manifest: dict[str, Any]
    runtime_settings: MvpRuntimeSettings
    sequential_settings: SequentialProtocolSettings
    task1_runtime_order: list[str]


def prepare_runtime_package(
    runtime_settings: MvpRuntimeSettings,
    *,
    base_dir: str | Path = "final_runtime",
    reports_dir: str | Path = "reports/export",
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
        artifacts_onnx_dir=artifacts_onnx_dir,
        artifacts_trt_dir=artifacts_trt_dir,
    )
    manifest_path = config_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    runtime_toml_path = config_dir / "runtime.toml"
    runtime_toml_path.write_text(
        _render_runtime_toml(
            runtime_settings=runtime_settings,
            manifest=manifest,
            base_dir=base,
        ),
        encoding="utf-8",
    )

    (base / "README.md").write_text(_render_runtime_readme(manifest), encoding="utf-8")
    (base / "CHECKLIST.md").write_text(_render_runtime_checklist(manifest), encoding="utf-8")
    (base / "RUNBOOK.md").write_text(_render_runtime_runbook(manifest), encoding="utf-8")
    (base / "SEQUENTIAL_COMPATIBILITY.md").write_text(
        _render_sequential_compatibility_guide(),
        encoding="utf-8",
    )
    (base / "OPERATOR_DRILL.md").write_text(_render_operator_drill_guide(), encoding="utf-8")
    (base / "run_competition.ps1").write_text(_render_competition_wrapper(), encoding="utf-8")
    return {
        "base_dir": str(base),
        "config_path": str(runtime_toml_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def load_runtime_bootstrap(
    config_path: str | Path,
    *,
    production: bool = True,
    base_url_override: str | None = None,
    username_override: str | None = None,
    password_override: str | None = None,
) -> RuntimeBootstrap:
    resolved_config_path = Path(config_path)
    runtime_config = load_runtime_config(resolved_config_path)
    manifest_path = _resolve_manifest_path(runtime_config, resolved_config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task1_runtime_order = select_enabled_runtime_order(
        list(runtime_config.get("task1", {}).get("runtime_order", [])),
        manifest,
        production=production,
    )
    runtime_settings = build_runtime_settings_from_package_config(
        runtime_config,
        manifest,
        task1_runtime_order=task1_runtime_order,
        production=production,
    )
    sequential_settings = build_sequential_protocol_settings(
        runtime_config,
        base_url_override=base_url_override,
        username_override=username_override,
        password_override=password_override,
    )
    return RuntimeBootstrap(
        config_path=resolved_config_path,
        manifest_path=manifest_path,
        runtime_config=runtime_config,
        manifest=manifest,
        runtime_settings=runtime_settings,
        sequential_settings=sequential_settings,
        task1_runtime_order=task1_runtime_order,
    )


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def build_runtime_settings_from_package_config(
    runtime_config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    task1_runtime_order: list[str] | None = None,
    production: bool = True,
) -> MvpRuntimeSettings:
    task1 = runtime_config.get("task1", {})
    warmup = runtime_config.get("warmup", {})
    task2 = runtime_config.get("task2", {})
    manifest_items = list(manifest.get("required", [])) + list(manifest.get("optional", []))

    resolved_order = task1_runtime_order or select_enabled_runtime_order(
        list(task1.get("runtime_order", [])),
        manifest,
        production=production,
    )
    enabled_stage_ids = [item for item in resolved_order if item != "synthetic"]

    task1_paths = {
        str(item["candidate"]): str(item["path"])
        for item in manifest_items
        if item.get("runtime") == "ultralytics"
        and item.get("path")
        and (not production or item.get("production_enabled", False))
    }
    onnx_paths = {
        str(item["candidate"]): str(item["path"])
        for item in manifest_items
        if item.get("runtime") == "onnxruntime"
        and item.get("path")
        and item.get("validation_status") is True
        and (not production or item.get("production_enabled", False))
    }
    onnx_metadata_paths = {
        str(item["candidate"]): str(item["metadata_path"])
        for item in manifest_items
        if item.get("runtime") == "onnxruntime"
        and item.get("metadata_path")
        and (not production or item.get("production_enabled", False))
    }
    trt_paths = {
        str(item["candidate"]): str(item["path"])
        for item in manifest_items
        if item.get("runtime") == "tensorrt"
        and item.get("path")
        and item.get("validation_status") is True
        and (not production or item.get("production_enabled", False))
    }
    trt_metadata_paths = {
        str(item["candidate"]): str(item["metadata_path"])
        for item in manifest_items
        if item.get("runtime") == "tensorrt"
        and item.get("metadata_path")
        and (not production or item.get("production_enabled", False))
    }
    return MvpRuntimeSettings(
        task1_detector_backend=str(task1.get("detector_backend", "yolo26n")),
        task1_model_runtime=str(task1.get("model_runtime", "tensorrt")),
        task1_device=str(task1.get("task1_device", "cuda:0")),
        task1_trt_precision=str(task1.get("trt_precision", "fp16")),
        task1_onnx_conf_threshold_offset=float(task1.get("onnx_conf_threshold_offset", 0.10)),
        task1_runtime_order=resolved_order,
        task1_enabled_stage_ids=enabled_stage_ids,
        task1_candidate_paths=task1_paths,
        task1_onnx_candidate_paths=onnx_paths,
        task1_onnx_metadata_candidate_paths=onnx_metadata_paths,
        task1_trt_candidate_paths=trt_paths,
        task1_trt_metadata_candidate_paths=trt_metadata_paths,
        task1_trt_warmup_runs=int(warmup.get("task1_trt_warmup_runs", 3)),
        task1_onnx_warmup_runs=int(warmup.get("task1_onnx_warmup_runs", 1)),
        task1_native_warmup_runs=int(warmup.get("task1_native_warmup_runs", 1)),
        task2_phase_primary_response_min_thermal=float(
            task2.get("phase_primary_response_min_thermal", 0.45)
        ),
        task2_confidence_floor_thermal=float(task2.get("confidence_floor_thermal", 0.40)),
        task2_sensor_hint_max_weight_thermal=float(
            task2.get("sensor_hint_max_weight_thermal", 0.15)
        ),
        task2_z_update_scale_thermal=float(task2.get("z_update_scale_thermal", 0.0)),
    )


def build_sequential_protocol_settings(
    runtime_config: dict[str, Any],
    *,
    base_url_override: str | None = None,
    username_override: str | None = None,
    password_override: str | None = None,
) -> SequentialProtocolSettings:
    server = runtime_config.get("server", {})
    runtime = runtime_config.get("runtime", {})
    sequential = runtime_config.get("sequential", {})
    return SequentialProtocolSettings(
        base_url=str(base_url_override or server.get("base_url", "http://127.0.0.1:5000/")),
        username=str(username_override or server.get("username", "team")),
        password=str(password_override or server.get("password", "password")),
        request_timeout_s=float(server.get("request_timeout_s", 5.0)),
        image_timeout_s=float(server.get("image_timeout_s", 10.0)),
        profile_name=str(runtime.get("profile_name", "competition_runtime")),
        wire_style=str(runtime.get("wire_style", "teknofest_v1")),
        wire_profile=str(sequential.get("wire_profile", "official_current")),
        warmup_timeout_s=float(runtime.get("warmup_timeout_s", 3.0)),
        first_frame_timeout_s=float(runtime.get("first_frame_timeout_s", 5.0)),
        retry_policy={
            "max_retries": int(runtime.get("max_retries", 2)),
            "backoff_s": float(runtime.get("backoff_s", 0.25)),
        },
    )


def select_enabled_runtime_order(
    runtime_order: list[str],
    manifest: dict[str, Any],
    *,
    production: bool = True,
) -> list[str]:
    if not runtime_order:
        runtime_order = list(MvpRuntimeSettings().task1_runtime_order)
    manifest_map = {
        _stage_id_for_item(item): item
        for item in list(manifest.get("required", [])) + list(manifest.get("optional", []))
        if item.get("runtime")
    }
    enabled_order: list[str] = []
    for stage_id in runtime_order:
        normalized_stage = str(stage_id).strip().lower()
        if normalized_stage == "synthetic":
            enabled_order.append("synthetic")
            continue
        item = manifest_map.get(normalized_stage)
        if item is None:
            continue
        if production and not bool(item.get("production_enabled")):
            continue
        runtime_name = str(item.get("runtime"))
        if runtime_name in {"onnxruntime", "tensorrt"} and not bool(item.get("validation_status")):
            continue
        enabled_order.append(normalized_stage)
    if "synthetic" not in enabled_order:
        enabled_order.append("synthetic")
    return enabled_order


def _collect_validation_statuses(reports_base: Path) -> dict[str, bool]:
    return {
        "onnxruntime:yolo26n": _read_validation_flag(reports_base / "task1_yolo26n_onnx_export_summary.json"),
        "onnxruntime:yolo11n": _read_validation_flag(reports_base / "task1_yolo11n_onnx_export_summary.json"),
        "tensorrt:yolo26n": _read_validation_flag(reports_base / "task1_trt_export_summary.json"),
    }


def _build_model_manifest(
    *,
    runtime_settings: MvpRuntimeSettings,
    copied_artifacts: dict[str, str | None],
    validation_status: dict[str, bool],
    artifacts_onnx_dir: Path,
    artifacts_trt_dir: Path,
) -> dict[str, Any]:
    required = [
        _artifact_entry(
            artifact_id="yolo26n.pt",
            path=_string_or_none(runtime_settings.resolve_task1_model_path("yolo26n")),
            metadata_path=None,
            candidate="yolo26n",
            runtime="ultralytics",
            required=True,
            validation_status=True,
            production_enabled=True,
        ),
        _artifact_entry(
            artifact_id="yolo26n.onnx",
            path=copied_artifacts.get("yolo26n_onnx"),
            metadata_path=_string_or_none(artifacts_onnx_dir / "yolo26n.metadata.json"),
            candidate="yolo26n",
            runtime="onnxruntime",
            required=True,
            validation_status=validation_status.get("onnxruntime:yolo26n", False),
            production_enabled=True,
        ),
    ]
    optional = [
        _artifact_entry(
            artifact_id="yolo26n.engine",
            path=copied_artifacts.get("yolo26n_trt"),
            metadata_path=_string_or_none(artifacts_trt_dir / "yolo26n.engine.metadata.json"),
            candidate="yolo26n",
            runtime="tensorrt",
            required=False,
            validation_status=validation_status.get("tensorrt:yolo26n", False),
            production_enabled=True,
        ),
        _artifact_entry(
            artifact_id="yolo11n.pt",
            path=_string_or_none(runtime_settings.resolve_task1_model_path("yolo11n")),
            metadata_path=None,
            candidate="yolo11n",
            runtime="ultralytics",
            required=False,
            validation_status=True,
            production_enabled=True,
        ),
        _artifact_entry(
            artifact_id="yolo11n.onnx",
            path=copied_artifacts.get("yolo11n_onnx"),
            metadata_path=_string_or_none(artifacts_onnx_dir / "yolo11n.metadata.json"),
            candidate="yolo11n",
            runtime="onnxruntime",
            required=False,
            validation_status=validation_status.get("onnxruntime:yolo11n", False),
            production_enabled=False,
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
    metadata_path: str | None,
    candidate: str,
    runtime: str,
    required: bool,
    validation_status: bool,
    production_enabled: bool,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "path": path,
        "metadata_path": metadata_path,
        "candidate": candidate,
        "runtime": runtime,
        "required": required,
        "validation_status": validation_status,
        "production_enabled": production_enabled,
    }


def _render_runtime_toml(
    *,
    runtime_settings: MvpRuntimeSettings,
    manifest: dict[str, Any],
    base_dir: Path,
) -> str:
    default_runtime = "tensorrt"
    if not _manifest_entry_valid(manifest, "yolo26n.engine", production=True):
        default_runtime = "onnxruntime"
        if not _manifest_entry_valid(manifest, "yolo26n.onnx", production=True):
            default_runtime = "ultralytics"

    lines = [
        "[server]",
        'base_url = "http://127.0.0.1:5000/"',
        'username = "team"',
        'password = "password"',
        "request_timeout_s = 5.0",
        "image_timeout_s = 10.0",
        "",
        "[runtime]",
        'mode = "sequential"',
        'profile_name = "competition_runtime"',
        'wire_style = "teknofest_v1"',
        'first_frame_failure_policy = "retry_same_frame_then_graceful_degrade"',
        "warmup_timeout_s = 3.0",
        "first_frame_timeout_s = 5.0",
        "max_retries = 2",
        "backoff_s = 0.25",
        "",
        "[sequential]",
        'wire_profile = "official_current"',
        '# Gerekirse draft profil acilabilir: draft_with_undefined',
        "",
        "[task1]",
        'detector_backend = "yolo26n"',
        f'model_runtime = "{default_runtime}"',
        f'task1_device = "{runtime_settings.task1_device or "cuda:0"}"',
        f'trt_precision = "{runtime_settings.task1_trt_precision}"',
        f'onnx_conf_threshold_offset = {float(runtime_settings.task1_onnx_conf_threshold_offset)}',
        "runtime_order = [",
    ]
    for item in runtime_settings.task1_runtime_order:
        lines.append(f'  "{item}",')
    lines.extend(
        [
            "]",
            "",
            "[warmup]",
            f"task1_trt_warmup_runs = {int(runtime_settings.task1_trt_warmup_runs)}",
            f"task1_onnx_warmup_runs = {int(runtime_settings.task1_onnx_warmup_runs)}",
            f"task1_native_warmup_runs = {int(runtime_settings.task1_native_warmup_runs)}",
            "",
            "[task2]",
            f"phase_primary_response_min_thermal = {float(runtime_settings.task2_phase_primary_response_min_thermal)}",
            f"confidence_floor_thermal = {float(runtime_settings.task2_confidence_floor_thermal)}",
            f"sensor_hint_max_weight_thermal = {float(runtime_settings.task2_sensor_hint_max_weight_thermal)}",
            f"z_update_scale_thermal = {float(runtime_settings.task2_z_update_scale_thermal)}",
            "",
            "[paths]",
            f'log_dir = "{(base_dir / "logs").as_posix()}"',
            f'cache_dir = "{(base_dir / "cache").as_posix()}"',
            f'manifest_path = "{(base_dir / "config" / "model_manifest.json").as_posix()}"',
            "",
            "[smoke]",
            'default_mode = "sequential"',
            "default_max_frames = 2",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_runtime_readme(manifest: dict[str, Any]) -> str:
    return (
        "# Final Runtime\n\n"
        "Bu klasor yarismada kullanilacak production runtime paketinin yerel iskeletidir.\n\n"
        "## Klasor Yapisi\n"
        "- `config/runtime.toml`: tek production config kaynagi\n"
        "- `config/model_manifest.json`: model/artifact ve production eligibility kararlari\n"
        "- `artifacts/onnx/`: ONNX dosyalari\n"
        "- `artifacts/trt/`: TensorRT engine ve metadata\n"
        "- `logs/`: runtime loglari\n"
        "- `cache/`: gecici yerel cache\n\n"
        "## Production Task 1 Karar Agaci\n"
        "- `tensorrt:yolo26n`\n"
        "- `onnxruntime:yolo26n`\n"
        "- `ultralytics:yolo26n`\n"
        "- `ultralytics:yolo11n`\n"
        "- `synthetic`\n\n"
        "## Onemli Notlar\n"
        "- `onnxruntime:yolo11n` export edilmis olsa da parity basarisiz oldugu icin production disabled durumundadir.\n"
        "- Varsayilan sequential wire profili `official_current` olup Task 3 undefined object sonuclari canonical modelde tutulur, production wire'a gonderilmez.\n"
        "- Teknik sartname taslagi undefined object isterse `final_runtime/config/runtime.toml` icinde `sequential.wire_profile = \"draft_with_undefined\"` acilabilir.\n"
        "- Ilk gercek frame alinmadan once warm-up tamamlanmis olmalidir.\n\n"
        "## Model Beklentileri\n"
        f"{_render_manifest_section(manifest)}\n"
        "## Baslatma\n"
        "- Production: `C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe tools/run_competition_runtime.py --config final_runtime/config/runtime.toml`\n"
        "- Batch smoke: `C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe tools/run_runtime_smoke.py --mode batch --frames 2`\n"
        "- Sequential smoke: `C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe tools/run_runtime_smoke.py --mode sequential --frames 2`\n"
    )


def _render_runtime_checklist(manifest: dict[str, Any]) -> str:
    return (
        "# Competition Checklist\n\n"
        "## Environment Preflight\n"
        "- Aktif interpreter `C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe` mi?\n"
        "- Python 3.10 aktif mi?\n"
        "- GPU driver ve CUDA gorunuyor mu?\n"
        "- `onnxruntime-gpu` ve TensorRT ortamda kurulu mu?\n"
        "- `final_runtime/config/runtime.toml` duzgun mu?\n\n"
        "## Dosya Kontrolu\n"
        f"{_render_manifest_checklist(manifest)}\n"
        "## Warm-up Kontrolu\n"
        "- Session acildiktan sonra warm-up loglari olusuyor mu?\n"
        "- `warmup_stage_completed` eventi goruluyor mu?\n"
        "- Ilk frame warm-up bitmeden istenmiyor mu?\n\n"
        "## Zorunlu Son Smoke\n"
        "- `C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe tools/run_runtime_smoke.py --mode sequential --frames 2`\n"
        "- `competition_runtime_summary.json` icinde aktif stage production zincirinden biri mi?\n\n"
        "## Sequential Wire Profili\n"
        "- Varsayilan: `official_current`\n"
        "- Draft undefined-object profili yalniz final endpoint bunu zorunlu kilarsa acilsin.\n"
        "- Draft profile acilirsa payload icine `detected_undefined_objects` girdigi validator ile dogrulansin.\n\n"
        "## Failure Mode Kontrolu\n"
        "- TRT fail ederse ONNX yolo26n devreye giriyor mu?\n"
        "- ONNX fail ederse native yolo26n devreye giriyor mu?\n"
        "- yolo26n tamamen fail ederse native yolo11n devreye giriyor mu?\n"
        "- Son fallback olarak synthetic aktif mi?\n"
    )


def _render_runtime_runbook(manifest: dict[str, Any]) -> str:
    return (
        "# Competition Day Runbook\n\n"
        "1. `requirements-runtime.txt` ile ortam hazirligini tamamla.\n"
        "2. `final_runtime/config/runtime.toml` icinde base_url, kullanici ve sifreyi doldur.\n"
        "3. Model dosyalarinin `model_manifest.json` ile uyumlu oldugunu kontrol et.\n"
        "4. `C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe tools/run_runtime_smoke.py --mode sequential --frames 2` ile son yerel smoke'u kos.\n"
        "5. Production komutu: `C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe tools/run_competition_runtime.py --config final_runtime/config/runtime.toml`\n"
        "6. Loglarda su eventleri gor: `warmup_policy_started`, `warmup_stage_completed`, `sequential_prediction_sent`.\n"
        "7. Ilk gercek frame fail ederse ayni frame icin bir sonraki eligible backend otomatik denenir.\n"
        "8. Prediction retry limiti dolarsa bir sonraki frame istenmez; oturum kontrollu kapatilir.\n\n"
        "## Sequential Wire Notu\n"
        "- Varsayilan wire profili `official_current` olarak kalir.\n"
        "- Final endpoint `detected_undefined_objects` isterse `draft_with_undefined` profilini acip validator ve mock ile tekrar smoke alin.\n\n"
        "## Native-only Fallback Karari\n"
        "- `yolo11n` ONNX parity kanitlanmadigi icin production kullanimi kapatildi.\n"
        "- `yolo11n` yalniz native fallback olarak kullanilir.\n\n"
        "## Required / Optional Ozet\n"
        f"{_render_manifest_section(manifest)}"
    )


def _render_competition_wrapper() -> str:
    return (
        "$ErrorActionPreference = 'Stop'\n"
        "$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path\n"
        "$gpuPython = 'C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe'\n"
        "if (Test-Path $gpuPython) {\n"
        "  & $gpuPython \"$scriptDir\\..\\tools\\run_competition_runtime.py\" --config \"$scriptDir\\config\\runtime.toml\"\n"
        "} else {\n"
        "  python \"$scriptDir\\..\\tools\\run_competition_runtime.py\" --config \"$scriptDir\\config\\runtime.toml\"\n"
        "}\n"
    )


def _render_sequential_compatibility_guide() -> str:
    return (
        "# Sequential Compatibility Checklist\n\n"
        "Bu belge mevcut production sequential adapter davranisinin son bilinen resmi repo/API sekliyle uyumunu takip etmek icindir.\n\n"
        "## Referans Snapshot\n"
        "- Son okunabilen resmi repo snapshot commit'i: `ce249fc8042ada788df438d2c0a1b76e28d6beaa`\n"
        "- Bu snapshot'ta kanitlanmis endpointler batch-sekillidir: `auth/`, `frames/`, `translation/`, `prediction/`, `session/`.\n"
        "- Sequential endpoint ailesi (`session/open`, `session/next`, `session/prediction`, `session/close`) production mock/profile varsayimidir; final sunucu yayini olmadan tam kanit sayilmaz.\n\n"
        "## Kanitlanmis Davranislar\n"
        "- Production runner yalniz sequential calisir.\n"
        "- `open_session -> warm-up -> fetch_next_frame -> send_wire_prediction -> fetch_next_frame` sirasi korunur.\n"
        "- Bir frame icin sonuc gonderilmeden sonraki frame istenmez.\n"
        "- Varsayilan wire profili `official_current` olup yalniz `session_id`, `frame_id`, `frame`, `detected_objects`, `detected_translations` gonderilir.\n"
        "- `detected_undefined_objects` canonical modelde tutulur ama varsayilan wire'a cikmaz.\n\n"
        "## Dormant Switch Path\n"
        "- Teknik sartname taslagindaki undefined-object alanlari gerekirse `sequential.wire_profile = \"draft_with_undefined\"` ile acilir.\n"
        "- Bu degisiklikten sonra sequential validator ve mock smoke yeniden alinmalidir.\n\n"
        "## Son Kontrol Listesi\n"
        "- Son erisilebilir resmi repo/API sekli yeniden okundu mu?\n"
        "- Endpoint adlari (`auth/open/next/prediction/close`) final sistemle uyumlu mu?\n"
        "- `frame_id` ve `session_id` alanlari final wire ile uyumlu mu?\n"
        "- Undefined object alanlari final endpoint tarafindan acikca isteniyor mu?\n"
        "- Retry ve timeout davranisi gercek sunucu ile tekrar smoke edildi mi?\n"
    )


def _render_operator_drill_guide() -> str:
    return (
        "# Operator Drill\n\n"
        "## Hazirlik\n"
        "1. GPU venv aktif: `C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe`\n"
        "2. `requirements-runtime.txt` yuklu.\n"
        "3. `final_runtime/config/model_manifest.json` icindeki required artifact'lar mevcut.\n\n"
        "## Drill Adimlari\n"
        "1. `C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe tools/run_runtime_smoke.py --mode sequential --frames 2`\n"
        "2. `C:\\Users\\Emre\\.venvs\\teknofest-gpu\\Scripts\\python.exe tools/run_competition_runtime.py --config final_runtime/config/runtime.toml --max-frames 2 --base-url <mock-url>`\n"
        "3. Loglarda `warmup_policy_started`, `warmup_stage_completed`, `sequential_prediction_sent` eventlerini dogrula.\n"
        "4. `competition_runtime_summary.json` icinde `active_task1_stage` alanini kontrol et.\n"
        "5. Prediction fail olursa bir sonraki frame istenmedigini logdan teyit et.\n"
        "6. Aktif stage `tensorrt:yolo26n`, `onnxruntime:yolo26n` veya `ultralytics:yolo26n` degilse production oturumunu baslatma.\n\n"
        "## Durdur / Yeniden Baslat Kurali\n"
        "- Aktif stage `synthetic` veya `ultralytics:yolo11n` ise production oturumu durdur.\n"
        "- Warm-up tamamlanmadiysa oturumu yeniden baslat.\n"
        "- Prediction retry limiti biterse logu sakla ve yeni oturum ac.\n"
    )


def _render_manifest_section(manifest: dict[str, Any]) -> str:
    lines = ["### Manifest"]
    for section_name in ("required", "optional"):
        lines.append(f"- {section_name}:")
        for item in manifest.get(section_name, []):
            lines.append(
                "  - {artifact} | runtime={runtime} | candidate={candidate} | validated={validated} | production_enabled={enabled}".format(
                    artifact=item.get("artifact_id"),
                    runtime=item.get("runtime"),
                    candidate=item.get("candidate"),
                    validated=item.get("validation_status"),
                    enabled=item.get("production_enabled"),
                )
            )
    return "\n".join(lines) + "\n"


def _render_manifest_checklist(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    for section_name in ("required", "optional"):
        for item in manifest.get(section_name, []):
            lines.append(
                "- [{status}] {artifact} ({runtime}, production_enabled={enabled})".format(
                    status="x" if item.get("path") else " ",
                    artifact=item.get("artifact_id"),
                    runtime=item.get("runtime"),
                    enabled=item.get("production_enabled"),
                )
            )
    return "\n".join(lines) + "\n"


def _resolve_manifest_path(runtime_config: dict[str, Any], config_path: Path) -> Path:
    raw_manifest_path = str(runtime_config.get("paths", {}).get("manifest_path", "config/model_manifest.json"))
    manifest_path = Path(raw_manifest_path)
    if manifest_path.is_absolute():
        return manifest_path
    candidates: list[Path] = []
    if raw_manifest_path.startswith("final_runtime/"):
        try:
            candidates.append((config_path.parent.parent / manifest_path.relative_to("final_runtime")).resolve())
        except Exception:
            pass
    candidates.extend(
        [
            (config_path.parent / manifest_path).resolve(),
            (config_path.parent.parent / manifest_path).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _copy_if_exists(source: str | Path | None, destination: Path) -> str | None:
    if source is None:
        return None
    source_path = Path(source)
    if not source_path.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source_path, destination)
    except PermissionError:
        if destination.exists() and destination.stat().st_size > 0:
            return str(destination)
        raise
    return str(destination)


def _read_validation_flag(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("validation_passed"))


def _manifest_entry_valid(manifest: dict[str, Any], artifact_id: str, *, production: bool) -> bool:
    for section in ("required", "optional"):
        for item in manifest.get(section, []):
            if item.get("artifact_id") != artifact_id:
                continue
            if production and not bool(item.get("production_enabled")):
                return False
            return bool(item.get("validation_status")) and bool(item.get("path"))
    return False


def _stage_id_for_item(item: dict[str, Any]) -> str:
    runtime_name = str(item.get("runtime", "")).lower()
    candidate_name = str(item.get("candidate", "")).lower()
    return f"{runtime_name}:{candidate_name}"


def _string_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None
