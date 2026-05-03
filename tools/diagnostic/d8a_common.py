from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from tools.diagnostic.d7_common import PASS_A_SCENARIO_IDS, REFERENCE_DIR, REFERENCE_IDS, frame_envelope, load_scenarios

OUTPUT_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-05-02_d8a_gpu_enable"
CPU_PASS_A_RESULTS_PATH = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-04-27_d5_mixed_bank_integration"
    / "pass_a_results.json"
)
D7_LATENCY_BREAKDOWN_PATH = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-05-02_d7_reality_check"
    / "latency_breakdown.json"
)
RGB_SAMPLE = (
    "rgb_reference_session",
    "data/THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1.MP4",
    180,
)
THERMAL_SAMPLE = (
    "thermal_cross_sensor_proxy",
    "data/THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_2_Termal.MP4",
    450,
)


def safe_mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return mean(items) if items else 0.0


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    bounded = max(0.0, min(float(quantile), 1.0))
    rank = bounded * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float]) -> dict[str, float]:
    items = [float(value) for value in values]
    return {
        "mean": round(safe_mean(items), 6),
        "p50": round(percentile(items, 0.50), 6),
        "p95": round(percentile(items, 0.95), 6),
        "max": round(max(items) if items else 0.0, 6),
    }


def minutes_for_frames(ms_per_frame: float, *, frame_count: int = 2250) -> float:
    return round((float(ms_per_frame) * float(frame_count)) / 60000.0, 3)


def make_gpu_settings(reference_dir: str | Path = REFERENCE_DIR, **overrides: object) -> MvpRuntimeSettings:
    payload: dict[str, object] = {
        "task3_mode": "yoloe_vp_lightglue",
        "task3_reference_dir": Path(reference_dir),
        "task3_eval_reference_dir": Path(reference_dir),
        "task3_yoloe_allow_cpu": False,
        "task3_yoloe_device": "cuda:0",
    }
    payload.update(overrides)
    return MvpRuntimeSettings(**payload)


def load_cpu_pass_a_results() -> dict[str, Any]:
    return json.loads(CPU_PASS_A_RESULTS_PATH.read_text(encoding="utf-8"))


def load_d7_breakdown() -> dict[str, Any]:
    if not D7_LATENCY_BREAKDOWN_PATH.exists():
        return {}
    return json.loads(D7_LATENCY_BREAKDOWN_PATH.read_text(encoding="utf-8"))


def d7_component_baseline() -> dict[str, float]:
    payload = load_d7_breakdown()
    frame_records = list(payload.get("frame_records", []))
    return {
        "yoloe_mean_ms": round(safe_mean(float(item.get("yoloe_inference_ms", 0.0)) for item in frame_records), 6),
        "lightglue_mean_ms": round(safe_mean(float(item.get("lightglue_total_ms", 0.0)) for item in frame_records), 6),
    }


def query_nvidia_smi() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,cuda_version",
        "--format=csv,noheader,nounits",
    ]
    payload: dict[str, Any] = {"available": False, "rows": [], "full_text": ""}
    try:
        completed = subprocess.run(query, capture_output=True, text=True, check=True, timeout=20)
        payload["available"] = True
        payload["rows"] = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    except Exception:
        pass
    try:
        completed = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True, timeout=20)
        payload["available"] = True
        payload["full_text"] = completed.stdout.strip()
    except Exception:
        pass
    return payload


def torch_runtime_info(torch_module: Any) -> dict[str, Any]:
    info = {
        "python_executable": sys.executable,
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(getattr(torch_module.version, "cuda", None)),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "device_count": int(torch_module.cuda.device_count()) if bool(torch_module.cuda.is_available()) else 0,
    }
    if bool(info["cuda_available"]):
        props = torch_module.cuda.get_device_properties(0)
        info["device_name"] = str(props.name)
        info["capability"] = f"sm_{int(props.major)}{int(props.minor)}"
        info["total_memory_bytes"] = int(props.total_memory)
    return info


def accept_signature(per_ref_counts: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    normalized = []
    for reference_id in REFERENCE_IDS:
        normalized.append((reference_id, int(per_ref_counts.get(reference_id, 0))))
    return tuple(normalized)

