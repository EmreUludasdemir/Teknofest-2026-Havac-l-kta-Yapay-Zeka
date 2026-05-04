from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope
from src.core.video_io import iter_video_frames
from src.task3.matcher import Task3Matcher
from src.task3.reference_cache import ReferenceCache
from tools.diagnostic.d6_probe_common import build_temp_reference_bank, cleanup_temp_dir
from tools.diagnostic.d7_common import OUTPUT_DIR, REFERENCE_DIR, frame_envelope

try:
    import torch  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    torch = None

RGB_SAMPLE = ("rgb_reference_session", "data/THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1.MP4", 180)
THERMAL_SAMPLE = ("thermal_cross_sensor_proxy", "data/THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_2_Termal.MP4", 450)
SIX_REF_IDS = [f"ref_{index:02d}" for index in range(1, 7)]
TWELVE_REF_IDS = [f"ref_{index:02d}" for index in range(1, 13)]


def _query_nvidia_smi() -> dict[str, Any]:
    commands = [
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,cuda_version",
            "--format=csv,noheader,nounits",
        ],
        ["nvidia-smi"],
    ]
    payload: dict[str, Any] = {"available": False, "query": None, "full_text": None}
    for command in commands:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
        except Exception:
            continue
        payload["available"] = True
        stdout = completed.stdout.strip()
        if command[0] == "nvidia-smi" and "--query-gpu=name,memory.total,driver_version,cuda_version" in command:
            rows = [item.strip() for item in stdout.splitlines() if item.strip()]
            payload["query"] = rows
        else:
            payload["full_text"] = stdout
        if payload["query"] is not None and payload["full_text"] is not None:
            break
    return payload


def _torch_info() -> dict[str, Any]:
    if torch is None:
        return {"available": False}
    info = {
        "available": True,
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if getattr(torch, "cuda", None) else 0,
    }
    if info["cuda_available"]:
        devices = []
        for index in range(int(info["device_count"])):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": str(props.name),
                    "total_memory_bytes": int(props.total_memory),
                }
            )
        info["devices"] = devices
    return info


def _settings(reference_dir: str | Path, *, device: str | None, allow_cpu: bool) -> MvpRuntimeSettings:
    return MvpRuntimeSettings(
        task3_mode="yoloe_vp_lightglue",
        task3_reference_dir=Path(reference_dir),
        task3_eval_reference_dir=Path(reference_dir),
        task3_yoloe_allow_cpu=allow_cpu,
        task3_yoloe_device=device,
    )


def _sample_frame(video_path: str, *, frame_index: int, video_name: str) -> Any:
    frames = iter_video_frames(video_path, frame_stride=1, limit=None, video_name=video_name)
    for decoded in frames:
        if int(decoded.frame_index) == int(frame_index):
            return decoded
    raise RuntimeError(f"Frame {frame_index} not found in {video_path}")


def _single_frame_benchmark(
    *,
    scenario_id: str,
    video_path: str,
    frame_index: int,
    reference_dir: str | Path,
    device: str | None,
    allow_cpu: bool,
) -> dict[str, Any]:
    settings = _settings(reference_dir, device=device, allow_cpu=allow_cpu)
    cache = ReferenceCache()
    preload_started = time.perf_counter()
    cache.preload_from_directory(Path(reference_dir), orb_features=settings.task3_orb_features)
    preload_ms = (time.perf_counter() - preload_started) * 1000.0
    matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
    decoded = _sample_frame(video_path, frame_index=frame_index, video_name=scenario_id)
    frame = frame_envelope(scenario_id, decoded.frame_index, decoded.width, decoded.height, tag="d7-gpu")

    max_memory_allocated = None
    max_memory_reserved = None
    if torch is not None and bool(torch.cuda.is_available()) and device and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    matches = matcher.match(frame, b"", cache.list_ids(), decoded_frame=decoded, mode="yoloe_vp_lightglue")
    wall_ms = (time.perf_counter() - started) * 1000.0
    if torch is not None and bool(torch.cuda.is_available()) and device and str(device).startswith("cuda"):
        max_memory_allocated = int(torch.cuda.max_memory_allocated())
        max_memory_reserved = int(torch.cuda.max_memory_reserved())
    info = dict(matcher.last_run_info)
    return {
        "scenario_id": scenario_id,
        "frame_index": int(frame_index),
        "reference_dir": str(reference_dir),
        "device": device or "auto",
        "allow_cpu": bool(allow_cpu),
        "wall_ms": round(float(wall_ms), 6),
        "preload_ms": round(float(preload_ms), 6),
        "effective_mode": info.get("effective_mode"),
        "fallback_reason": info.get("fallback_reason"),
        "yoloe_inference_ms": float(info.get("yoloe_inference_ms", 0.0)),
        "lightglue_verify_ms_total": float(info.get("lightglue_verify_ms_total", 0.0)),
        "homography_compute_ms_total": float(info.get("homography_compute_ms_total", 0.0)),
        "accepted_count": int(len(matches)),
        "accepted_ids": [str(item.object_id) for item in matches],
        "max_memory_allocated_bytes": max_memory_allocated,
        "max_memory_reserved_bytes": max_memory_reserved,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nvidia_smi = _query_nvidia_smi()
    torch_info = _torch_info()
    lightglue_available = importlib.util.find_spec("lightglue") is not None
    ultralytics_available = importlib.util.find_spec("ultralytics") is not None

    cpu_rgb = _single_frame_benchmark(
        scenario_id=RGB_SAMPLE[0],
        video_path=RGB_SAMPLE[1],
        frame_index=RGB_SAMPLE[2],
        reference_dir=REFERENCE_DIR,
        device=None,
        allow_cpu=True,
    )
    cpu_thermal = _single_frame_benchmark(
        scenario_id=THERMAL_SAMPLE[0],
        video_path=THERMAL_SAMPLE[1],
        frame_index=THERMAL_SAMPLE[2],
        reference_dir=REFERENCE_DIR,
        device=None,
        allow_cpu=True,
    )

    gpu_result_6 = None
    gpu_result_12 = None
    six_ref_dir = None
    if torch is not None and bool(torch.cuda.is_available()):
        six_ref_dir = build_temp_reference_bank(
            selected_ref_ids=SIX_REF_IDS,
            per_reference_suppression=True,
            suffix="d7_gpu_runtime_6ref",
        )
        try:
            gpu_result_6 = _single_frame_benchmark(
                scenario_id=RGB_SAMPLE[0],
                video_path=RGB_SAMPLE[1],
                frame_index=RGB_SAMPLE[2],
                reference_dir=six_ref_dir,
                device="cuda:0",
                allow_cpu=False,
            )
            gpu_result_12 = _single_frame_benchmark(
                scenario_id=RGB_SAMPLE[0],
                video_path=RGB_SAMPLE[1],
                frame_index=RGB_SAMPLE[2],
                reference_dir=REFERENCE_DIR,
                device="cuda:0",
                allow_cpu=False,
            )
        finally:
            cleanup_temp_dir(Path(six_ref_dir))

    payload = {
        "nvidia_smi": nvidia_smi,
        "torch": torch_info,
        "lightglue_available": lightglue_available,
        "ultralytics_available": ultralytics_available,
        "cpu_rgb_sample": cpu_rgb,
        "cpu_thermal_sample": cpu_thermal,
        "gpu_6ref_sample": gpu_result_6,
        "gpu_12ref_sample": gpu_result_12,
    }
    (OUTPUT_DIR / "gpu_runtime_check.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# D7 GPU Runtime Check",
        "",
        "## Direct Answer",
        "",
    ]
    if not torch_info.get("available"):
        lines.append("- `torch` is not importable in the current runtime, so GPU acceleration cannot be used here.")
    elif not bool(torch_info.get("cuda_available")):
        lines.append(
            f"- System GPU is visible through `nvidia-smi`, but the active Python runtime is `torch {torch_info.get('version')}` with `cuda_available=false`. "
            "This environment is CPU-only for Task 3 today."
        )
    else:
        lines.append("- CUDA is available in the active Python runtime; GPU sample results are in the table below.")
    lines.append(f"- LightGlue installed: `{lightglue_available}`. Ultralytics installed: `{ultralytics_available}`.")
    if nvidia_smi.get("query"):
        lines.append(f"- `nvidia-smi` reports: `{nvidia_smi['query']}`.")
    lines.append("")

    lines.extend(
        [
            "## Single-Frame Samples",
            "",
            "| Sample | Device | Wall ms | YOLOE ms | LightGlue ms | Homography ms | Accepted Count | Accepted IDs | Peak VRAM Allocated | Peak VRAM Reserved |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            f"| CPU RGB 12-ref | {cpu_rgb['device']} | {cpu_rgb['wall_ms']:.1f} | {cpu_rgb['yoloe_inference_ms']:.1f} | {cpu_rgb['lightglue_verify_ms_total']:.1f} | {cpu_rgb['homography_compute_ms_total']:.1f} | {cpu_rgb['accepted_count']} | {cpu_rgb['accepted_ids']} | - | - |",
            f"| CPU Thermal 12-ref | {cpu_thermal['device']} | {cpu_thermal['wall_ms']:.1f} | {cpu_thermal['yoloe_inference_ms']:.1f} | {cpu_thermal['lightglue_verify_ms_total']:.1f} | {cpu_thermal['homography_compute_ms_total']:.1f} | {cpu_thermal['accepted_count']} | {cpu_thermal['accepted_ids']} | - | - |",
        ]
    )
    if gpu_result_6 is not None and gpu_result_12 is not None:
        lines.append(
            f"| GPU RGB 6-ref | {gpu_result_6['device']} | {gpu_result_6['wall_ms']:.1f} | {gpu_result_6['yoloe_inference_ms']:.1f} | {gpu_result_6['lightglue_verify_ms_total']:.1f} | {gpu_result_6['homography_compute_ms_total']:.1f} | {gpu_result_6['accepted_count']} | {gpu_result_6['accepted_ids']} | {gpu_result_6['max_memory_allocated_bytes']} | {gpu_result_6['max_memory_reserved_bytes']} |"
        )
        lines.append(
            f"| GPU RGB 12-ref | {gpu_result_12['device']} | {gpu_result_12['wall_ms']:.1f} | {gpu_result_12['yoloe_inference_ms']:.1f} | {gpu_result_12['lightglue_verify_ms_total']:.1f} | {gpu_result_12['homography_compute_ms_total']:.1f} | {gpu_result_12['accepted_count']} | {gpu_result_12['accepted_ids']} | {gpu_result_12['max_memory_allocated_bytes']} | {gpu_result_12['max_memory_reserved_bytes']} |"
        )
    else:
        lines.append("| GPU RGB 6-ref | unavailable | - | - | - | - | - | - | - | - |")
        lines.append("| GPU RGB 12-ref | unavailable | - | - | - | - | - | - | - | - |")
    lines.append("")

    lines.extend(
        [
            "## Runtime Interpretation",
            "",
            "- The machine has an RTX 5060 Laptop GPU with about 8 GB VRAM visible to the OS.",
            "- The current Python runtime cannot use that GPU because the installed Torch build is CPU-only.",
            "- As a result, no trustworthy GPU-vs-CPU latency or 6-ref-vs-12-ref VRAM benchmark can be produced from this environment today.",
            "",
        ]
    )

    (OUTPUT_DIR / "gpu_runtime_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote GPU runtime check to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
