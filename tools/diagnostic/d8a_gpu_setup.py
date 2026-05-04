from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.video_io import iter_video_frames
from src.task3.matcher import Task3Matcher
from src.task3.reference_cache import ReferenceCache
from tools.diagnostic.d8a_common import (
    OUTPUT_DIR,
    REFERENCE_DIR,
    RGB_SAMPLE,
    THERMAL_SAMPLE,
    d7_component_baseline,
    frame_envelope,
    make_gpu_settings,
    query_nvidia_smi,
    torch_runtime_info,
)

try:
    import torch  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torch import failed in D8-A GPU setup") from exc


def _sample_frame(video_path: str, *, frame_index: int, video_name: str) -> Any:
    frames = iter_video_frames(video_path, frame_stride=1, limit=None, video_name=video_name)
    for decoded in frames:
        if int(decoded.frame_index) == int(frame_index):
            return decoded
    raise RuntimeError(f"Frame {frame_index} not found in {video_path}")


def _single_frame_gpu_smoke(
    *,
    scenario_id: str,
    video_path: str,
    frame_index: int,
) -> dict[str, Any]:
    settings = make_gpu_settings(REFERENCE_DIR)
    cache = ReferenceCache()
    preload_started = time.perf_counter()
    cache.preload_from_directory(REFERENCE_DIR, orb_features=settings.task3_orb_features)
    preload_ms = (time.perf_counter() - preload_started) * 1000.0
    matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
    decoded = _sample_frame(video_path, frame_index=frame_index, video_name=scenario_id)
    frame = frame_envelope(scenario_id, decoded.frame_index, decoded.width, decoded.height, tag="d8a-smoke")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    matches = matcher.match(frame, b"", cache.list_ids(), decoded_frame=decoded, mode="yoloe_vp_lightglue")
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    info = dict(matcher.last_run_info)
    backend = matcher.experimental_backend
    return {
        "scenario_id": scenario_id,
        "frame_index": int(frame_index),
        "modality": str(decoded.modality),
        "preload_ms": round(float(preload_ms), 6),
        "wall_ms": round(float(wall_ms), 6),
        "effective_mode": str(info.get("effective_mode")),
        "fallback_reason": info.get("fallback_reason"),
        "backend_device": getattr(backend, "device", None),
        "yoloe_inference_ms": round(float(info.get("yoloe_inference_ms", 0.0)), 6),
        "lightglue_verify_ms_total": round(float(info.get("lightglue_verify_ms_total", 0.0)), 6),
        "homography_compute_ms_total": round(float(info.get("homography_compute_ms_total", 0.0)), 6),
        "accepted_count": int(len(matches)),
        "accepted_ids": [str(item.object_id) for item in matches],
        "yoloe_routed_refs": list(info.get("yoloe_routed_refs", [])),
        "orb_routed_refs": list(info.get("orb_routed_refs", [])),
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA still unavailable in active runtime")

    nvidia_smi = query_nvidia_smi()
    runtime = torch_runtime_info(torch)
    component_baseline = d7_component_baseline()

    tensor_started = time.perf_counter()
    first = torch.randn((1024, 1024), device="cuda")
    second = torch.randn((1024, 1024), device="cuda")
    total = (first @ second).sum()
    torch.cuda.synchronize()
    tensor_ms = (time.perf_counter() - tensor_started) * 1000.0

    thermal_smoke = _single_frame_gpu_smoke(
        scenario_id=THERMAL_SAMPLE[0],
        video_path=THERMAL_SAMPLE[1],
        frame_index=THERMAL_SAMPLE[2],
    )
    rgb_smoke = _single_frame_gpu_smoke(
        scenario_id=RGB_SAMPLE[0],
        video_path=RGB_SAMPLE[1],
        frame_index=RGB_SAMPLE[2],
    )

    payload = {
        "runtime": runtime,
        "nvidia_smi": nvidia_smi,
        "d7_component_baseline": component_baseline,
        "tensor_smoke": {
            "wall_ms": round(float(tensor_ms), 6),
            "value": round(float(total.item()), 6),
        },
        "thermal_smoke": thermal_smoke,
        "rgb_smoke": rgb_smoke,
    }
    (OUTPUT_DIR / "gpu_smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total_vram_gb = float(runtime.get("total_memory_bytes", 0)) / 1e9 if runtime.get("total_memory_bytes") else 0.0
    lines = [
        "# D8-A GPU Smoke",
        "",
        "## Runtime",
        "",
        f"- Python: `{runtime['python_executable']}`",
        f"- Torch: `{runtime['torch_version']}` with CUDA runtime `{runtime['torch_cuda_version']}`",
        f"- CUDA available: `{runtime['cuda_available']}` devices=`{runtime['device_count']}`",
        f"- GPU: `{runtime.get('device_name', 'unknown')}` capability `{runtime.get('capability', '-')}` VRAM `{total_vram_gb:.2f} GB`",
        f"- `nvidia-smi` query: `{nvidia_smi.get('rows', [])}`",
        "",
        "## Smoke Table",
        "",
        "| Sample | Modality | Backend Device | Wall ms | YOLOE ms | LightGlue Verify ms | Homography ms | Accepts | Peak Alloc VRAM MiB | Peak Reserved VRAM MiB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| thermal frame {thermal_smoke['frame_index']} | {thermal_smoke['modality']} | {thermal_smoke.get('backend_device') or '-'} | {thermal_smoke['wall_ms']:.1f} | {thermal_smoke['yoloe_inference_ms']:.1f} | {thermal_smoke['lightglue_verify_ms_total']:.1f} | {thermal_smoke['homography_compute_ms_total']:.1f} | {thermal_smoke['accepted_ids']} | {thermal_smoke['peak_vram_allocated_bytes'] / (1024.0 * 1024.0):.1f} | {thermal_smoke['peak_vram_reserved_bytes'] / (1024.0 * 1024.0):.1f} |",
        f"| rgb frame {rgb_smoke['frame_index']} | {rgb_smoke['modality']} | {rgb_smoke.get('backend_device') or '-'} | {rgb_smoke['wall_ms']:.1f} | {rgb_smoke['yoloe_inference_ms']:.1f} | {rgb_smoke['lightglue_verify_ms_total']:.1f} | {rgb_smoke['homography_compute_ms_total']:.1f} | {rgb_smoke['accepted_ids']} | {rgb_smoke['peak_vram_allocated_bytes'] / (1024.0 * 1024.0):.1f} | {rgb_smoke['peak_vram_reserved_bytes'] / (1024.0 * 1024.0):.1f} |",
        "",
        "## Comparison Notes",
        "",
        f"- D7 CPU intrusive component baseline for comparison: YOLOE mean `{float(component_baseline['yoloe_mean_ms']):.1f} ms`, LightGlue mean `{float(component_baseline['lightglue_mean_ms']):.1f} ms`.",
        f"- CUDA tensor smoke completed in `{tensor_ms:.1f} ms`; value checksum `{float(total.item()):.4f}` confirms real kernel execution.",
        f"- Thermal sample fallback reason: `{thermal_smoke['fallback_reason']}`. RGB sample fallback reason: `{rgb_smoke['fallback_reason']}`.",
        "",
    ]
    (OUTPUT_DIR / "gpu_smoke.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote GPU smoke outputs to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
