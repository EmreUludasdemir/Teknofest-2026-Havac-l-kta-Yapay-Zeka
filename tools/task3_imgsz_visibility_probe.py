"""Probe script for YOLOE visual-prompt visibility at higher imgsz values.

This script is diagnostic-only. It measures pre-gate candidate visibility for
selected references without modifying production settings or pipeline code.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.core.vision import is_cv2_available
from src.core.video_io import iter_video_frames
from src.evaluation.task3_manifest_eval import load_task3_manifest
from src.task3.experimental.backend import YoloeVpLightGlueBackend
from src.task3.reference_cache import ReferenceCache


OUTPUT_DIR = Path("_logs") / "imgsz_visibility_probe"
RGB_SCENARIO_ID = "rgb_reference_session"
THERMAL_SCENARIO_ID = "thermal_cross_sensor_proxy"
TARGET_REFS = ("ref_01", "ref_02", "ref_03")
SANITY_RGB_REFS = ("ref_05", "ref_06")
SANITY_THERMAL_REFS = ("ref_04",)
IMG_SIZES = (1280, 1920, 2560)
CONF_BUCKETS = (
    ("0.0-0.1", 0.0, 0.1),
    ("0.1-0.2", 0.1, 0.2),
    ("0.2-0.3", 0.2, 0.3),
    ("0.3+", 0.3, float("inf")),
)


def _bucketize_confidence(confidence: float) -> str:
    for label, lower, upper in CONF_BUCKETS:
        if lower <= confidence < upper:
            return label
    return "0.3+"


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(value) for value in values)
    index = max(int(round(0.95 * (len(sorted_values) - 1))), 0)
    return float(sorted_values[index])


def _build_subset_cache(full_cache: ReferenceCache, reference_ids: tuple[str, ...]) -> ReferenceCache:
    cache = ReferenceCache()
    for reference_id in reference_ids:
        item = full_cache.get(reference_id)
        if item is None:
            continue
        cache.put(reference_id, dict(item))
    return cache


def _build_backend(
    *,
    settings: MvpRuntimeSettings,
    full_cache: ReferenceCache,
    reference_ids: tuple[str, ...],
) -> YoloeVpLightGlueBackend:
    subset_cache = _build_subset_cache(full_cache, reference_ids)
    backend = YoloeVpLightGlueBackend(reference_cache=subset_cache, runtime_settings=settings)
    backend._ensure_ready(list(reference_ids))  # noqa: SLF001 - diagnostic reuse
    return backend


def _collect_pre_gate_counts(
    backend: YoloeVpLightGlueBackend,
    *,
    video_path: str,
    video_name: str,
    frame_limit: int,
    reference_ids: tuple[str, ...],
    imgsz: int,
) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]

    ref_names = backend.reference_bank.ref_names or list(reference_ids)
    allowed_ids = set(reference_ids)
    candidate_counts = Counter()
    confidence_buckets: dict[str, Counter[str]] = {reference_id: Counter() for reference_id in reference_ids}
    inference_ms: list[float] = []

    if str(backend.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        peak_vram_mb: float | None = 0.0
    else:
        peak_vram_mb = None

    for decoded_frame in iter_video_frames(
        video_path,
        frame_stride=1,
        limit=frame_limit,
        video_name=video_name,
    ):
        start = time.perf_counter()
        results = backend.model.predict(
            decoded_frame.bgr,
            conf=backend.runtime_settings.task3_yoloe_conf,
            iou=backend.runtime_settings.task3_yoloe_iou,
            imgsz=imgsz,
            max_det=max(backend.runtime_settings.task3_yoloe_max_det_per_class * max(len(reference_ids), 1), 1),
            verbose=False,
            device=backend.device,
        )
        inference_ms.append((time.perf_counter() - start) * 1000.0)
        if peak_vram_mb is not None:
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        if not results:
            continue
        prediction = results[0]
        boxes = getattr(prediction, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        raw_classes = boxes.cls.cpu().numpy().astype(int)
        raw_confidences = boxes.conf.cpu().numpy()
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, class_id in enumerate(raw_classes):
            if not 0 <= int(class_id) < len(ref_names):
                continue
            ref_name = ref_names[int(class_id)]
            if ref_name not in allowed_ids:
                continue
            grouped[int(class_id)].append(index)
        for class_id, indices in grouped.items():
            sorted_indices = sorted(indices, key=lambda idx: -float(raw_confidences[idx]))
            for index in sorted_indices[: backend.runtime_settings.task3_yoloe_max_det_per_class]:
                ref_name = ref_names[int(class_id)]
                confidence = float(raw_confidences[index])
                candidate_counts[ref_name] += 1
                confidence_buckets[ref_name][_bucketize_confidence(confidence)] += 1

    return {
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "confidence_buckets": {
            reference_id: {label: int(confidence_buckets[reference_id].get(label, 0)) for label, _, _ in CONF_BUCKETS}
            for reference_id in reference_ids
        },
        "median_inference_ms": round(statistics.median(inference_ms), 4) if inference_ms else 0.0,
        "p95_inference_ms": round(_p95(inference_ms), 4),
        "peak_vram_mb": None if peak_vram_mb is None else round(float(peak_vram_mb), 4),
        "frame_count": len(inference_ms),
    }


def _load_scenarios(settings: MvpRuntimeSettings) -> dict[str, dict[str, Any]]:
    manifest = load_task3_manifest(settings.task3_eval_manifest_path)
    return {scenario["id"]: scenario for scenario in manifest["scenarios"]}


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Task 3 YOLOE imgsz Visibility Probe")
    lines.append("")
    lines.append("## Visibility Sweep")
    lines.append("")
    lines.append("| imgsz | candidates ref_01 | ref_02 | ref_03 | median inference ms | p95 ms | peak VRAM |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["visibility_table"]:
        peak_vram = "-" if row["peak_vram_mb"] is None else row["peak_vram_mb"]
        lines.append(
            f"| {row['imgsz']} | {row['ref_01']} | {row['ref_02']} | {row['ref_03']} | {row['median_inference_ms']} | {row['p95_inference_ms']} | {peak_vram} |"
        )
    lines.append("")
    lines.append("## Confidence Buckets")
    lines.append("")
    for imgsz, refs_payload in payload["confidence_buckets"].items():
        lines.append(f"### imgsz={imgsz}")
        for reference_id, buckets in refs_payload.items():
            lines.append(f"- `{reference_id}`: {json.dumps(buckets, ensure_ascii=False)}")
        lines.append("")
    lines.append("## Sanity Check @ 1920")
    lines.append("")
    for key, result in payload["sanity_check"].items():
        lines.append(f"- `{key}`: candidates={result['candidate_counts']}, median_ms={result['median_inference_ms']}, p95_ms={result['p95_inference_ms']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not is_cv2_available():
        raise RuntimeError("OpenCV unavailable")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = MvpRuntimeSettings()
    scenarios = _load_scenarios(settings)
    rgb_scenario = scenarios[RGB_SCENARIO_ID]
    thermal_scenario = scenarios[THERMAL_SCENARIO_ID]

    full_cache = ReferenceCache()
    full_cache.preload_from_directory(settings.task3_eval_reference_dir, orb_features=settings.task3_orb_features)

    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("torch unavailable for probe") from exc

    base_settings = replace(
        settings,
        task3_reference_dir=settings.task3_eval_reference_dir,
        task3_yoloe_allow_cpu=settings.task3_yoloe_allow_cpu or not bool(torch.cuda.is_available()),
    )

    visibility_rows: list[dict[str, Any]] = []
    confidence_payload: dict[str, Any] = {}
    raw_results: dict[str, Any] = {}
    for imgsz in IMG_SIZES:
        probe_settings = replace(base_settings, task3_yoloe_imgsz=imgsz)
        backend = _build_backend(settings=probe_settings, full_cache=full_cache, reference_ids=TARGET_REFS)
        result = _collect_pre_gate_counts(
            backend,
            video_path=rgb_scenario["video"],
            video_name=rgb_scenario["id"],
            frame_limit=200,
            reference_ids=TARGET_REFS,
            imgsz=imgsz,
        )
        raw_results[str(imgsz)] = result
        confidence_payload[str(imgsz)] = result["confidence_buckets"]
        visibility_rows.append(
            {
                "imgsz": imgsz,
                "ref_01": int(result["candidate_counts"].get("ref_01", 0)),
                "ref_02": int(result["candidate_counts"].get("ref_02", 0)),
                "ref_03": int(result["candidate_counts"].get("ref_03", 0)),
                "median_inference_ms": result["median_inference_ms"],
                "p95_inference_ms": result["p95_inference_ms"],
                "peak_vram_mb": result["peak_vram_mb"],
            }
        )

    sanity_check: dict[str, Any] = {}
    baseline_row = next((row for row in visibility_rows if row["imgsz"] == 1280), None)
    if baseline_row and any(int(baseline_row[reference_id]) == 0 for reference_id in TARGET_REFS):
        probe_settings = replace(base_settings, task3_yoloe_imgsz=1920)
        rgb_backend = _build_backend(settings=probe_settings, full_cache=full_cache, reference_ids=SANITY_RGB_REFS)
        thermal_backend = _build_backend(settings=probe_settings, full_cache=full_cache, reference_ids=SANITY_THERMAL_REFS)
        sanity_check["rgb_reference_session_ref_05_ref_06"] = _collect_pre_gate_counts(
            rgb_backend,
            video_path=rgb_scenario["video"],
            video_name=rgb_scenario["id"],
            frame_limit=50,
            reference_ids=SANITY_RGB_REFS,
            imgsz=1920,
        )
        sanity_check["thermal_cross_sensor_proxy_ref_04"] = _collect_pre_gate_counts(
            thermal_backend,
            video_path=thermal_scenario["video"],
            video_name=thermal_scenario["id"],
            frame_limit=50,
            reference_ids=SANITY_THERMAL_REFS,
            imgsz=1920,
        )

    payload = {
        "device": "cuda" if bool(torch.cuda.is_available()) else "cpu",
        "vram_measured": bool(torch.cuda.is_available()),
        "visibility_table": visibility_rows,
        "confidence_buckets": confidence_payload,
        "raw_results": raw_results,
        "sanity_check": sanity_check,
    }

    (OUTPUT_DIR / "task3_imgsz_visibility_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "task3_imgsz_visibility_probe.md").write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
