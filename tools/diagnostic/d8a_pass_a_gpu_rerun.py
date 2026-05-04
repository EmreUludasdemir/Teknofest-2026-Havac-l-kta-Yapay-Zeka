from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.video_io import iter_video_frames
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches
from tools.diagnostic.d8a_common import (
    OUTPUT_DIR,
    PASS_A_SCENARIO_IDS,
    REFERENCE_IDS,
    accept_signature,
    frame_envelope,
    load_cpu_pass_a_results,
    load_scenarios,
    make_gpu_settings,
    minutes_for_frames,
    percentile,
    safe_mean,
    summarize,
)

try:
    import torch  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torch import failed in D8-A GPU rerun") from exc

PASS_A_EXPECTATIONS = {
    "rgb_reference_session": {"guard_accept_count": 8, "per_ref": {"ref_02": 1, "ref_03": 5, "ref_05": 1, "ref_06": 1}},
    "thermal_cross_sensor_proxy": {"guard_accept_count": 1, "per_ref": {"ref_04": 1}},
    "rgb_absent_target_proxy_2025": {"guard_accept_count": 1, "per_ref": {"ref_02": 1}},
    "thermal_absent_target_proxy_2025": {"guard_accept_count": 0, "per_ref": {}},
}


@dataclass(slots=True)
class FrameRecord:
    scenario_id: str
    frame_index: int
    modality: str
    wall_ms: float
    yoloe_ms: float
    lightglue_ms: float
    homography_ms: float
    accepted_ids: list[str]
    expected_accept_ids: list[str]
    unexpected_accept_ids: list[str]
    effective_mode: str
    fallback_reason: str | None
    backend_device: str | None
    peak_vram_allocated_bytes: int
    peak_vram_reserved_bytes: int


@dataclass(slots=True)
class ScenarioResult:
    scenario_id: str
    total_frames: int
    accepted_match_count: int
    false_positive_proxy_count: int
    guard_accept_count: int
    unexpected_accept_count: int
    per_ref_accept_counts: dict[str, int]
    frame_records: list[FrameRecord]


def _is_false_positive_proxy(match: Any, settings: Any) -> bool:
    score = float(match.metadata.get("match_score", 0.0))
    source = str(match.metadata.get("matcher_source", ""))
    if "template" in source:
        return score < 0.88
    if "yoloe" in source:
        yoloe_info = match.metadata.get("task3_yoloe", {})
        return not bool(yoloe_info.get("verify_passed", False))
    if "learned" in source:
        return float(match.metadata.get("similarity", 0.0)) < settings.task3_learned_min_similarity
    return float(match.metadata.get("inlier_ratio", 0.0)) < 0.45


def _evaluate_scenario(scenario: Any) -> ScenarioResult:
    settings = make_gpu_settings(scenario.references_dir)
    cache = ReferenceCache()
    cache.preload_from_directory(Path(scenario.references_dir), orb_features=settings.task3_orb_features)
    matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
    reference_ids = cache.list_ids()
    expected_set = set(scenario.expected_present_refs)

    accepted_match_count = 0
    false_positive_proxy_count = 0
    guard_accept_count = 0
    unexpected_accept_count = 0
    per_ref_accept_counts: Counter[str] = Counter()
    frame_records: list[FrameRecord] = []

    torch.cuda.empty_cache()
    frames = iter_video_frames(
        scenario.video,
        frame_stride=scenario.frame_stride,
        limit=int(scenario.frame_limit) if scenario.frame_limit is not None else None,
        video_name=scenario.scenario_id,
    )
    total_frames = 0
    for decoded in frames:
        total_frames += 1
        frame = frame_envelope(scenario.scenario_id, decoded.frame_index, decoded.width, decoded.height, tag="d8a-gpu")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        frame_started = time.perf_counter()
        raw_matches = matcher.match(frame, b"", reference_ids, decoded_frame=decoded, mode="yoloe_vp_lightglue")
        filtered = filter_no_match_candidates(
            raw_matches,
            min_score=settings.task3_min_score,
            mode="yoloe_vp_lightglue",
            yoloe_min_score=settings.task3_yoloe_min_score,
            modality=decoded.modality,
            yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
            ambiguity_margin=settings.task3_ambiguity_margin,
            suppression_mode=cache.get_candidate_suppression_mode(),
        )
        verified = verify_matches(
            frame,
            filtered,
            decoded_frame=decoded,
            min_inliers=settings.task3_match_min_inliers,
        )
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - frame_started) * 1000.0
        info = dict(matcher.last_run_info)
        backend = matcher.experimental_backend

        accepted_ids = [str(match.object_id) for match in verified]
        expected_accept_ids = [reference_id for reference_id in accepted_ids if reference_id in expected_set]
        unexpected_accept_ids = [reference_id for reference_id in accepted_ids if reference_id not in expected_set]

        accepted_match_count += len(verified)
        if scenario.reference_mode == "synthetic_absent":
            guard_accept_count += len(verified)
        else:
            guard_accept_count += len(expected_accept_ids)
            unexpected_accept_count += len(unexpected_accept_ids)

        for match in verified:
            per_ref_accept_counts[str(match.object_id)] += 1
            if _is_false_positive_proxy(match, settings):
                false_positive_proxy_count += 1

        frame_records.append(
            FrameRecord(
                scenario_id=scenario.scenario_id,
                frame_index=int(decoded.frame_index),
                modality=str(decoded.modality),
                wall_ms=round(float(wall_ms), 6),
                yoloe_ms=round(float(info.get("yoloe_inference_ms", 0.0)), 6),
                lightglue_ms=round(float(info.get("lightglue_verify_ms_total", 0.0)), 6),
                homography_ms=round(float(info.get("homography_compute_ms_total", 0.0)), 6),
                accepted_ids=accepted_ids,
                expected_accept_ids=expected_accept_ids,
                unexpected_accept_ids=unexpected_accept_ids,
                effective_mode=str(info.get("effective_mode", "unknown")),
                fallback_reason=info.get("fallback_reason"),
                backend_device=getattr(backend, "device", None),
                peak_vram_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_vram_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            )
        )

    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        total_frames=total_frames,
        accepted_match_count=accepted_match_count,
        false_positive_proxy_count=false_positive_proxy_count,
        guard_accept_count=guard_accept_count,
        unexpected_accept_count=unexpected_accept_count,
        per_ref_accept_counts={reference_id: int(per_ref_accept_counts.get(reference_id, 0)) for reference_id in REFERENCE_IDS},
        frame_records=frame_records,
    )


def _scenario_latency_stats(frame_records: list[dict[str, Any]] | list[FrameRecord]) -> dict[str, dict[str, float]]:
    def _value(record: Any, key: str) -> float:
        return float(record[key]) if isinstance(record, dict) else float(getattr(record, key))

    return {
        "wall": summarize(_value(record, "wall_ms") for record in frame_records),
        "yoloe": summarize(_value(record, "yoloe_ms") for record in frame_records),
        "lightglue": summarize(_value(record, "lightglue_ms") for record in frame_records),
        "homography": summarize(_value(record, "homography_ms") for record in frame_records),
    }


def _validate_against_frozen(result: ScenarioResult) -> list[str]:
    expectation = PASS_A_EXPECTATIONS[result.scenario_id]
    failures: list[str] = []
    if int(result.guard_accept_count) != int(expectation["guard_accept_count"]):
        failures.append(
            f"guard_accept_count expected {expectation['guard_accept_count']} got {result.guard_accept_count}"
        )
    for reference_id, expected_count in expectation["per_ref"].items():
        actual_count = int(result.per_ref_accept_counts.get(reference_id, 0))
        if actual_count != int(expected_count):
            failures.append(f"{reference_id} expected {expected_count} got {actual_count}")
    return failures


def _write_markdown(
    *,
    cpu_results: dict[str, Any],
    gpu_results: dict[str, ScenarioResult],
) -> tuple[dict[str, Any], list[str]]:
    scenario_rows: list[dict[str, Any]] = []
    all_cpu_frames: list[dict[str, Any]] = []
    all_gpu_frames: list[FrameRecord] = []
    frozen_failures: list[str] = []
    determinism_diffs: list[str] = []

    lines = [
        "# D8-A Pass A GPU Rerun",
        "",
        "## Scoreboard",
        "",
        "| Scenario | CPU Guard | GPU Guard | CPU Accepts | GPU Accepts | CPU FP | GPU FP | CPU Unexpected | GPU Unexpected | Deterministic | Frozen |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario_id in PASS_A_SCENARIO_IDS:
        cpu = cpu_results[scenario_id]
        gpu = gpu_results[scenario_id]
        same_signature = accept_signature(cpu.get("per_ref_accept_counts", {})) == accept_signature(gpu.per_ref_accept_counts)
        same_accept_count = int(cpu.get("accepted_match_count", 0)) == int(gpu.accepted_match_count)
        deterministic = same_signature and same_accept_count
        if not deterministic:
            delta = []
            for reference_id in REFERENCE_IDS:
                cpu_count = int(cpu.get("per_ref_accept_counts", {}).get(reference_id, 0))
                gpu_count = int(gpu.per_ref_accept_counts.get(reference_id, 0))
                if cpu_count != gpu_count:
                    delta.append(f"{reference_id}:{cpu_count}->{gpu_count}")
            determinism_diffs.append(f"{scenario_id}: " + ", ".join(delta))
        frozen_errors = _validate_against_frozen(gpu)
        frozen_status = "pass" if not frozen_errors else "fail"
        for error in frozen_errors:
            frozen_failures.append(f"{scenario_id}: {error}")
        lines.append(
            f"| {scenario_id} | {int(cpu.get('guard_accept_count', 0))} | {gpu.guard_accept_count} | "
            f"{int(cpu.get('accepted_match_count', 0))} | {gpu.accepted_match_count} | "
            f"{int(cpu.get('false_positive_proxy_count', 0))} | {gpu.false_positive_proxy_count} | "
            f"{int(cpu.get('unexpected_accept_count', 0))} | {gpu.unexpected_accept_count} | "
            f"{'same' if deterministic else 'diff'} | {frozen_status} |"
        )
        cpu_frames = list(cpu.get("frame_records", []))
        all_cpu_frames.extend(cpu_frames)
        all_gpu_frames.extend(gpu.frame_records)
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "cpu_latency": _scenario_latency_stats(cpu_frames),
                "gpu_latency": _scenario_latency_stats(gpu.frame_records),
                "cpu_accepts": int(cpu.get("accepted_match_count", 0)),
                "gpu_accepts": int(gpu.accepted_match_count),
                "cpu_guard": int(cpu.get("guard_accept_count", 0)),
                "gpu_guard": int(gpu.guard_accept_count),
                "cpu_fp": int(cpu.get("false_positive_proxy_count", 0)),
                "gpu_fp": int(gpu.false_positive_proxy_count),
                "gpu_peak_reserved_mib": round(
                    max(float(record.peak_vram_reserved_bytes) for record in gpu.frame_records) / (1024.0 * 1024.0),
                    3,
                )
                if gpu.frame_records
                else 0.0,
            }
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Scenario | Frames | CPU Wall P50 | GPU Wall P50 | CPU Wall P95 | GPU Wall P95 | CPU YOLOE P50 | GPU YOLOE P50 | CPU LightGlue P50 | GPU LightGlue P50 | GPU Peak Reserved MiB |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in scenario_rows:
        lines.append(
            f"| {row['scenario_id']} | {len(gpu_results[row['scenario_id']].frame_records)} | "
            f"{row['cpu_latency']['wall']['p50']:.1f} | {row['gpu_latency']['wall']['p50']:.1f} | "
            f"{row['cpu_latency']['wall']['p95']:.1f} | {row['gpu_latency']['wall']['p95']:.1f} | "
            f"{row['cpu_latency']['yoloe']['p50']:.1f} | {row['gpu_latency']['yoloe']['p50']:.1f} | "
            f"{row['cpu_latency']['lightglue']['p50']:.1f} | {row['gpu_latency']['lightglue']['p50']:.1f} | "
            f"{row['gpu_peak_reserved_mib']:.1f} |"
        )

    cpu_overall = _scenario_latency_stats(all_cpu_frames)
    gpu_overall = _scenario_latency_stats(all_gpu_frames)
    gpu_minutes = minutes_for_frames(gpu_overall["wall"]["mean"])
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- CPU wall p50/p95/max: `{cpu_overall['wall']['p50']:.1f} / {cpu_overall['wall']['p95']:.1f} / {cpu_overall['wall']['max']:.1f} ms`.",
            f"- GPU wall p50/p95/max: `{gpu_overall['wall']['p50']:.1f} / {gpu_overall['wall']['p95']:.1f} / {gpu_overall['wall']['max']:.1f} ms`.",
            f"- CPU YOLOE p50/p95: `{cpu_overall['yoloe']['p50']:.1f} / {cpu_overall['yoloe']['p95']:.1f} ms`.",
            f"- GPU YOLOE p50/p95: `{gpu_overall['yoloe']['p50']:.1f} / {gpu_overall['yoloe']['p95']:.1f} ms`.",
            f"- CPU LightGlue verify-total p50/p95: `{cpu_overall['lightglue']['p50']:.1f} / {cpu_overall['lightglue']['p95']:.1f} ms`.",
            f"- GPU LightGlue verify-total p50/p95: `{gpu_overall['lightglue']['p50']:.1f} / {gpu_overall['lightglue']['p95']:.1f} ms`.",
            f"- GPU 2250-frame extrapolation from mean wall: `{gpu_minutes:.1f} dakika`.",
            "",
            "## Determinism",
            "",
        ]
    )
    if determinism_diffs:
        for item in determinism_diffs:
            lines.append(f"- DIFF: {item}")
    else:
        lines.append("- CPU and GPU produced the same per-reference accept signature on all four frozen scenarios.")
    lines.extend(["", "## Frozen Guard", ""])
    if frozen_failures:
        for item in frozen_failures:
            lines.append(f"- FAIL: {item}")
    else:
        lines.append("- PASS: GPU rerun matched the frozen D2-promote guard under D5 counting semantics.")
    lines.append("")
    (OUTPUT_DIR / "pass_a_gpu_rerun.md").write_text("\n".join(lines), encoding="utf-8")

    summary_payload = {
        "cpu_overall": cpu_overall,
        "gpu_overall": gpu_overall,
        "gpu_minutes_2250_frames": gpu_minutes,
        "frozen_failures": frozen_failures,
        "determinism_diffs": determinism_diffs,
        "scenario_rows": scenario_rows,
    }
    return summary_payload, frozen_failures


def _write_summary(payload: dict[str, Any], frozen_failures: list[str], gpu_results: dict[str, ScenarioResult]) -> None:
    gpu_p50 = float(payload["gpu_overall"]["wall"]["p50"])
    gpu_p95 = float(payload["gpu_overall"]["wall"]["p95"])
    if gpu_p50 <= 1000.0:
        decision = "mevcut mimari korunur"
        lane = "D8-A stay"
    elif gpu_p50 <= 3000.0:
        decision = "LightGlue config tweak"
        lane = "D8-B"
    elif gpu_p50 <= 8000.0:
        decision = "simplified pipeline"
        lane = "D8-C"
    else:
        decision = "mimari degisiklik / strategic review"
        lane = "D8-D"

    fallback_frames = []
    for scenario_id, result in gpu_results.items():
        for record in result.frame_records:
            if record.fallback_reason:
                fallback_frames.append(f"{scenario_id}:{record.frame_index}:{record.fallback_reason}")

    lines = [
        "# D8-A Summary",
        "",
        "| Senaryo | GPU Latency p50 | Karar |",
        "| --- | --- | --- |",
        f"| overall_pass_a | {gpu_p50:.1f} ms/frame | {decision} ({lane}) |",
        "",
        "## Answers",
        "",
        f"1. Current GPU-enabled Pass A mean extrapolation is `{float(payload['gpu_minutes_2250_frames']):.1f} dakika`; "
        f"overall wall p50/p95 is `{gpu_p50:.1f} / {gpu_p95:.1f} ms`, so the decision lane is `{lane}`.",
        "2. Measurement runtime is `.venv` with `torch 2.9.1+cu128`; the earlier `2.5.1+cu121` route was discarded because it could not execute kernels on `sm_120`.",
        f"3. CPU vs GPU frozen scoreboard {'matched' if not payload['determinism_diffs'] else 'diverged'}; determinism drift count is `{len(payload['determinism_diffs'])}`.",
        f"4. Frozen D2-promote guard {'still fails identically to CPU' if frozen_failures and not payload['determinism_diffs'] else ('passed' if not frozen_failures else 'failed with GPU-specific drift')}; "
        f"failure count is `{len(frozen_failures)}`. Fallback frames observed: `{fallback_frames if fallback_frames else []}`.",
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA unavailable for D8-A Pass A GPU rerun")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(scenario_ids=PASS_A_SCENARIO_IDS)
    cpu_results = load_cpu_pass_a_results()

    gpu_results: dict[str, ScenarioResult] = {}
    for scenario_id in PASS_A_SCENARIO_IDS:
        gpu_results[scenario_id] = _evaluate_scenario(scenarios[scenario_id])

    (OUTPUT_DIR / "pass_a_gpu_rerun.json").write_text(
        json.dumps({scenario_id: asdict(result) for scenario_id, result in gpu_results.items()}, indent=2),
        encoding="utf-8",
    )
    summary_payload, frozen_failures = _write_markdown(cpu_results=cpu_results, gpu_results=gpu_results)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_summary(summary_payload, frozen_failures, gpu_results)
    print(f"Wrote Pass A GPU rerun outputs to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
