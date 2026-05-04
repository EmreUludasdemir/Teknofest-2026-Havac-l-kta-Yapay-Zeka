from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope
from src.core.video_io import iter_video_frames
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches
from tools.diagnostic.d6_probe_common import MANIFEST_PATH, build_temp_reference_bank, cleanup_temp_dir

OUTPUT_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-04-27_d5_mixed_bank_integration"
CURRENT_PASS_A_PATH = OUTPUT_DIR / "pass_a_results.json"
PASS_A_SCENARIO_IDS = (
    "rgb_reference_session",
    "thermal_cross_sensor_proxy",
    "rgb_absent_target_proxy_2025",
    "thermal_absent_target_proxy_2025",
)
REFERENCE_IDS = tuple(f"ref_{index:02d}" for index in range(1, 13))


@dataclass(slots=True)
class ScenarioConfig:
    scenario_id: str
    name: str
    video: str
    references_dir: str
    reference_mode: str
    frame_stride: int
    frame_limit: int | None
    expected_present_refs: list[str]
    modality_routing_breach_refs: list[str]
    ambiguous_routing_refs: list[str]
    tags: list[str]


def _load_manifest() -> list[ScenarioConfig]:
    base_settings = MvpRuntimeSettings()
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    scenarios: list[ScenarioConfig] = []
    for index, raw in enumerate(payload.get("scenarios", []), start=1):
        scenario_id = str(raw.get("id") or raw.get("name") or raw.get("scenario_id") or f"scenario_{index:02d}")
        scenarios.append(
            ScenarioConfig(
                scenario_id=scenario_id,
                name=str(raw.get("name") or scenario_id),
                video=str(raw["video"]),
                references_dir=str(raw.get("references_dir") or raw.get("reference_dir") or ""),
                reference_mode=str(raw.get("reference_mode") or "present_targets"),
                frame_stride=int(raw.get("frame_stride", base_settings.task3_eval_frame_stride)),
                frame_limit=raw.get("frame_limit", base_settings.task3_eval_frame_limit),
                expected_present_refs=list(raw.get("expected_present_refs", [])),
                modality_routing_breach_refs=list(raw.get("modality_routing_breach_refs", [])),
                ambiguous_routing_refs=list(raw.get("ambiguous_routing_refs", [])),
                tags=list(raw.get("tags", [])),
            )
        )
    return scenarios


def _make_settings(reference_dir: str | Path) -> MvpRuntimeSettings:
    return MvpRuntimeSettings(
        task3_mode="yoloe_vp_lightglue",
        task3_reference_dir=Path(reference_dir),
        task3_eval_reference_dir=Path(reference_dir),
        task3_yoloe_allow_cpu=True,
    )


def _frame_envelope(scenario_id: str, frame_index: int, width: int, height: int) -> FrameEnvelope:
    return FrameEnvelope(
        frame_url=f"http://task3-d6-latency/frames/{frame_index + 1}/",
        image_url=f"/task3-d6-latency/{frame_index + 1}.jpg",
        video_name=scenario_id,
        translation_x=0.0,
        translation_y=0.0,
        translation_z=0.0,
        health_status="1",
        metadata={"frame_index": frame_index, "image_width": width, "image_height": height},
    )


def _evaluate_scenario(scenario: ScenarioConfig) -> dict[str, Any]:
    settings = _make_settings(scenario.references_dir)
    cache = ReferenceCache()
    cache.preload_from_directory(Path(scenario.references_dir), orb_features=settings.task3_orb_features)
    matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
    reference_ids = cache.list_ids()

    frame_records: list[dict[str, Any]] = []
    frames = iter_video_frames(
        scenario.video,
        frame_stride=scenario.frame_stride,
        limit=int(scenario.frame_limit) if scenario.frame_limit is not None else None,
        video_name=scenario.scenario_id,
    )
    for decoded in frames:
        frame = _frame_envelope(scenario.scenario_id, decoded.frame_index, decoded.width, decoded.height)
        active_ids = cache.filter_reference_ids_by_detector_modality(reference_ids, modality=decoded.modality)
        both_active_ids = [reference_id for reference_id in active_ids if cache.get_detector(reference_id) == "both"]
        frame_start = time.perf_counter()
        raw_matches = matcher.match(frame, b"", reference_ids, decoded_frame=decoded, mode="yoloe_vp_lightglue")
        task3_info = dict(matcher.last_run_info)
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
        wall_ms = (time.perf_counter() - frame_start) * 1000.0
        frame_records.append(
            {
                "scenario_id": scenario.scenario_id,
                "frame_index": int(decoded.frame_index),
                "modality": str(decoded.modality),
                "wall_ms": round(wall_ms, 6),
                "yoloe_ms": round(float(task3_info.get("yoloe_inference_ms", 0.0)), 6),
                "lightglue_ms": round(float(task3_info.get("lightglue_verify_ms_total", 0.0)), 6),
                "homography_ms": round(float(task3_info.get("homography_compute_ms_total", 0.0)), 6),
                "active_ref_count": len(active_ids),
                "both_active_ref_count": len(both_active_ids),
                "raw_candidate_count": len(raw_matches),
                "filtered_candidate_count": len(filtered),
                "verified_count": len(verified),
                "yoloe_routed_refs": list(task3_info.get("yoloe_routed_refs", [])),
                "orb_routed_refs": list(task3_info.get("orb_routed_refs", [])),
            }
        )
    return {
        "scenario_id": scenario.scenario_id,
        "reference_mode": scenario.reference_mode,
        "frame_records": frame_records,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(percentile, 1.0)) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _stats(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(record[key]) for record in records]
    return {
        "mean": float(statistics.mean(values)) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": float(max(values)) if values else 0.0,
    }


def _load_current_results() -> dict[str, dict[str, Any]]:
    return json.loads(CURRENT_PASS_A_PATH.read_text(encoding="utf-8"))


def _build_counterfactual_results() -> dict[str, dict[str, Any]]:
    scenarios = {scenario.scenario_id: scenario for scenario in _load_manifest()}
    overrides = {
        "ref_07": {
            "detector": "orb",
            "detector_modalities": ["rgb"],
            "modality": "rgb",
            "rationale": "D6 counterfactual single-path routing for latency estimate",
        },
        "ref_11": {
            "detector": "yoloe",
            "detector_modalities": ["thermal"],
            "modality": "thermal",
            "rationale": "D6 counterfactual single-path routing for latency estimate",
        },
    }
    temp_dir = build_temp_reference_bank(
        selected_ref_ids=list(REFERENCE_IDS),
        per_reference_suppression=True,
        overrides=overrides,
        suffix="d6_latency_single_path",
    )
    try:
        results: dict[str, dict[str, Any]] = {}
        for scenario_id in PASS_A_SCENARIO_IDS:
            source = scenarios[scenario_id]
            scenario = ScenarioConfig(
                scenario_id=source.scenario_id,
                name=source.name,
                video=source.video,
                references_dir=str(temp_dir),
                reference_mode=source.reference_mode,
                frame_stride=source.frame_stride,
                frame_limit=source.frame_limit,
                expected_present_refs=list(source.expected_present_refs),
                modality_routing_breach_refs=list(source.modality_routing_breach_refs),
                ambiguous_routing_refs=list(source.ambiguous_routing_refs),
                tags=list(source.tags),
            )
            results[scenario_id] = _evaluate_scenario(scenario)
        return results
    finally:
        cleanup_temp_dir(temp_dir)


def _frame_stats_by_scenario(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for scenario_id, scenario in results.items():
        frames = list(scenario["frame_records"])
        payload[scenario_id] = {
            "frames": len(frames),
            "yoloe_ms": _stats(frames, "yoloe_ms"),
            "lightglue_ms": _stats(frames, "lightglue_ms"),
            "wall_ms": _stats(frames, "wall_ms"),
            "active_ref_mean": float(statistics.mean(float(record["active_ref_count"]) for record in frames)) if frames else 0.0,
            "both_active_ref_mean": float(statistics.mean(float(record["both_active_ref_count"]) for record in frames)) if frames else 0.0,
            "raw_candidate_mean": float(statistics.mean(float(record["raw_candidate_count"]) for record in frames)) if frames else 0.0,
            "filtered_candidate_mean": float(statistics.mean(float(record["filtered_candidate_count"]) for record in frames)) if frames else 0.0,
            "verified_mean": float(statistics.mean(float(record["verified_count"]) for record in frames)) if frames else 0.0,
        }
    return payload


def _all_frames(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for scenario in results.values():
        merged.extend(list(scenario["frame_records"]))
    return merged


def _delta(current: float, counterfactual: float) -> dict[str, float]:
    absolute = current - counterfactual
    relative = 0.0 if current == 0.0 else (absolute / current) * 100.0
    return {"absolute": float(absolute), "relative_percent": float(relative)}


def _render_stat_row(current_stats: dict[str, float], counter_stats: dict[str, float]) -> str:
    delta = _delta(float(current_stats["mean"]), float(counter_stats["mean"]))
    return (
        f"`mean={current_stats['mean']:.1f}->{counter_stats['mean']:.1f} ms` "
        f"`p50={current_stats['p50']:.1f}->{counter_stats['p50']:.1f}` "
        f"`p95={current_stats['p95']:.1f}->{counter_stats['p95']:.1f}` "
        f"`max={current_stats['max']:.1f}->{counter_stats['max']:.1f}` "
        f"`delta={delta['absolute']:.1f} ms ({delta['relative_percent']:.1f}%)`"
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    current = _load_current_results()
    counterfactual = _build_counterfactual_results()

    current_stats = _frame_stats_by_scenario(current)
    counter_stats = _frame_stats_by_scenario(counterfactual)
    overall_current = _all_frames(current)
    overall_counter = _all_frames(counterfactual)

    payload = {
        "current": current,
        "counterfactual_single_path": counterfactual,
        "current_stats": current_stats,
        "counterfactual_stats": counter_stats,
        "overall": {
            "current": {
                "yoloe_ms": _stats(overall_current, "yoloe_ms"),
                "lightglue_ms": _stats(overall_current, "lightglue_ms"),
                "wall_ms": _stats(overall_current, "wall_ms"),
            },
            "counterfactual_single_path": {
                "yoloe_ms": _stats(overall_counter, "yoloe_ms"),
                "lightglue_ms": _stats(overall_counter, "lightglue_ms"),
                "wall_ms": _stats(overall_counter, "wall_ms"),
            },
        },
    }
    (OUTPUT_DIR / "latency_real.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    overall_current_wall = payload["overall"]["current"]["wall_ms"]
    overall_counter_wall = payload["overall"]["counterfactual_single_path"]["wall_ms"]
    overall_current_yoloe = payload["overall"]["current"]["yoloe_ms"]
    overall_counter_yoloe = payload["overall"]["counterfactual_single_path"]["yoloe_ms"]
    overall_current_lg = payload["overall"]["current"]["lightglue_ms"]
    overall_counter_lg = payload["overall"]["counterfactual_single_path"]["lightglue_ms"]
    wall_delta = _delta(float(overall_current_wall["mean"]), float(overall_counter_wall["mean"]))
    yoloe_delta = _delta(float(overall_current_yoloe["mean"]), float(overall_counter_yoloe["mean"]))
    lg_delta = _delta(float(overall_current_lg["mean"]), float(overall_counter_lg["mean"]))
    wall_p95_delta = _delta(float(overall_current_wall["p95"]), float(overall_counter_wall["p95"]))
    if wall_delta["absolute"] > 0.0:
        wall_line = (
            f"- Forcing the two ambiguous refs to single-path saves mean wall time by `{wall_delta['absolute']:.1f} ms` "
            f"(`{wall_delta['relative_percent']:.1f}%`)."
        )
    else:
        wall_line = (
            f"- Forcing the two ambiguous refs to single-path does not improve mean wall time in this measurement. "
            f"The counterfactual is `{abs(wall_delta['absolute']):.1f} ms` slower on mean "
            f"(`{abs(wall_delta['relative_percent']):.1f}%`), while wall-clock `p95` still improves by "
            f"`{wall_p95_delta['absolute']:.1f} ms` (`{wall_p95_delta['relative_percent']:.1f}%`)."
        )
    if yoloe_delta["absolute"] > 0.0 and lg_delta["absolute"] > 0.0:
        component_line = (
            f"- The observed win comes from both components: YOLOE mean delta `{yoloe_delta['absolute']:.1f} ms`, "
            f"LightGlue mean delta `{lg_delta['absolute']:.1f} ms`."
        )
    else:
        component_line = (
            f"- There is no consistent component-level win in this run: YOLOE mean delta `{yoloe_delta['absolute']:.1f} ms`, "
            f"LightGlue mean delta `{lg_delta['absolute']:.1f} ms`."
        )

    lines = [
        "# D6 Latency Real",
        "",
        "- Scope: D5 Pass A's 4 frozen scenarios, measured from existing frame traces and a read-only counterfactual rerun where `ref_07` and `ref_11` are forced to single-path routing.",
        "- Competition budget: `1000 ms/frame` hard limit for 1 FPS.",
        "- Counterfactual routing: `ref_07 -> orb,rgb-only`, `ref_11 -> yoloe,thermal-only`.",
        "",
        "## Direct Answer",
        "",
        f"- Current Pass A is far above the 1 FPS budget: overall wall-clock `p50={overall_current_wall['p50']:.1f} ms`, `p95={overall_current_wall['p95']:.1f} ms`, `max={overall_current_wall['max']:.1f} ms`.",
        wall_line,
        component_line,
        "- Thermal scenarios remain the worst-case path. The counterfactual removes `ref_07` from thermal routing and removes `ref_11` from ORB, but the pipeline still stays well above budget.",
        "",
        "## Scenario Stats",
        "",
        "| Scenario | Current Wall | Counterfactual Wall | Current YOLOE | Counterfactual YOLOE | Current LightGlue | Counterfactual LightGlue | Both-Ref Mean | Raw Candidate Mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario_id in PASS_A_SCENARIO_IDS:
        current_entry = current_stats[scenario_id]
        counter_entry = counter_stats[scenario_id]
        lines.append(
            f"| {scenario_id} | mean={current_entry['wall_ms']['mean']:.1f} p95={current_entry['wall_ms']['p95']:.1f} | "
            f"mean={counter_entry['wall_ms']['mean']:.1f} p95={counter_entry['wall_ms']['p95']:.1f} | "
            f"mean={current_entry['yoloe_ms']['mean']:.1f} | mean={counter_entry['yoloe_ms']['mean']:.1f} | "
            f"mean={current_entry['lightglue_ms']['mean']:.1f} | mean={counter_entry['lightglue_ms']['mean']:.1f} | "
            f"{current_entry['both_active_ref_mean']:.1f}->{counter_entry['both_active_ref_mean']:.1f} | "
            f"{current_entry['raw_candidate_mean']:.2f}->{counter_entry['raw_candidate_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Wall: {_render_stat_row(overall_current_wall, overall_counter_wall)}",
            f"- YOLOE: {_render_stat_row(overall_current_yoloe, overall_counter_yoloe)}",
            f"- LightGlue: {_render_stat_row(overall_current_lg, overall_counter_lg)}",
            "",
        ]
    )

    (OUTPUT_DIR / "latency_real.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote latency report to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
