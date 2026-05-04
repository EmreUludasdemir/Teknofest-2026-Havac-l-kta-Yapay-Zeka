from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
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

MANIFEST_PATH = PROJECT_ROOT / "data" / "task3_eval_manifest.json"
REFERENCE_DIR = PROJECT_ROOT / "data" / "references" / "2026_baseline"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-04-27_d5_mixed_bank_integration"
)

PASS_A_SCENARIO_IDS = (
    "rgb_reference_session",
    "thermal_cross_sensor_proxy",
    "rgb_absent_target_proxy_2025",
    "thermal_absent_target_proxy_2025",
)
PASS_B_SCENARIO_IDS = PASS_A_SCENARIO_IDS + (
    "mixed_bank_rgb_video",
    "mixed_bank_thermal_video",
)
PASS_A_EXPECTATIONS = {
    "rgb_reference_session": {"guard_accept_count": 8, "per_ref": {"ref_02": 1, "ref_03": 5, "ref_05": 1, "ref_06": 1}},
    "thermal_cross_sensor_proxy": {"guard_accept_count": 1, "per_ref": {"ref_04": 1}},
    "rgb_absent_target_proxy_2025": {"guard_accept_count": 1, "per_ref": {"ref_02": 1}},
    "thermal_absent_target_proxy_2025": {"guard_accept_count": 0, "per_ref": {}},
}
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


@dataclass(slots=True)
class FrameRecord:
    scenario_id: str
    frame_index: int
    modality: str
    wall_ms: float
    yoloe_ms: float
    lightglue_ms: float
    homography_ms: float
    active_ref_count: int
    single_path_active_ref_count: int
    both_active_ref_count: int
    raw_candidate_count: int
    filtered_candidate_count: int
    verified_count: int
    ambiguous_raw_candidate_count: int
    ambiguous_filtered_candidate_count: int
    ambiguous_verified_count: int
    expected_verified_count: int
    unexpected_verified_count: int
    breach_verified_count: int
    accepted_ids: list[str]
    expected_accept_ids: list[str]
    unexpected_accept_ids: list[str]
    ambiguous_accept_ids: list[str]
    breach_accept_ids: list[str]
    yoloe_routed_refs: list[str]
    orb_routed_refs: list[str]
    effective_mode: str
    fallback_reason: str | None


@dataclass(slots=True)
class ScenarioResult:
    scenario_id: str
    pass_name: str
    reference_mode: str
    total_frames: int
    accepted_match_count: int
    false_positive_proxy_count: int
    guard_accept_count: int
    unexpected_accept_count: int
    ambiguous_accept_count: int
    breach_accept_count: int
    per_ref_accept_counts: dict[str, int]
    frame_records: list[FrameRecord]


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(percentile, 1.0)) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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
                references_dir=str(raw.get("references_dir") or raw.get("reference_dir") or REFERENCE_DIR),
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
        frame_url=f"http://task3-d5/frames/{frame_index + 1}/",
        image_url=f"/task3-d5/{frame_index + 1}.jpg",
        video_name=scenario_id,
        translation_x=0.0,
        translation_y=0.0,
        translation_z=0.0,
        health_status="1",
        metadata={"frame_index": frame_index, "image_width": width, "image_height": height},
    )


def _is_false_positive_proxy(match: Any, settings: MvpRuntimeSettings) -> bool:
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


def _evaluate_scenario(scenario: ScenarioConfig, *, pass_name: str) -> ScenarioResult:
    settings = _make_settings(scenario.references_dir)
    cache = ReferenceCache()
    cache.preload_from_directory(Path(scenario.references_dir), orb_features=settings.task3_orb_features)
    matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
    reference_ids = cache.list_ids()

    accepted_match_count = 0
    false_positive_proxy_count = 0
    guard_accept_count = 0
    unexpected_accept_count = 0
    ambiguous_accept_count = 0
    breach_accept_count = 0
    per_ref_accept_counts: Counter[str] = Counter()
    frame_records: list[FrameRecord] = []

    expected_set = set(scenario.expected_present_refs)
    ambiguous_set = set(scenario.ambiguous_routing_refs)
    breach_set = set(scenario.modality_routing_breach_refs)

    frames = iter_video_frames(
        scenario.video,
        frame_stride=scenario.frame_stride,
        limit=int(scenario.frame_limit) if scenario.frame_limit is not None else None,
        video_name=scenario.scenario_id,
    )
    total_frames = 0
    for ordinal, decoded in enumerate(frames):
        total_frames += 1
        frame = _frame_envelope(scenario.scenario_id, decoded.frame_index, decoded.width, decoded.height)
        active_ids = cache.filter_reference_ids_by_detector_modality(reference_ids, modality=decoded.modality)
        both_active_ids = [reference_id for reference_id in active_ids if cache.get_detector(reference_id) == "both"]
        single_path_active_ids = [reference_id for reference_id in active_ids if cache.get_detector(reference_id) != "both"]

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

        accepted_ids = [match.object_id for match in verified]
        expected_accept_ids = [reference_id for reference_id in accepted_ids if reference_id in expected_set]
        breach_accept_ids = [reference_id for reference_id in accepted_ids if reference_id in breach_set]
        ambiguous_accept_ids = [reference_id for reference_id in accepted_ids if reference_id in ambiguous_set]
        unexpected_accept_ids = [
            reference_id
            for reference_id in accepted_ids
            if reference_id not in expected_set and reference_id not in breach_set and reference_id not in ambiguous_set
        ]

        ambiguous_raw_candidate_count = sum(1 for match in raw_matches if match.object_id in ambiguous_set)
        ambiguous_filtered_candidate_count = sum(1 for match in filtered if match.object_id in ambiguous_set)

        accepted_match_count += len(verified)
        for match in verified:
            per_ref_accept_counts[match.object_id] += 1
            if _is_false_positive_proxy(match, settings):
                false_positive_proxy_count += 1

        if scenario.reference_mode == "synthetic_absent":
            guard_accept_count += len(verified)
        else:
            guard_accept_count += len(expected_accept_ids)
            unexpected_accept_count += len(unexpected_accept_ids)
        ambiguous_accept_count += len(ambiguous_accept_ids)
        breach_accept_count += len(breach_accept_ids)

        frame_records.append(
            FrameRecord(
                scenario_id=scenario.scenario_id,
                frame_index=int(decoded.frame_index),
                modality=str(decoded.modality),
                wall_ms=round(wall_ms, 6),
                yoloe_ms=round(float(task3_info.get("yoloe_inference_ms", 0.0)), 6),
                lightglue_ms=round(float(task3_info.get("lightglue_verify_ms_total", 0.0)), 6),
                homography_ms=round(float(task3_info.get("homography_compute_ms_total", 0.0)), 6),
                active_ref_count=len(active_ids),
                single_path_active_ref_count=len(single_path_active_ids),
                both_active_ref_count=len(both_active_ids),
                raw_candidate_count=len(raw_matches),
                filtered_candidate_count=len(filtered),
                verified_count=len(verified),
                ambiguous_raw_candidate_count=ambiguous_raw_candidate_count,
                ambiguous_filtered_candidate_count=ambiguous_filtered_candidate_count,
                ambiguous_verified_count=len(ambiguous_accept_ids),
                expected_verified_count=len(expected_accept_ids),
                unexpected_verified_count=len(unexpected_accept_ids),
                breach_verified_count=len(breach_accept_ids),
                accepted_ids=accepted_ids,
                expected_accept_ids=expected_accept_ids,
                unexpected_accept_ids=unexpected_accept_ids,
                ambiguous_accept_ids=ambiguous_accept_ids,
                breach_accept_ids=breach_accept_ids,
                yoloe_routed_refs=list(task3_info.get("yoloe_routed_refs", [])),
                orb_routed_refs=list(task3_info.get("orb_routed_refs", [])),
                effective_mode=str(task3_info.get("effective_mode", "unknown")),
                fallback_reason=task3_info.get("fallback_reason"),
            )
        )

    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        pass_name=pass_name,
        reference_mode=scenario.reference_mode,
        total_frames=total_frames,
        accepted_match_count=accepted_match_count,
        false_positive_proxy_count=false_positive_proxy_count,
        guard_accept_count=guard_accept_count,
        unexpected_accept_count=unexpected_accept_count,
        ambiguous_accept_count=ambiguous_accept_count,
        breach_accept_count=breach_accept_count,
        per_ref_accept_counts={reference_id: int(per_ref_accept_counts.get(reference_id, 0)) for reference_id in REFERENCE_IDS},
        frame_records=frame_records,
    )


def _evaluate_pass(scenarios: list[ScenarioConfig], *, pass_name: str, scenario_ids: tuple[str, ...]) -> dict[str, ScenarioResult]:
    selected = {scenario.scenario_id: scenario for scenario in scenarios}
    results: dict[str, ScenarioResult] = {}
    for scenario_id in scenario_ids:
        results[scenario_id] = _evaluate_scenario(selected[scenario_id], pass_name=pass_name)
    return results


def _validate_pass_a(results: dict[str, ScenarioResult]) -> list[str]:
    failures: list[str] = []
    for scenario_id, expectation in PASS_A_EXPECTATIONS.items():
        result = results[scenario_id]
        if result.guard_accept_count != int(expectation["guard_accept_count"]):
            failures.append(
                f"{scenario_id}: guard_accept_count expected {expectation['guard_accept_count']} got {result.guard_accept_count}"
            )
        expected_counts = expectation["per_ref"]
        for reference_id, expected_count in expected_counts.items():
            actual = int(result.per_ref_accept_counts.get(reference_id, 0))
            if actual != expected_count:
                failures.append(f"{scenario_id}: {reference_id} expected {expected_count} got {actual}")
        unexpected_expected_refs = [
            reference_id
            for reference_id in REFERENCE_IDS
            if reference_id not in expected_counts and int(result.per_ref_accept_counts.get(reference_id, 0)) > 0
        ]
        if scenario_id == "thermal_absent_target_proxy_2025" and unexpected_expected_refs:
            failures.append(
                f"{scenario_id}: expected zero accepts but got {unexpected_expected_refs}"
            )
    return failures


def _validate_pass_b(results: dict[str, ScenarioResult]) -> tuple[list[str], list[str]]:
    failures = _validate_pass_a({scenario_id: results[scenario_id] for scenario_id in PASS_A_SCENARIO_IDS})
    warnings: list[str] = []

    rgb_mixed = results["mixed_bank_rgb_video"]
    thermal_mixed = results["mixed_bank_thermal_video"]

    if rgb_mixed.breach_accept_count != 0:
        failures.append(f"mixed_bank_rgb_video: breach_accept_count expected 0 got {rgb_mixed.breach_accept_count}")
    if thermal_mixed.breach_accept_count != 0:
        failures.append(
            f"mixed_bank_thermal_video: breach_accept_count expected 0 got {thermal_mixed.breach_accept_count}"
        )

    if rgb_mixed.ambiguous_accept_count > 0:
        warnings.append(
            f"mixed_bank_rgb_video: ambiguous accepts observed ({rgb_mixed.ambiguous_accept_count}) via ref_07/ref_11"
        )
    if thermal_mixed.ambiguous_accept_count > 0:
        warnings.append(
            f"mixed_bank_thermal_video: ambiguous accepts observed ({thermal_mixed.ambiguous_accept_count}) via ref_07/ref_11"
        )

    return failures, warnings


def _render_summary_table(results: dict[str, ScenarioResult], scenarios: dict[str, ScenarioConfig], *, title: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Scenario | Mode | Frames | Production Accepts | Guard Count | Unexpected Accepts | Ambiguous Accepts | Breach Accepts | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario_id, result in results.items():
        decision = "pending"
        if scenario_id in PASS_A_EXPECTATIONS:
            decision = "pass" if result.guard_accept_count == PASS_A_EXPECTATIONS[scenario_id]["guard_accept_count"] else "fail"
        elif scenario_id.startswith("mixed_bank_"):
            decision = "pass" if result.breach_accept_count == 0 else "fail"
        lines.append(
            f"| {scenario_id} | {scenarios[scenario_id].reference_mode} | {result.total_frames} | {result.accepted_match_count} | "
            f"{result.guard_accept_count} | {result.unexpected_accept_count} | {result.ambiguous_accept_count} | "
            f"{result.breach_accept_count} | {decision} |"
        )
    lines.append("")
    return lines


def _render_per_ref_table(results: dict[str, ScenarioResult], *, title: str) -> list[str]:
    header = "| Scenario | " + " | ".join(REFERENCE_IDS) + " |"
    separator = "| --- | " + " | ".join("---" for _ in REFERENCE_IDS) + " |"
    lines = [f"## {title}", "", header, separator]
    for scenario_id, result in results.items():
        counts = [str(int(result.per_ref_accept_counts.get(reference_id, 0))) for reference_id in REFERENCE_IDS]
        lines.append(f"| {scenario_id} | " + " | ".join(counts) + " |")
    lines.append("")
    return lines


def _write_extended_scoreboard(
    pass_a_results: dict[str, ScenarioResult],
    pass_b_results: dict[str, ScenarioResult] | None,
    scenarios: dict[str, ScenarioConfig],
    pass_a_failures: list[str],
    pass_b_failures: list[str] | None = None,
    pass_b_warnings: list[str] | None = None,
) -> None:
    lines = ["# D5 Extended Scoreboard", ""]
    lines.extend(_render_summary_table(pass_a_results, scenarios, title="Pass A"))
    lines.extend(_render_per_ref_table(pass_a_results, title="Pass A Per-Ref Breakdown"))

    lines.append("### Pass A Frozen Guard")
    lines.append("")
    if pass_a_failures:
        for failure in pass_a_failures:
            lines.append(f"- FAIL: {failure}")
    else:
        lines.append("- PASS: frozen guard matched the D2-promote baseline under D5 counting semantics.")
    lines.append("")

    if pass_b_results is not None:
        lines.extend(_render_summary_table(pass_b_results, scenarios, title="Pass B"))
        lines.extend(_render_per_ref_table(pass_b_results, title="Pass B Per-Ref Breakdown"))
        lines.append("### Pass B Hard Gates")
        lines.append("")
        if pass_b_failures:
            for failure in pass_b_failures:
                lines.append(f"- FAIL: {failure}")
        else:
            lines.append("- PASS: all hard gates held in the six-scenario extended run.")
        if pass_b_warnings:
            for warning in pass_b_warnings:
            lines.append(f"- WARN: {warning}")
        else:
            lines.append("- WARN: no ambiguous-routing accepts were observed in mixed-bank scenarios.")
        lines.append("")
    else:
        lines.append("### Pass B")
        lines.append("")
        lines.append("- Not run. D5 stop condition triggered after Pass A regression.")
        lines.append("")

    (OUTPUT_DIR / "extended_scoreboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_modality_routing_audit(results: dict[str, ScenarioResult], scenarios: dict[str, ScenarioConfig]) -> None:
    mixed_ids = ("mixed_bank_rgb_video", "mixed_bank_thermal_video")
    lines = [
        "# D5 Modality Routing Audit",
        "",
        "| Scenario | Breach Refs | Breach Accepts | Ambiguous Refs | Ambiguous Accepts | Unexpected Accepts |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for scenario_id in mixed_ids:
        result = results[scenario_id]
        scenario = scenarios[scenario_id]
        lines.append(
            f"| {scenario_id} | {','.join(scenario.modality_routing_breach_refs) or '-'} | {result.breach_accept_count} | "
            f"{','.join(scenario.ambiguous_routing_refs) or '-'} | {result.ambiguous_accept_count} | {result.unexpected_accept_count} |"
        )

    lines.extend(["", "## Breach Frames", ""])
    breach_rows = []
    for scenario_id in mixed_ids:
        for record in results[scenario_id].frame_records:
            if record.breach_accept_ids:
                breach_rows.append((scenario_id, record.frame_index, record.modality, record.breach_accept_ids, record.accepted_ids))
    if breach_rows:
        for scenario_id, frame_index, modality, breach_ids, accepted_ids in breach_rows:
            lines.append(
                f"- `{scenario_id}` frame `{frame_index}` modality `{modality}` breach accepts `{breach_ids}` full accepts `{accepted_ids}`"
            )
    else:
        lines.append("- No hard-gate modality routing breach was observed in Pass B.")

    lines.extend(["", "## Ambiguous Routing Frames", ""])
    ambiguous_rows = []
    for scenario_id in mixed_ids:
        for record in results[scenario_id].frame_records:
            if record.ambiguous_accept_ids:
                ambiguous_rows.append(
                    (scenario_id, record.frame_index, record.modality, record.ambiguous_accept_ids, record.accepted_ids)
                )
    if ambiguous_rows:
        for scenario_id, frame_index, modality, ambiguous_ids, accepted_ids in ambiguous_rows:
            lines.append(
                f"- `{scenario_id}` frame `{frame_index}` modality `{modality}` ambiguous accepts `{ambiguous_ids}` full accepts `{accepted_ids}`"
            )
    else:
        lines.append("- No ambiguous-routing accept was observed in Pass B.")

    (OUTPUT_DIR / "modality_routing_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latency_report(results: dict[str, ScenarioResult]) -> None:
    lines = [
        "# D5 Latency Per Frame",
        "",
        "| Scenario | Avg Wall ms/frame | P95 Wall ms | Max Wall ms | Avg Active Refs | Avg Single-Path Refs | Avg Both Refs | Avg ms per Single-Path Ref | Avg Both Filtered Candidates/frame | Budget |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario_id, result in results.items():
        wall_values = [record.wall_ms for record in result.frame_records]
        active_ref_values = [float(record.active_ref_count) for record in result.frame_records]
        single_path_values = [float(record.single_path_active_ref_count) for record in result.frame_records]
        both_values = [float(record.both_active_ref_count) for record in result.frame_records]
        both_filtered_values = [float(record.ambiguous_filtered_candidate_count) for record in result.frame_records]
        per_single_path_ref_values = [
            record.wall_ms / max(record.single_path_active_ref_count, 1) for record in result.frame_records
        ]
        avg_wall_ms = _safe_mean(wall_values)
        budget = "pass" if avg_wall_ms <= 1000.0 else "fail"
        lines.append(
            f"| {scenario_id} | {avg_wall_ms:.3f} | {_percentile(wall_values, 0.95):.3f} | {max(wall_values) if wall_values else 0.0:.3f} | "
            f"{_safe_mean(active_ref_values):.3f} | {_safe_mean(single_path_values):.3f} | {_safe_mean(both_values):.3f} | "
            f"{_safe_mean(per_single_path_ref_values):.3f} | {_safe_mean(both_filtered_values):.3f} | {budget} |"
        )

    lines.extend(
        [
            "",
            "## Ambiguous Ref Contribution",
            "",
            "| Scenario | ref_07 Accepts | ref_11 Accepts | Avg Ambiguous Raw Candidates/frame | Avg Ambiguous Filtered Candidates/frame | Avg Ambiguous Verified/frame |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for scenario_id, result in results.items():
        ambiguous_raw = [float(record.ambiguous_raw_candidate_count) for record in result.frame_records]
        ambiguous_filtered = [float(record.ambiguous_filtered_candidate_count) for record in result.frame_records]
        ambiguous_verified = [float(record.ambiguous_verified_count) for record in result.frame_records]
        lines.append(
            f"| {scenario_id} | {result.per_ref_accept_counts.get('ref_07', 0)} | {result.per_ref_accept_counts.get('ref_11', 0)} | "
            f"{_safe_mean(ambiguous_raw):.3f} | {_safe_mean(ambiguous_filtered):.3f} | {_safe_mean(ambiguous_verified):.3f} |"
        )

    (OUTPUT_DIR / "latency_per_frame.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = {scenario.scenario_id: scenario for scenario in _load_manifest()}

    pass_a_results = _evaluate_pass(list(scenarios.values()), pass_name="pass_a", scenario_ids=PASS_A_SCENARIO_IDS)
    pass_a_failures = _validate_pass_a(pass_a_results)
    _write_extended_scoreboard(pass_a_results, None, scenarios, pass_a_failures)
    _write_json(
        OUTPUT_DIR / "pass_a_results.json",
        {scenario_id: asdict(result) for scenario_id, result in pass_a_results.items()},
    )
    if pass_a_failures:
        return 2

    pass_b_results = _evaluate_pass(list(scenarios.values()), pass_name="pass_b", scenario_ids=PASS_B_SCENARIO_IDS)
    pass_b_failures, pass_b_warnings = _validate_pass_b(pass_b_results)
    _write_extended_scoreboard(pass_a_results, pass_b_results, scenarios, pass_a_failures, pass_b_failures, pass_b_warnings)
    _write_modality_routing_audit(pass_b_results, scenarios)
    _write_latency_report(pass_b_results)
    _write_json(
        OUTPUT_DIR / "pass_b_results.json",
        {scenario_id: asdict(result) for scenario_id, result in pass_b_results.items()},
    )
    return 3 if pass_b_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
