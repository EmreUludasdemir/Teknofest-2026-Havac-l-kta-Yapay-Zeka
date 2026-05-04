from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.diagnostic.d8a_common import PASS_A_SCENARIO_IDS, load_scenarios, minutes_for_frames, safe_mean
from tools.diagnostic.d8a_pass_a_gpu_rerun import ScenarioResult, _evaluate_scenario

try:
    import torch  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torch import failed in D8-C final ref_12 manifest probe") from exc

OUTPUT_DIR = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-05-02_d8c_final_ref12_manifest"
)
BASELINE_GPU_RESULTS_PATH = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-05-02_d8b_ref07_routing_override"
    / "pass_a_gpu_post_ref07_fix.json"
)


def _load_baseline() -> dict[str, Any]:
    return json.loads(BASELINE_GPU_RESULTS_PATH.read_text(encoding="utf-8"))


def _per_ref_triplet(result: dict[str, Any] | ScenarioResult) -> str:
    def _count(ref_id: str) -> int:
        if isinstance(result, dict):
            return int(result.get("per_ref_accept_counts", {}).get(ref_id, 0))
        return int(result.per_ref_accept_counts.get(ref_id, 0))

    return f"ref_04={_count('ref_04')}, ref_11={_count('ref_11')}, ref_12={_count('ref_12')}"


def _scenario_wall_mean(result: dict[str, Any] | ScenarioResult) -> float:
    if isinstance(result, dict):
        records = list(result.get("frame_records", []))
        return safe_mean(float(record.get("wall_ms", 0.0)) for record in records)
    return safe_mean(float(record.wall_ms) for record in result.frame_records)


def _scenario_p50(result: dict[str, Any] | ScenarioResult) -> float:
    if isinstance(result, dict):
        values = sorted(float(record.get("wall_ms", 0.0)) for record in result.get("frame_records", []))
    else:
        values = sorted(float(record.wall_ms) for record in result.frame_records)
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2 == 1:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _missing_expected_refs(result: ScenarioResult, expected_refs: list[str]) -> list[str]:
    return [ref_id for ref_id in expected_refs if int(result.per_ref_accept_counts.get(ref_id, 0)) == 0]


def _write_markdown(
    baseline: dict[str, Any],
    current: dict[str, ScenarioResult],
    expected_map: dict[str, list[str]],
) -> dict[str, Any]:
    lines = [
        "# D8-C Final Pass A Post ref_12 Manifest Promotion",
        "",
        "## Comparison",
        "",
        "| Scenario | D8-B Accepts | D8-C-Final Accepts | D8-B Guard | D8-C-Final Guard | D8-B Unexpected | D8-C-Final Unexpected | D8-B ref04/ref11/ref12 | D8-C-Final ref04/ref11/ref12 | Missing Expected Refs | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    decisions: dict[str, str] = {}
    thermal_cross = current["thermal_cross_sensor_proxy"]
    thermal_cross_missing = _missing_expected_refs(thermal_cross, expected_map["thermal_cross_sensor_proxy"])
    thermal_cross_hard_gate = int(thermal_cross.unexpected_accept_count) == 0
    thermal_absent_hard_gate = int(current["thermal_absent_target_proxy_2025"].accepted_match_count) == 0

    for scenario_id in PASS_A_SCENARIO_IDS:
        before = baseline[scenario_id]
        after = current[scenario_id]
        missing_expected = _missing_expected_refs(after, expected_map[scenario_id])
        if scenario_id == "thermal_cross_sensor_proxy":
            decision = "pass" if thermal_cross_hard_gate else "fail"
        elif scenario_id == "thermal_absent_target_proxy_2025":
            decision = "pass" if thermal_absent_hard_gate else "fail"
        else:
            decision = "monitor"
        decisions[scenario_id] = decision
        missing_label = ",".join(missing_expected) if missing_expected else "-"
        lines.append(
            f"| {scenario_id} | {int(before['accepted_match_count'])} | {after.accepted_match_count} | "
            f"{int(before['guard_accept_count'])} | {after.guard_accept_count} | "
            f"{int(before.get('unexpected_accept_count', 0))} | {after.unexpected_accept_count} | "
            f"{_per_ref_triplet(before)} | {_per_ref_triplet(after)} | {missing_label} | {decision} |"
        )

    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Scenario | D8-B Mean Wall ms | D8-C-Final Mean Wall ms | D8-B P50 ms | D8-C-Final P50 ms |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for scenario_id in PASS_A_SCENARIO_IDS:
        before = baseline[scenario_id]
        after = current[scenario_id]
        lines.append(
            f"| {scenario_id} | {_scenario_wall_mean(before):.1f} | {_scenario_wall_mean(after):.1f} | "
            f"{_scenario_p50(before):.1f} | {_scenario_p50(after):.1f} |"
        )

    total_mean_ms = safe_mean(_scenario_wall_mean(result) for result in current.values())
    total_minutes = minutes_for_frames(total_mean_ms)
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- `thermal_cross_sensor_proxy` unexpected accepts: `{thermal_cross.unexpected_accept_count}` -> `{'PASS' if thermal_cross_hard_gate else 'FAIL'}`.",
            f"- `thermal_cross_sensor_proxy` missing expected refs (informational): `{thermal_cross_missing}`.",
            f"- `thermal_absent_target_proxy_2025 = 0 FP`: `{'PASS' if thermal_absent_hard_gate else 'FAIL'}`.",
            (
                "- Manifest evaluator semantics are non-strict: `expected_present_refs` controls guard/unexpected accounting, "
                "but missing expected refs do not auto-fail the scenario."
            ),
            f"- Pass A overall mean wall extrapolation remains `{total_minutes:.1f} dakika` for `2250` frames.",
            "",
        ]
    )
    (OUTPUT_DIR / "pass_a_post_ref12_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "thermal_cross_hard_gate": thermal_cross_hard_gate,
        "thermal_cross_missing_expected_refs": thermal_cross_missing,
        "thermal_absent_hard_gate": thermal_absent_hard_gate,
        "overall_minutes_2250_frames": total_minutes,
        "decisions": decisions,
    }
    return payload


def _write_summary(payload: dict[str, Any], current: dict[str, ScenarioResult]) -> None:
    thermal_cross = current["thermal_cross_sensor_proxy"]
    rgb_ref = current["rgb_reference_session"]
    rgb_absent = current["rgb_absent_target_proxy_2025"]
    thermal_absent = current["thermal_absent_target_proxy_2025"]
    lines = [
        "# D8-C Final Summary",
        "",
        f"- `ref_12` manifest promotion result on `thermal_cross_sensor_proxy`: unexpected accepts=`{thermal_cross.unexpected_accept_count}` -> `{'PASS' if payload['thermal_cross_hard_gate'] else 'FAIL'}`.",
        f"- `thermal_cross_sensor_proxy` accepts: total=`{thermal_cross.accepted_match_count}`, `ref_04={int(thermal_cross.per_ref_accept_counts.get('ref_04', 0))}`, `ref_12={int(thermal_cross.per_ref_accept_counts.get('ref_12', 0))}`.",
        f"- `ref_04` remains a known missing expected ref: `{payload['thermal_cross_missing_expected_refs']}`.",
        f"- `rgb_reference_session` unchanged at total accepts=`{rgb_ref.accepted_match_count}`.",
        f"- `rgb_absent_target_proxy_2025` unchanged at total accepts=`{rgb_absent.accepted_match_count}`.",
        f"- `thermal_absent_target_proxy_2025` unchanged at total accepts=`{thermal_absent.accepted_match_count}`.",
        "- `ref_04` candidate-generation fix is intentionally not pursued in this pass; this is a manifest semantics decision only.",
        f"- Overall 2250-frame extrapolation: `{float(payload['overall_minutes_2250_frames']):.1f} dakika`.",
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA unavailable for D8-C final ref_12 manifest probe")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline()
    scenarios = load_scenarios(scenario_ids=PASS_A_SCENARIO_IDS)
    current: dict[str, ScenarioResult] = {}
    for scenario_id in PASS_A_SCENARIO_IDS:
        current[scenario_id] = _evaluate_scenario(scenarios[scenario_id])

    (OUTPUT_DIR / "pass_a_post_ref12_manifest.json").write_text(
        json.dumps({scenario_id: asdict(result) for scenario_id, result in current.items()}, indent=2),
        encoding="utf-8",
    )
    expected_map = {scenario_id: list(scenarios[scenario_id].expected_present_refs) for scenario_id in PASS_A_SCENARIO_IDS}
    payload = _write_markdown(baseline, current, expected_map)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary(payload, current)
    print(f"Wrote D8-C final ref_12 manifest outputs to {OUTPUT_DIR}")
    return 0 if bool(payload["thermal_cross_hard_gate"]) and bool(payload["thermal_absent_hard_gate"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
