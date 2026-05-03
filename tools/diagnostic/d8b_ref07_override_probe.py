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
    raise RuntimeError("torch import failed in D8-B ref_07 override probe") from exc

OUTPUT_DIR = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-05-02_d8b_ref07_routing_override"
)
BASELINE_GPU_RESULTS_PATH = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-05-02_d8a_gpu_enable"
    / "pass_a_gpu_rerun.json"
)


def _load_baseline() -> dict[str, Any]:
    return json.loads(BASELINE_GPU_RESULTS_PATH.read_text(encoding="utf-8"))


def _per_ref_triplet(result: dict[str, Any] | ScenarioResult) -> str:
    def _count(ref_id: str) -> int:
        if isinstance(result, dict):
            return int(result.get("per_ref_accept_counts", {}).get(ref_id, 0))
        return int(result.per_ref_accept_counts.get(ref_id, 0))

    return f"ref_07={_count('ref_07')}, ref_11={_count('ref_11')}, ref_12={_count('ref_12')}"


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


def _write_markdown(baseline: dict[str, Any], current: dict[str, ScenarioResult]) -> dict[str, Any]:
    lines = [
        "# D8-B Pass A GPU Post ref_07 Fix",
        "",
        "## Comparison",
        "",
        "| Scenario | D8-A Accepts | D8-B Accepts | D8-A Guard | D8-B Guard | D8-A Unexpected | D8-B Unexpected | D8-A ref07/ref11/ref12 | D8-B ref07/ref11/ref12 | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    hard_gate_ok = int(current["thermal_absent_target_proxy_2025"].accepted_match_count) == 0
    thermal_cross_ref12_preserved = int(current["thermal_cross_sensor_proxy"].per_ref_accept_counts.get("ref_12", 0)) == int(
        baseline["thermal_cross_sensor_proxy"]["per_ref_accept_counts"].get("ref_12", 0)
    )
    rgb_ref11_preserved = int(current["rgb_reference_session"].per_ref_accept_counts.get("ref_11", 0)) == int(
        baseline["rgb_reference_session"]["per_ref_accept_counts"].get("ref_11", 0)
    )

    decisions: dict[str, str] = {}
    for scenario_id in PASS_A_SCENARIO_IDS:
        before = baseline[scenario_id]
        after = current[scenario_id]
        if scenario_id == "thermal_absent_target_proxy_2025":
            decision = "pass" if hard_gate_ok else "fail"
        elif scenario_id == "thermal_cross_sensor_proxy":
            decision = "pass" if int(after.per_ref_accept_counts.get("ref_07", 0)) == 0 else "fail"
        else:
            decision = "monitor"
        decisions[scenario_id] = decision
        lines.append(
            f"| {scenario_id} | {int(before['accepted_match_count'])} | {after.accepted_match_count} | "
            f"{int(before['guard_accept_count'])} | {after.guard_accept_count} | "
            f"{int(before.get('unexpected_accept_count', 0))} | {after.unexpected_accept_count} | "
            f"{_per_ref_triplet(before)} | {_per_ref_triplet(after)} | {decision} |"
        )

    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Scenario | D8-A Mean Wall ms | D8-B Mean Wall ms | D8-A P50 ms | D8-B P50 ms |",
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
            f"- Hard gate `thermal_absent_target_proxy_2025 = 0 FP`: `{'PASS' if hard_gate_ok else 'FAIL'}`.",
            f"- `ref_12` thermal accept preservation in `thermal_cross_sensor_proxy`: `{'PASS' if thermal_cross_ref12_preserved else 'FAIL'}`.",
            f"- `ref_11` RGB-side accept preservation in `rgb_reference_session`: `{'PASS' if rgb_ref11_preserved else 'FAIL'}`.",
            f"- Pass A overall mean wall extrapolation remains `{total_minutes:.1f} dakika` for `2250` frames.",
            "",
        ]
    )
    (OUTPUT_DIR / "pass_a_gpu_post_ref07_fix.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "hard_gate_ok": hard_gate_ok,
        "thermal_cross_ref12_preserved": thermal_cross_ref12_preserved,
        "rgb_ref11_preserved": rgb_ref11_preserved,
        "overall_minutes_2250_frames": total_minutes,
        "decisions": decisions,
    }
    return payload


def _write_summary(payload: dict[str, Any], current: dict[str, ScenarioResult]) -> None:
    hard_gate_ok = bool(payload["hard_gate_ok"])
    thermal_cross = current["thermal_cross_sensor_proxy"]
    thermal_absent = current["thermal_absent_target_proxy_2025"]
    rgb_ref = current["rgb_reference_session"]
    lines = [
        "# D8-B Summary",
        "",
        f"- Hard gate `thermal_absent_target_proxy_2025 = 0 FP`: `{'PASS' if hard_gate_ok else 'FAIL'}`.",
        f"- `thermal_cross_sensor_proxy` post-fix accepts: total=`{thermal_cross.accepted_match_count}`, "
        f"`ref_07={int(thermal_cross.per_ref_accept_counts.get('ref_07', 0))}`, "
        f"`ref_12={int(thermal_cross.per_ref_accept_counts.get('ref_12', 0))}`.",
        f"- `thermal_absent_target_proxy_2025` post-fix accepts: total=`{thermal_absent.accepted_match_count}`, "
        f"`ref_07={int(thermal_absent.per_ref_accept_counts.get('ref_07', 0))}`.",
        f"- `rgb_reference_session` post-fix accepts: total=`{rgb_ref.accepted_match_count}`, "
        f"`ref_07={int(rgb_ref.per_ref_accept_counts.get('ref_07', 0))}`, "
        f"`ref_11={int(rgb_ref.per_ref_accept_counts.get('ref_11', 0))}`.",
        f"- `ref_12` preservation: `{'PASS' if payload['thermal_cross_ref12_preserved'] else 'FAIL'}`. "
        f"`ref_11` preservation: `{'PASS' if payload['rgb_ref11_preserved'] else 'FAIL'}`.",
        f"- Overall 2250-frame extrapolation: `{float(payload['overall_minutes_2250_frames']):.1f} dakika`.",
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA unavailable for D8-B ref_07 override probe")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline()
    scenarios = load_scenarios(scenario_ids=PASS_A_SCENARIO_IDS)
    current: dict[str, ScenarioResult] = {}
    for scenario_id in PASS_A_SCENARIO_IDS:
        current[scenario_id] = _evaluate_scenario(scenarios[scenario_id])

    (OUTPUT_DIR / "pass_a_gpu_post_ref07_fix.json").write_text(
        json.dumps({scenario_id: asdict(result) for scenario_id, result in current.items()}, indent=2),
        encoding="utf-8",
    )
    payload = _write_markdown(baseline, current)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary(payload, current)
    print(f"Wrote D8-B ref_07 override outputs to {OUTPUT_DIR}")
    return 0 if bool(payload["hard_gate_ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
