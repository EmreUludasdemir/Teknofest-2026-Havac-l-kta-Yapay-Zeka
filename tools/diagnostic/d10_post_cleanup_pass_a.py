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
    raise RuntimeError("torch import failed for D10 post-cleanup validation") from exc


OUTPUT_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-05-03_d10_repo_cleanup"
BASELINE_PATH = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-05-02_d8c_final_ref12_manifest"
    / "pass_a_post_ref12_manifest.json"
)


def _load_baseline() -> dict[str, Any]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _wall_stats(result: ScenarioResult) -> dict[str, float]:
    walls = [float(record.wall_ms) for record in result.frame_records]
    if not walls:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(walls)
    middle = len(ordered) // 2
    p50 = ordered[middle] if len(ordered) % 2 == 1 else (ordered[middle - 1] + ordered[middle]) / 2.0
    p95_index = min(max(int(round((len(ordered) - 1) * 0.95)), 0), len(ordered) - 1)
    return {
        "mean": round(safe_mean(walls), 6),
        "p50": round(float(p50), 6),
        "p95": round(float(ordered[p95_index]), 6),
        "max": round(float(max(ordered)), 6),
    }


def _compare(baseline: dict[str, Any], current: ScenarioResult) -> list[str]:
    failures: list[str] = []
    for key in ("accepted_match_count", "guard_accept_count", "unexpected_accept_count", "false_positive_proxy_count"):
        if int(baseline.get(key, 0)) != int(getattr(current, key)):
            failures.append(f"{key}: expected {int(baseline.get(key, 0))} got {int(getattr(current, key))}")
    baseline_per_ref = dict(baseline.get("per_ref_accept_counts", {}))
    if baseline_per_ref != current.per_ref_accept_counts:
        diffs = []
        for ref_id in sorted(set(baseline_per_ref) | set(current.per_ref_accept_counts)):
            before = int(baseline_per_ref.get(ref_id, 0))
            after = int(current.per_ref_accept_counts.get(ref_id, 0))
            if before != after:
                diffs.append(f"{ref_id}:{before}->{after}")
        failures.append("per_ref_accept_counts: " + ", ".join(diffs))
    return failures


def main() -> int:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA unavailable for D10 post-cleanup validation")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline()
    scenarios = load_scenarios(scenario_ids=PASS_A_SCENARIO_IDS)
    current: dict[str, ScenarioResult] = {}
    failures: list[str] = []
    all_wall_ms: list[float] = []

    for scenario_id in PASS_A_SCENARIO_IDS:
        result = _evaluate_scenario(scenarios[scenario_id])
        current[scenario_id] = result
        all_wall_ms.extend(float(record.wall_ms) for record in result.frame_records)
        for failure in _compare(baseline[scenario_id], result):
            failures.append(f"{scenario_id}: {failure}")

    overall_mean = safe_mean(all_wall_ms)
    overall_minutes = minutes_for_frames(overall_mean)
    overall_stats = _wall_stats(
        ScenarioResult(
            scenario_id="overall",
            total_frames=sum(result.total_frames for result in current.values()),
            accepted_match_count=sum(result.accepted_match_count for result in current.values()),
            false_positive_proxy_count=sum(result.false_positive_proxy_count for result in current.values()),
            guard_accept_count=sum(result.guard_accept_count for result in current.values()),
            unexpected_accept_count=sum(result.unexpected_accept_count for result in current.values()),
            per_ref_accept_counts={},
            frame_records=[record for result in current.values() for record in result.frame_records],
        )
    )
    latency_failures: list[str] = []
    if float(overall_stats["p50"]) >= 1000.0:
        latency_failures.append(f"p50 expected < 1000 ms, got {overall_stats['p50']:.1f}")
    if float(overall_minutes) >= 60.0:
        latency_failures.append(f"2250-frame extrapolation expected < 60 min, got {overall_minutes:.1f}")
    failures.extend(f"latency: {item}" for item in latency_failures)

    payload = {scenario_id: asdict(result) for scenario_id, result in current.items()}
    (OUTPUT_DIR / "post_cleanup_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# D10 Post-Cleanup Validation",
        "",
        "| Scenario | Baseline Accept | Current Accept | Baseline Guard | Current Guard | Baseline Unexpected | Current Unexpected | Baseline FP | Current FP | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario_id in PASS_A_SCENARIO_IDS:
        before = baseline[scenario_id]
        after = current[scenario_id]
        scenario_failures = _compare(before, after)
        lines.append(
            f"| {scenario_id} | {int(before['accepted_match_count'])} | {after.accepted_match_count} | "
            f"{int(before['guard_accept_count'])} | {after.guard_accept_count} | "
            f"{int(before['unexpected_accept_count'])} | {after.unexpected_accept_count} | "
            f"{int(before['false_positive_proxy_count'])} | {after.false_positive_proxy_count} | "
            f"{'PASS' if not scenario_failures else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            f"- Overall wall p50/p95/max: `{overall_stats['p50']:.1f} / {overall_stats['p95']:.1f} / {overall_stats['max']:.1f} ms`.",
            f"- 2250-frame extrapolation: `{overall_minutes:.1f} dakika`.",
            "",
            "## Gate",
            "",
        ]
    )
    if failures:
        for item in failures:
            lines.append(f"- FAIL: {item}")
    else:
        lines.append("- PASS: Cleanup branch preserved D8-C-Final scoreboard and stayed within latency gates.")
    lines.append("")
    (OUTPUT_DIR / "post_cleanup_validation.md").write_text("\n".join(lines), encoding="utf-8")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
