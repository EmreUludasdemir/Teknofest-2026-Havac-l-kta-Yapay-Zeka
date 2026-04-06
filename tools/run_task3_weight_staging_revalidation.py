from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.evaluation.task3_experimental import compare_task3_baseline_vs_experimental
from src.task3.experimental.weight_staging import (
    STAGING_TRUE,
    evaluate_weight_staging,
    render_weight_staging_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3 YOLOE weight staging and revalidation")
    parser.add_argument("--manifest", default="data/task3_eval_manifest.json")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--search-root", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    staging = evaluate_weight_staging(
        MvpRuntimeSettings(),
        search_roots=args.search_root or None,
    )
    (output_dir / "task3_weight_staging_status.json").write_text(json.dumps(staging, indent=2), encoding="utf-8")
    (output_dir / "task3_weight_staging_status.md").write_text(
        render_weight_staging_markdown(staging),
        encoding="utf-8",
    )

    if staging.get("status") == STAGING_TRUE:
        settings = MvpRuntimeSettings(
            task3_experimental_enabled=True,
            task3_experimental_mode="yoloe_prompted",
            task3_experimental_tracking="light",
            task3_experimental_verifier="orb_homography",
            task3_experimental_model_path=Path(str(staging["chosen_weight"]["path"])),
        )
        payload = compare_task3_baseline_vs_experimental(
            manifest_path=args.manifest,
            runtime_settings=settings,
            output_dir=output_dir,
        )
        comparison = payload["comparison"]
        recommendation = str(comparison.get("recommendation", "EXPERIMENTAL ONLY"))
        true_exercised = True
    else:
        comparison_path = output_dir / "task3_baseline_vs_experimental.json"
        payload = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.exists() else {}
        payload["weight_staging"] = staging
        payload["phase_current_recommendation"] = "BLOCKED_BY_WEIGHT"
        payload["revalidation_performed"] = False
        comparison_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_blocked_comparison_markdown(output_dir, payload, staging)
        recommendation = "BLOCKED_BY_WEIGHT"
        true_exercised = False

    (output_dir / "task3_merge_recommendation.md").write_text(
        _render_merge_recommendation(staging, payload, recommendation, true_exercised),
        encoding="utf-8",
    )
    print(json.dumps({"status": staging.get("status"), "recommendation": recommendation}, indent=2))
    return 0


def _write_blocked_comparison_markdown(output_dir: Path, payload: dict[str, object], staging: dict[str, object]) -> None:
    baseline = payload.get("baseline", {}).get("aggregate", {}) if isinstance(payload.get("baseline"), dict) else {}
    experimental = payload.get("experimental", {}).get("aggregate", {}) if isinstance(payload.get("experimental"), dict) else {}
    output = (
        "# Task 3 Baseline vs Experimental\n\n"
        f"- Phase status: `BLOCKED_BY_WEIGHT`\n"
        f"- Revalidation with real YOLOE weight: `False`\n"
        f"- Weight staging status: `{staging.get('status')}`\n"
        "- Note: The metrics below are the last prompt_approx comparison already measured on this branch.\n\n"
        "| Metric | Baseline | Experimental | Delta |\n"
        "| --- | --- | --- | --- |\n"
        f"| Accepted Match Count | {baseline.get('accepted_match_count', '-')} | {experimental.get('accepted_match_count', '-')} | "
        f"{int(experimental.get('accepted_match_count', 0)) - int(baseline.get('accepted_match_count', 0)) if baseline and experimental else '-'} |\n"
        f"| Present Detection Frames | {baseline.get('target_present_detection_success_frames', '-')} | {experimental.get('target_present_detection_success_frames', '-')} | "
        f"{int(experimental.get('target_present_detection_success_frames', 0)) - int(baseline.get('target_present_detection_success_frames', 0)) if baseline and experimental else '-'} |\n"
        f"| False Positive Proxy | {baseline.get('false_positive_proxy_count', '-')} | {experimental.get('false_positive_proxy_count', '-')} | "
        f"{int(experimental.get('false_positive_proxy_count', 0)) - int(baseline.get('false_positive_proxy_count', 0)) if baseline and experimental else '-'} |\n"
        f"| Absent-target No-match Rate | {baseline.get('absent_target_no_match_rate', '-')} | {experimental.get('absent_target_no_match_rate', '-')} | "
        f"{round(float(experimental.get('absent_target_no_match_rate', 0.0)) - float(baseline.get('absent_target_no_match_rate', 0.0)), 6) if baseline and experimental else '-'} |\n"
        f"| Re-detection Success Count | {baseline.get('re_detection_success_count', '-')} | {experimental.get('re_detection_success_count', '-')} | "
        f"{int(experimental.get('re_detection_success_count', 0)) - int(baseline.get('re_detection_success_count', 0)) if baseline and experimental else '-'} |\n"
        f"| Runtime P50 ms | {baseline.get('runtime_p50_ms', '-')} | {experimental.get('runtime_p50_ms', '-')} | "
        f"{round(float(experimental.get('runtime_p50_ms', 0.0)) - float(baseline.get('runtime_p50_ms', 0.0)), 4) if baseline and experimental else '-'} |\n"
        f"| Runtime P95 ms | {baseline.get('runtime_p95_ms', '-')} | {experimental.get('runtime_p95_ms', '-')} | "
        f"{round(float(experimental.get('runtime_p95_ms', 0.0)) - float(baseline.get('runtime_p95_ms', 0.0)), 4) if baseline and experimental else '-'} |\n"
        f"| Peak VRAM MB | {baseline.get('peak_vram_mb', '-')} | {experimental.get('peak_vram_mb', '-')} | - |\n"
    )
    (output_dir / "task3_baseline_vs_experimental.md").write_text(output, encoding="utf-8")


def _render_merge_recommendation(
    staging: dict[str, object],
    payload: dict[str, object],
    recommendation: str,
    true_exercised: bool,
) -> str:
    baseline = payload.get("baseline", {}).get("aggregate", {}) if isinstance(payload.get("baseline"), dict) else {}
    experimental = payload.get("experimental", {}).get("aggregate", {}) if isinstance(payload.get("experimental"), dict) else {}
    comparison = payload.get("comparison", {}) if isinstance(payload.get("comparison"), dict) else {}
    chosen = staging.get("chosen_weight") or {}
    recall_delta = (
        int(experimental.get("target_present_detection_success_frames", 0))
        - int(baseline.get("target_present_detection_success_frames", 0))
        if baseline and experimental
        else None
    )
    runtime_ok = comparison.get("runtime_ok")
    return (
        "# Task 3 Merge Recommendation\n\n"
        f"- Final decision: `{recommendation}`\n"
        f"- True YOLOE exercised: `{true_exercised}`\n"
        f"- Weight staging status: `{staging.get('status')}`\n"
        f"- Weight file used: `{chosen.get('path') if chosen else 'None'}`\n"
        f"- Present-target recall delta: `{recall_delta}`\n"
        f"- Runtime acceptable: `{runtime_ok}`\n"
        f"- Integration status: `{comparison.get('integration_status', staging.get('status'))}`\n\n"
        "## Recommendation Logic\n"
        f"- If no real local YOLOE weight is staged, branch status stays `BLOCKED_BY_WEIGHT`.\n"
        f"- If true YOLOE is exercised but measured gain is still weak, keep branch `EXPERIMENTAL ONLY`.\n"
        f"- Partial cherry-pick is only reasonable for isolation/evaluation scaffolding, not production Task 3 replacement.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
