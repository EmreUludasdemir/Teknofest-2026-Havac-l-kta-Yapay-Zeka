from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.task2_thermal_microlever import evaluate_task2_thermal_microlever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 2 thermal-only microlever evaluator")
    parser.add_argument("--manifest", default="src/evaluation/task2_thermal_microlever_manifest.json")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument(
        "--variant",
        default=None,
        choices=(None, "baseline", "thermal_confidence_floor_045", "thermal_sensor_hint_weight_008"),
    )
    args = parser.parse_args(argv)

    payload = evaluate_task2_thermal_microlever(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        scenario_id=args.scenario_id,
        variant=args.variant,
    )
    print(json.dumps({"final_recommendation": payload.get("final_recommendation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
