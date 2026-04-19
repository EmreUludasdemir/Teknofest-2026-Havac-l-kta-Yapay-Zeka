from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.evaluation.task3_manifest_eval import evaluate_task3_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 3 manifest-driven ORB vs YOLOE comparison")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default="_logs/reports_generated/task3_manifest")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--scenario-id", action="append", default=None)
    parser.add_argument("--debug-dump-rejects", action="store_true")
    parser.add_argument("--allow-cpu-yoloe", action="store_true")
    args = parser.parse_args(argv)

    settings = MvpRuntimeSettings()
    if args.allow_cpu_yoloe:
        settings = replace(settings, task3_yoloe_allow_cpu=True)
    if args.debug_dump_rejects:
        settings = replace(settings, task3_debug_dump_rejects=True)
    resolved_output_dir = Path(args.output_dir)
    if args.run_tag:
        resolved_output_dir = resolved_output_dir / f"{date.today().isoformat()}_{args.run_tag}"
    evaluate_task3_manifest(
        runtime_settings=settings,
        manifest_path=args.manifest or settings.task3_eval_manifest_path,
        output_dir=resolved_output_dir,
        scenario_ids=tuple(args.scenario_id) if args.scenario_id else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
