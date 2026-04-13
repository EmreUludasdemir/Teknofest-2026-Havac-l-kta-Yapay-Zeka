from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.evaluation.export_readiness import evaluate_export_readiness
from src.evaluation.task2_long_sequence import (
    evaluate_task2_long_sequences,
    snapshot_existing_task2_report,
    write_task2_after_snapshot,
    write_task2_comparison,
)
from src.evaluation.task3_baseline import evaluate_task3_baseline
from src.tools.report_paths import GENERATED_REPORTS_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TEKNOFEST Faz 6 rapor orkestrasyonu")
    parser.add_argument("--output-dir", default=str(GENERATED_REPORTS_ROOT))
    parser.add_argument("--task2-limit", type=int, default=600)
    parser.add_argument("--task2-stride", type=int, default=4)
    parser.add_argument("--task3-limit", type=int, default=45)
    parser.add_argument("--task3-stride", type=int, default=60)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    settings = MvpRuntimeSettings(
        task2_eval_sequence_limit=args.task2_limit,
        task2_eval_frame_stride=args.task2_stride,
        task3_eval_frame_limit=args.task3_limit,
        task3_eval_frame_stride=args.task3_stride,
    )

    snapshot_existing_task2_report(output_dir, suffix="before")
    evaluate_task2_long_sequences(runtime_settings=settings, output_dir=output_dir)
    write_task2_after_snapshot(output_dir)
    write_task2_comparison(output_dir)

    evaluate_task3_baseline(runtime_settings=settings, output_dir=output_dir, mode="orb_template")
    evaluate_task3_baseline(runtime_settings=settings, output_dir=output_dir, mode="learned_descriptor")

    evaluate_export_readiness(reports_dir=output_dir, tests_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
