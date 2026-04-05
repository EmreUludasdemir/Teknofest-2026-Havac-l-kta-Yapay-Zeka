from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.evaluation.task2_long_sequence import evaluate_task2_long_sequences
from src.evaluation.task3_baseline import evaluate_task3_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TEKNOFEST Faz 5 replay raporlari")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--task2-limit", type=int, default=600)
    parser.add_argument("--task2-stride", type=int, default=4)
    parser.add_argument("--task3-limit", type=int, default=90)
    parser.add_argument("--task3-stride", type=int, default=60)
    args = parser.parse_args(argv)

    settings = MvpRuntimeSettings(
        task2_eval_sequence_limit=args.task2_limit,
        task2_eval_frame_stride=args.task2_stride,
        task3_eval_frame_limit=args.task3_limit,
        task3_eval_frame_stride=args.task3_stride,
    )
    evaluate_task2_long_sequences(runtime_settings=settings, output_dir=args.output_dir)
    evaluate_task3_baseline(runtime_settings=settings, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
