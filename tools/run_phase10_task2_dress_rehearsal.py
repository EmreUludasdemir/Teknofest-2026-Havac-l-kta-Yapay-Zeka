from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.evaluation.task2_long_sequence import evaluate_task2_phase10_dress_rehearsal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 10 Task 2 dress rehearsal")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--sequence-limit", type=int, default=600)
    args = parser.parse_args(argv)

    payload = evaluate_task2_phase10_dress_rehearsal(
        runtime_settings=replace(MvpRuntimeSettings(), task2_eval_sequence_limit=args.sequence_limit),
        root_dir=args.root_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload["assessment"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
