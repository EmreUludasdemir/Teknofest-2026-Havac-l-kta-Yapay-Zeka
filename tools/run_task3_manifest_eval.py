from __future__ import annotations

import argparse
import sys
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
    args = parser.parse_args(argv)

    settings = MvpRuntimeSettings()
    evaluate_task3_manifest(
        runtime_settings=settings,
        manifest_path=args.manifest or settings.task3_eval_manifest_path,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
