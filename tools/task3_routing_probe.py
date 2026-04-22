from __future__ import annotations

"""Probe utility for standalone Task 3 detector-assignment policy."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task3.routing_policy import assign_detector


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Task 3 detector assignments.")
    parser.add_argument("images", nargs="+", type=Path, help="One or more image paths to route.")
    args = parser.parse_args()

    for image_path in args.images:
        assignment = assign_detector(image_path)
        print(f"{image_path.name}: detector={assignment.detector} modalities={assignment.detector_modalities}")
        print(f"  confidence={assignment.confidence} rationale={assignment.rationale}")
        print(f"  signals={assignment.signals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
