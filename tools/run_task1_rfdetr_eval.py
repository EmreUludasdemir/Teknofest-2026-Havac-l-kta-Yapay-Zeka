from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.task1_rfdetr_comparison import run_task1_rfdetr_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Task 1 experimental Task 1 comparison")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument(
        "--variant",
        action="append",
        choices=["baseline_yolo26n", "control_yolo11n", "tiled_yolo26n_probe", "rtdetr_probe", "rfdetr_base", "candidate_probe", "candidate_tiled_probe"],
        help="Variant(s) to execute. Defaults to baseline_yolo26n, control_yolo11n, tiled_yolo26n_probe.",
    )
    parser.add_argument("--tile-size", type=int, default=960)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--dataset-root")
    parser.add_argument("--labels-format", choices=["coco", "yolo"])
    parser.add_argument("--candidate-name")
    parser.add_argument("--candidate-path")
    parser.add_argument("--candidate-variant-name", default="candidate_probe")
    args = parser.parse_args(argv)
    payload = run_task1_rfdetr_comparison(
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        variants=args.variant,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        dataset_root=args.dataset_root,
        labels_format=args.labels_format,
        candidate_name=args.candidate_name,
        candidate_path=args.candidate_path,
        candidate_variant_name=args.candidate_variant_name,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "blockers": payload["blockers"],
                "requested_variants": payload["requested_variants"],
                "proxy_summary": payload["proxy_summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
