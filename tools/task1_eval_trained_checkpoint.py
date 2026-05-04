from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.task1_rfdetr_comparison import run_task1_rfdetr_comparison
from src.task1.experimental import TARGET_CLASS_NAMES, collect_split_class_support, extract_detection_metrics


def _load_ultralytics() -> Any:
    from ultralytics import YOLO  # type: ignore[import-not-found]

    return YOLO


def _run_label_aware_eval(
    *,
    candidate_path: str,
    dataset_yaml: str,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    YOLO = _load_ultralytics()
    model = YOLO(candidate_path)
    results = model.val(data=dataset_yaml, split=split, imgsz=imgsz, batch=batch, device=device)
    support = collect_split_class_support(dataset_yaml, split)
    payload = extract_detection_metrics(results, class_names=list(TARGET_CLASS_NAMES), split_support=support)
    payload["split"] = split
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Task 1 checkpoint.")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--dataset-root")
    parser.add_argument("--labels-format", choices=["coco", "yolo"])
    parser.add_argument("--dataset-yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tiled", action="store_true")
    args = parser.parse_args(argv)

    if args.dataset_yaml:
        payload = _run_label_aware_eval(
            candidate_path=args.candidate_path,
            dataset_yaml=args.dataset_yaml,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "candidate_name": args.candidate_name,
                    "candidate_path": args.candidate_path,
                    "mode": "label_aware",
                    "evaluation": payload,
                },
                indent=2,
            )
        )
        return 0

    payload = run_task1_rfdetr_comparison(
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        variants=["candidate_tiled_probe" if args.tiled else "candidate_probe"],
        dataset_root=args.dataset_root,
        labels_format=args.labels_format,
        candidate_name=args.candidate_name,
        candidate_path=args.candidate_path,
    )
    variant_key = "candidate_tiled_probe" if args.tiled else "candidate_probe"
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "blockers": payload["blockers"],
                "candidate_variant": payload["variants"].get(variant_key),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
