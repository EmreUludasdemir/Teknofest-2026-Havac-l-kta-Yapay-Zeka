from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.experimental import (
    DATA_ROOT,
    LOCAL_FRAMES_EXTRACT_DIR,
    LOCAL_LABELS_EXTRACT_DIR,
    REPORTS_ROOT,
    TASK1_PUBLIC_DOWNLOADS_ROOT,
    TASK1_STAGING_ROOT,
    TASK1_YOLO_ROOT,
    build_combined_yolo_dataset,
    build_data_quality_report,
    build_local_yolo_dataset,
    convert_visdrone_to_yolo,
    create_offline_weather_augmented_subset,
    extract_zip_safe,
    initial_experiment_results_payload,
    render_class_mapping_report,
    render_training_plan_markdown,
    stage_unlabeled_local_videos,
)


def render_quality_markdown(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Task 1 Data Quality Report",
            "",
            f"- image count: `{payload['image_count']}`",
            f"- label count: `{payload['label_count']}`",
            f"- decode failures: `{payload['decode_failures']}`",
            f"- empty label files: `{payload['empty_label_files']}`",
            f"- bbox out of range: `{payload['bbox_out_of_range']}`",
            f"- total boxes: `{payload['total_boxes']}`",
            f"- tiny box count: `{payload['tiny_box_count']}`",
            f"- tiny box ratio: `{payload['tiny_box_ratio']}`",
            f"- exact duplicate images: `{payload['exact_duplicate_images']}`",
            f"- near duplicate images: `{payload['near_duplicate_images']}`",
            f"- missing paired labels: `{payload['missing_paired_labels']}`",
        ]
    )


def render_experiment_results_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Task 1 Experiment Results",
        "",
        f"**Decision:** `{payload['decision']}`",
        "",
    ]
    for name, result in payload["experiments"].items():
        lines.extend(
            [
                f"## `{name}`",
                "",
                f"- status: `{result['status']}`",
                f"- reason: `{result['reason']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare local/public Task 1 detection data into YOLO format.")
    parser.add_argument("--public-download-root", default=str(TASK1_PUBLIC_DOWNLOADS_ROOT))
    parser.add_argument("--reports-root", default=str(REPORTS_ROOT))
    args = parser.parse_args(argv)

    reports_root = Path(args.reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    if not LOCAL_FRAMES_EXTRACT_DIR.exists():
        extract_zip_safe(PROJECT_ROOT / "frames.zip", LOCAL_FRAMES_EXTRACT_DIR)
    if not LOCAL_LABELS_EXTRACT_DIR.exists():
        extract_zip_safe(PROJECT_ROOT / "labels.zip", LOCAL_LABELS_EXTRACT_DIR)

    local_only = build_local_yolo_dataset(TASK1_YOLO_ROOT / "local_only")
    stage_unlabeled_local_videos(PROJECT_ROOT)
    quality = build_data_quality_report()

    public_root = Path(args.public_download_root)
    public_only = convert_visdrone_to_yolo(public_root, TASK1_YOLO_ROOT / "public_only")
    if public_only["status"] == "ready":
        create_offline_weather_augmented_subset(public_only["dataset_root"], split_name="train", fraction=0.15)

    combined = build_combined_yolo_dataset(
        TASK1_YOLO_ROOT / "public_only",
        TASK1_YOLO_ROOT / "local_only",
        TASK1_YOLO_ROOT / "combined",
        local_train_oversample_factor=2,
    )
    if combined["split_counts"]["train"] > 0:
        create_offline_weather_augmented_subset(combined["dataset_root"], split_name="train", fraction=0.15)

    (reports_root / "task1_class_mapping_report.md").write_text(render_class_mapping_report(), encoding="utf-8")
    (reports_root / "task1_data_quality_report.md").write_text(render_quality_markdown(quality), encoding="utf-8")
    public_ready = public_only["status"] == "ready"
    local_ready = local_only["split_counts"]["train"] > 0
    (reports_root / "task1_training_plan.md").write_text(
        render_training_plan_markdown(public_ready=public_ready, local_ready=local_ready),
        encoding="utf-8",
    )
    experiment_results = initial_experiment_results_payload(public_ready=public_ready, local_ready=local_ready)
    (reports_root / "task1_experiment_results.json").write_text(json.dumps(experiment_results, indent=2), encoding="utf-8")
    (reports_root / "task1_experiment_results.md").write_text(
        render_experiment_results_markdown(experiment_results),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "local_only": local_only,
                "public_only": public_only,
                "combined": combined,
                "quality": quality,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
