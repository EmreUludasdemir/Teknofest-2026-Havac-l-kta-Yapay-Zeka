from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.experimental import (
    LOCAL_FRAMES_EXTRACT_DIR,
    LOCAL_LABELS_EXTRACT_DIR,
    REPORTS_ROOT,
    TASK1_PUBLIC_DOWNLOADS_ROOT,
    build_local_label_pair_summary,
    collect_repo_file_inventory,
    extract_zip_safe,
    inspect_zip_archive,
)


def render_inventory_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Task 1 Dataset Inventory",
        "",
        "## Repo File Inventory",
        "",
        f"- zip count: `{payload['repo_inventory']['zip_files_count']}`",
        f"- image count: `{payload['repo_inventory']['images_count']}`",
        f"- video count: `{payload['repo_inventory']['videos_count']}`",
        f"- label/config count: `{payload['repo_inventory']['label_and_config_files_count']}`",
        "",
        "## Zip Truth",
        "",
    ]
    for item in payload["zip_inventories"]:
        lines.extend(
            [
                f"### `{Path(item['path']).name}`",
                "",
                f"- content type: `{item['content_type']}`",
                f"- label format: `{item['label_format']}`",
                f"- image count: `{item['image_count']}`",
                f"- label count: `{item['label_count']}`",
                f"- video count: `{item['video_count']}`",
                f"- bad member: `{item['bad_member']}`",
                f"- class counts: `{item['class_counts']}`",
                f"- empty label files: `{item['empty_label_files']}`",
                f"- invalid label lines: `{item['invalid_label_lines']}`",
                "",
            ]
        )
    pair = payload.get("local_pair_summary")
    if pair:
        lines.extend(
            [
                "## Local Paired Truth",
                "",
                f"- image count: `{pair['image_count']}`",
                f"- label count: `{pair['label_count']}`",
                f"- matched pairs: `{pair['matched_pairs']}`",
                f"- missing image count: `{pair['missing_image_count']}`",
                f"- missing label count: `{pair['missing_label_count']}`",
                f"- frame index range: `{pair['frame_index_min']}..{pair['frame_index_max']}`",
                f"- inferred frame stride: `{pair['frame_stride']}`",
                "",
            ]
        )
    public_status = payload.get("public_download_status", {})
    datasets = public_status.get("datasets", [])
    if datasets:
        lines.extend(["## Public Download Status", ""])
        for item in datasets:
            lines.extend(
                [
                    f"### `{item['name']}`",
                    "",
                    f"- status: `{item['status']}`",
                    f"- modality: `{item['modality']}`",
                    f"- similarity: `{item['similarity']}`",
                    f"- classes: `{item['classes']}`",
                    f"- notes: `{item['notes']}`",
                    "",
                ]
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory local Task 1 zip/image/video assets")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-json", default=str(REPORTS_ROOT / "task1_dataset_inventory.json"))
    parser.add_argument("--output-md", default=str(REPORTS_ROOT / "task1_dataset_inventory.md"))
    parser.add_argument("--extract", action="store_true", help="Extract frames.zip and labels.zip safely.")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root)
    repo_inventory = collect_repo_file_inventory(project_root)
    zip_paths = [Path(path) for path in repo_inventory["zip_files"]]
    zip_inventories = [inspect_zip_archive(path).to_dict() for path in zip_paths]

    extraction_results: list[dict[str, object]] = []
    frames_zip = next((path for path in zip_paths if path.name.lower() == "frames.zip"), None)
    labels_zip = next((path for path in zip_paths if path.name.lower() == "labels.zip"), None)
    local_pair_summary = None
    if args.extract and frames_zip is not None and labels_zip is not None:
        extraction_results.append(extract_zip_safe(frames_zip, LOCAL_FRAMES_EXTRACT_DIR).to_dict())
        extraction_results.append(extract_zip_safe(labels_zip, LOCAL_LABELS_EXTRACT_DIR).to_dict())
        local_pair_summary = build_local_label_pair_summary()

    payload = {
        "project_root": str(project_root),
        "repo_inventory": repo_inventory,
        "zip_inventories": zip_inventories,
        "extraction_results": extraction_results,
        "local_pair_summary": local_pair_summary,
        "public_download_status": json.loads((TASK1_PUBLIC_DOWNLOADS_ROOT / "download_status.json").read_text(encoding="utf-8"))
        if (TASK1_PUBLIC_DOWNLOADS_ROOT / "download_status.json").exists()
        else {},
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_inventory_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
