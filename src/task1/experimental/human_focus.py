from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from src.task1.experimental.data_pipeline import IMAGE_SUFFIXES, parse_yolo_label_line, write_yolo_dataset_yaml


def _bbox_area(parsed: dict[str, float | int]) -> float:
    return float(parsed["width"]) * float(parsed["height"])


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "tiny_ratio_le_0_001": None}
    ordered = sorted(values)
    count = len(ordered)

    def pick(p: float) -> float:
        return ordered[min(count - 1, max(0, int((count - 1) * p)))]

    tiny_count = sum(1 for value in ordered if value <= 0.001)
    return {
        "count": count,
        "p10": pick(0.10),
        "p25": pick(0.25),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "tiny_ratio_le_0_001": tiny_count / count,
    }


def collect_human_focus_audit(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    result: dict[str, Any] = {
        "split_class_counts": {},
        "split_source_class_counts": {},
        "split_area_stats": {},
        "human_source_stats": {},
        "human_image_stats": {},
        "diagnosis": [],
    }
    for split in ("train", "val", "test"):
        label_dir = root / "labels" / split
        split_counts: Counter[str] = Counter()
        split_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
        split_area_values: dict[str, list[float]] = {"0": [], "1": [], "2": []}
        human_source_values: dict[str, list[float]] = defaultdict(list)
        human_file_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "human_files": 0, "small_human_files": 0, "human_boxes": 0})
        for label_path in sorted(label_dir.rglob("*.txt")):
            source = "local" if label_path.stem.startswith("local_") else "public" if label_path.stem.startswith("public_") else "other"
            human_file_counts[source]["files"] += 1
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            has_human = False
            has_small_human = False
            for line in text.splitlines():
                parsed = parse_yolo_label_line(line)
                if parsed is None:
                    continue
                class_id = str(int(parsed["class_id"]))
                area = _bbox_area(parsed)
                split_counts[class_id] += 1
                split_source_counts[source][class_id] += 1
                split_area_values.setdefault(class_id, []).append(area)
                if class_id == "1":
                    has_human = True
                    human_file_counts[source]["human_boxes"] += 1
                    human_source_values[source].append(area)
                    if area <= 0.001:
                        has_small_human = True
            if has_human:
                human_file_counts[source]["human_files"] += 1
            if has_small_human:
                human_file_counts[source]["small_human_files"] += 1
        result["split_class_counts"][split] = dict(split_counts)
        result["split_source_class_counts"][split] = {key: dict(value) for key, value in split_source_counts.items()}
        result["split_area_stats"][split] = {class_id: _quantiles(values) for class_id, values in split_area_values.items()}
        for source, values in human_source_values.items():
            result["human_source_stats"][f"{split}_{source}"] = _quantiles(values)
        for source, stats in human_file_counts.items():
            result["human_image_stats"][f"{split}_{source}"] = stats

    train_humans = result["split_class_counts"].get("train", {}).get("1", 0)
    train_local_humans = result["split_source_class_counts"].get("train", {}).get("local", {}).get("1", 0)
    test_humans = result["split_class_counts"].get("test", {}).get("1", 0)
    test_local_humans = result["split_source_class_counts"].get("test", {}).get("local", {}).get("1", 0)
    if train_humans:
        local_share = train_local_humans / train_humans
        if local_share < 0.05:
            result["diagnosis"].append(
                f"train_local_human_share_low:{local_share:.4f}"
            )
    if test_humans and test_local_humans == test_humans:
        result["diagnosis"].append("test_humans_are_local_only")
    test_human_stats = result["split_area_stats"].get("test", {}).get("1", {})
    if test_human_stats.get("tiny_ratio_le_0_001") == 1.0:
        result["diagnosis"].append("all_holdout_humans_are_tiny")
    result["diagnosis"].append("occlusion_not_directly_measurable_from_bbox_labels")
    return result


def _augment_image(image: Any, mode: str) -> Any:
    if mode == "night":
        return cv2.convertScaleAbs(image, alpha=0.62, beta=-14)
    if mode == "blur":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if mode == "lowres":
        height, width = image.shape[:2]
        down = cv2.resize(image, (max(1, width // 2), max(1, height // 2)), interpolation=cv2.INTER_AREA)
        return cv2.resize(down, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def build_human_focus_dataset(
    base_dataset_root: str | Path,
    output_root: str | Path,
    yaml_path: str | Path,
    *,
    class_names: list[str],
    local_human_repeat_factor: int = 6,
    tiny_human_threshold: float = 0.001,
    tiny_human_augmentations: tuple[str, ...] = ("night", "blur", "lowres"),
) -> dict[str, Any]:
    base_root = Path(base_dataset_root)
    destination = Path(output_root)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(base_root, destination)
    added_repeats = 0
    added_augmentations = 0
    skipped_augmentations = 0
    train_images_dir = destination / "images" / "train"
    train_labels_dir = destination / "labels" / "train"
    for label_path in sorted(train_labels_dir.glob("local_*.txt")):
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parsed_rows = [parse_yolo_label_line(line) for line in text.splitlines()]
        parsed_rows = [row for row in parsed_rows if row is not None]
        human_rows = [row for row in parsed_rows if int(row["class_id"]) == 1]
        if not human_rows:
            continue
        image_path = next(
            (train_images_dir / f"{label_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES if (train_images_dir / f"{label_path.stem}{suffix}").exists()),
            None,
        )
        if image_path is None:
            continue
        for repeat_index in range(1, max(local_human_repeat_factor, 1)):
            duplicate_image = train_images_dir / f"{label_path.stem}__humanrep{repeat_index}{image_path.suffix.lower()}"
            duplicate_label = train_labels_dir / f"{label_path.stem}__humanrep{repeat_index}.txt"
            shutil.copy2(image_path, duplicate_image)
            shutil.copy2(label_path, duplicate_label)
            added_repeats += 1
        if cv2 is None:
            skipped_augmentations += len(tiny_human_augmentations)
            continue
        if any(_bbox_area(row) <= tiny_human_threshold for row in human_rows):
            image = cv2.imread(str(image_path))
            if image is None:
                skipped_augmentations += len(tiny_human_augmentations)
                continue
            for augmentation in tiny_human_augmentations:
                augmented = _augment_image(image, augmentation)
                aug_image = train_images_dir / f"{label_path.stem}__{augmentation}{image_path.suffix.lower()}"
                aug_label = train_labels_dir / f"{label_path.stem}__{augmentation}.txt"
                cv2.imwrite(str(aug_image), augmented)
                shutil.copy2(label_path, aug_label)
                added_augmentations += 1
    write_yolo_dataset_yaml(yaml_path, destination, class_names)
    return {
        "dataset_root": str(destination),
        "yaml_path": str(Path(yaml_path)),
        "local_human_repeat_factor": local_human_repeat_factor,
        "tiny_human_threshold": tiny_human_threshold,
        "tiny_human_augmentations": list(tiny_human_augmentations),
        "added_repeat_images": added_repeats,
        "added_augmented_images": added_augmentations,
        "skipped_augmentations": skipped_augmentations,
    }


def render_human_focus_audit_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Task 1 Human Focus Audit",
            "",
            "## Diagnosis",
            "",
            *[f"- {item}" for item in payload.get("diagnosis", [])],
            "",
            "## Train/Test Human Source Truth",
            "",
            f"- train local human boxes: `{payload['split_source_class_counts'].get('train', {}).get('local', {}).get('1', 0)}`",
            f"- train public human boxes: `{payload['split_source_class_counts'].get('train', {}).get('public', {}).get('1', 0)}`",
            f"- test local human boxes: `{payload['split_source_class_counts'].get('test', {}).get('local', {}).get('1', 0)}`",
            "",
            "## Human Size Summary",
            "",
            f"- train local insan: `{json.dumps(payload['human_source_stats'].get('train_local', {}), ensure_ascii=False)}`",
            f"- train public insan: `{json.dumps(payload['human_source_stats'].get('train_public', {}), ensure_ascii=False)}`",
            f"- test local insan: `{json.dumps(payload['human_source_stats'].get('test_local', {}), ensure_ascii=False)}`",
        ]
    )
