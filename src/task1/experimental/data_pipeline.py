from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
REPORTS_ROOT = PROJECT_ROOT / "reports"
TASK1_RAW_ROOT = DATA_ROOT / "task1_raw"
TASK1_PUBLIC_DOWNLOADS_ROOT = TASK1_RAW_ROOT / "public_downloads"
TASK1_LOCAL_EXTRACT_ROOT = TASK1_RAW_ROOT / "local_zip_extract"
TASK1_STAGING_ROOT = DATA_ROOT / "task1_staging"
TASK1_YOLO_ROOT = DATA_ROOT / "task1_yolo"
LOCAL_FRAMES_EXTRACT_DIR = TASK1_LOCAL_EXTRACT_ROOT / "frames"
LOCAL_LABELS_EXTRACT_DIR = TASK1_LOCAL_EXTRACT_ROOT / "labels"
TARGET_CLASS_NAMES = ["kara_tasiti", "insan", "uap_uai_ozel"]
INVENTORY_CLASS_NAMES = [
    "insan",
    "kara_tasiti",
    "deniz_tasiti",
    "hava_araci",
    "uap_uai_ozel",
    "ignore",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
LABEL_SUFFIXES = {".txt", ".json", ".xml", ".yaml", ".yml"}
LOCAL_CANONICAL_TO_TRAINING = {0: 0, 1: 1, 2: 2, 3: 2}
VISDRONE_TO_TRAINING: dict[int, int | None] = {
    0: None,
    1: 1,
    2: 1,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: None,
}
HIT_UAV_TO_TRAINING: dict[str, int | None] = {
    "person": 1,
    "car": 0,
    "othervehicle": 0,
    "other_vehicle": 0,
    "bicycle": 0,
    "dontcare": None,
}
SEADRONESEE_TO_INVENTORY: dict[str, str] = {
    "boat": "deniz_tasiti",
    "jetski": "deniz_tasiti",
    "lifesaving_appliance": "deniz_tasiti",
    "buoy": "deniz_tasiti",
    "swimmer": "insan",
}


@dataclass(slots=True)
class ZipInventory:
    path: str
    content_type: str
    label_format: str
    image_count: int
    label_count: int
    video_count: int
    approx_entries: int
    bad_member: str | None
    class_counts: dict[str, int]
    empty_label_files: int
    invalid_label_lines: int
    sample_entries: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_type": self.content_type,
            "label_format": self.label_format,
            "image_count": self.image_count,
            "label_count": self.label_count,
            "video_count": self.video_count,
            "approx_entries": self.approx_entries,
            "bad_member": self.bad_member,
            "class_counts": self.class_counts,
            "empty_label_files": self.empty_label_files,
            "invalid_label_lines": self.invalid_label_lines,
            "sample_entries": self.sample_entries,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ExtractionResult:
    archive_path: str
    destination: str
    extracted_file_count: int
    bad_member: str | None
    skipped_unsafe_members: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_path": self.archive_path,
            "destination": self.destination,
            "extracted_file_count": self.extracted_file_count,
            "bad_member": self.bad_member,
            "skipped_unsafe_members": self.skipped_unsafe_members,
        }


def collect_repo_file_inventory(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    search_plan = {
        "zip_files": [(project_root, {".zip"})],
        "images": [(project_root / "data", IMAGE_SUFFIXES)],
        "videos": [(project_root / "data", VIDEO_SUFFIXES)],
        "label_and_config_files": [(project_root / "data", LABEL_SUFFIXES)],
    }
    for key, root_specs in search_plan.items():
        matches: list[str] = []
        for root, suffixes in root_specs:
            if not root.exists():
                continue
            if root == project_root:
                paths = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
            else:
                paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
            matches.extend(str(path) for path in sorted(paths))
        payload[key] = matches
        payload[f"{key}_count"] = len(matches)
    return payload


def inspect_zip_archive(zip_path: str | Path) -> ZipInventory:
    archive_path = Path(zip_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = [item for item in archive.namelist() if not item.endswith("/")]
        suffix_counter = Counter(Path(name).suffix.lower() for name in names)
        image_count = sum(suffix_counter[suffix] for suffix in IMAGE_SUFFIXES)
        label_count = sum(suffix_counter[suffix] for suffix in LABEL_SUFFIXES)
        video_count = sum(suffix_counter[suffix] for suffix in VIDEO_SUFFIXES)
        content_type = _infer_zip_content_type(image_count, label_count, video_count, len(names))
        label_format = _infer_zip_label_format(archive, names)
        bad_member = archive.testzip()
        class_counts: Counter[str] = Counter()
        empty_label_files = 0
        invalid_label_lines = 0
        notes: list[str] = []
        if label_format == "yolo":
            for name in names:
                if Path(name).suffix.lower() != ".txt":
                    continue
                raw_text = archive.read(name).decode("utf-8", errors="ignore")
                stripped = raw_text.strip()
                if not stripped:
                    empty_label_files += 1
                    continue
                for line in raw_text.splitlines():
                    parsed = parse_yolo_label_line(line)
                    if parsed is None:
                        invalid_label_lines += 1
                        continue
                    class_counts[str(parsed["class_id"])] += 1
        if bad_member:
            notes.append(f"crc_failed:{bad_member}")
        return ZipInventory(
            path=str(archive_path),
            content_type=content_type,
            label_format=label_format,
            image_count=image_count,
            label_count=label_count,
            video_count=video_count,
            approx_entries=len(names),
            bad_member=bad_member,
            class_counts=dict(sorted(class_counts.items(), key=lambda item: int(item[0]))),
            empty_label_files=empty_label_files,
            invalid_label_lines=invalid_label_lines,
            sample_entries=names[:5],
            notes=notes,
        )


def extract_zip_safe(zip_path: str | Path, destination_dir: str | Path) -> ExtractionResult:
    archive_path = Path(zip_path)
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    skipped_unsafe: list[str] = []
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = destination / member.filename
            resolved = member_path.resolve()
            if not _is_within(destination.resolve(), resolved):
                skipped_unsafe.append(member.filename)
                continue
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, member_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1
    return ExtractionResult(
        archive_path=str(archive_path),
        destination=str(destination),
        extracted_file_count=extracted,
        bad_member=bad_member,
        skipped_unsafe_members=skipped_unsafe,
    )


def parse_yolo_label_line(line: str) -> dict[str, float | int] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError:
        return None
    return {
        "class_id": class_id,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }


def build_local_label_pair_summary(
    frames_dir: str | Path = LOCAL_FRAMES_EXTRACT_DIR,
    labels_dir: str | Path = LOCAL_LABELS_EXTRACT_DIR,
) -> dict[str, Any]:
    frames_root = Path(frames_dir)
    labels_root = Path(labels_dir)
    images = sorted(path for path in frames_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(path for path in labels_root.rglob("*.txt") if path.is_file())
    image_names = {path.stem for path in images}
    label_names = {path.stem for path in labels}
    missing_image = sorted(label_names - image_names)
    missing_label = sorted(image_names - label_names)
    frame_indices = sorted(_frame_index_from_stem(path.stem) for path in images)
    return {
        "image_count": len(images),
        "label_count": len(labels),
        "matched_pairs": len(image_names & label_names),
        "missing_image_count": len(missing_image),
        "missing_label_count": len(missing_label),
        "missing_images": missing_image[:10],
        "missing_labels": missing_label[:10],
        "frame_index_min": frame_indices[0] if frame_indices else None,
        "frame_index_max": frame_indices[-1] if frame_indices else None,
        "frame_stride": _infer_constant_stride(frame_indices),
    }


def split_local_dataset_by_frame_index(
    frames_dir: str | Path = LOCAL_FRAMES_EXTRACT_DIR,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[Path]]:
    frames_root = Path(frames_dir)
    images = sorted(path for path in frames_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        return {"train": [], "val": [], "test": []}
    ordered = sorted(images, key=lambda item: _frame_index_from_stem(item.stem))
    total = len(ordered)
    train_cutoff = max(1, int(math.floor(total * train_ratio)))
    val_cutoff = max(train_cutoff + 1, int(math.floor(total * (train_ratio + val_ratio))))
    return {
        "train": ordered[:train_cutoff],
        "val": ordered[train_cutoff:val_cutoff],
        "test": ordered[val_cutoff:],
    }


def build_local_yolo_dataset(
    output_root: str | Path,
    *,
    frames_dir: str | Path = LOCAL_FRAMES_EXTRACT_DIR,
    labels_dir: str | Path = LOCAL_LABELS_EXTRACT_DIR,
) -> dict[str, Any]:
    root = Path(output_root)
    _reset_dataset_root(root)
    split_map = split_local_dataset_by_frame_index(frames_dir)
    labels_lookup = {path.stem: path for path in Path(labels_dir).rglob("*.txt") if path.is_file()}
    split_counts: dict[str, int] = {}
    class_counts: Counter[str] = Counter()
    empty_labels = 0
    for split_name, image_paths in split_map.items():
        images_dir = root / "images" / split_name
        labels_out_dir = root / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_out_dir.mkdir(parents=True, exist_ok=True)
        split_counts[split_name] = 0
        for image_path in image_paths:
            label_path = labels_lookup.get(image_path.stem)
            if label_path is None:
                continue
            converted_lines: list[str] = []
            raw_text = label_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                empty_labels += 1
            for line in raw_text.splitlines():
                parsed = parse_yolo_label_line(line)
                if parsed is None:
                    continue
                mapped_class = LOCAL_CANONICAL_TO_TRAINING.get(int(parsed["class_id"]))
                if mapped_class is None:
                    continue
                class_counts[str(mapped_class)] += 1
                converted_lines.append(
                    f"{mapped_class} {parsed['x_center']:.6f} {parsed['y_center']:.6f} {parsed['width']:.6f} {parsed['height']:.6f}"
                )
            shutil.copy2(image_path, images_dir / image_path.name)
            (labels_out_dir / f"{image_path.stem}.txt").write_text("\n".join(converted_lines), encoding="utf-8")
            split_counts[split_name] += 1
    write_yolo_dataset_yaml(DATA_ROOT / "task1_local_only.yaml", root, TARGET_CLASS_NAMES)
    return {
        "dataset_root": str(root),
        "split_counts": split_counts,
        "class_counts": dict(sorted(class_counts.items(), key=lambda item: int(item[0]))),
        "empty_label_files": empty_labels,
    }


def convert_visdrone_to_yolo(download_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    source_root = Path(download_root)
    destination_root = Path(output_root)
    _reset_dataset_root(destination_root)
    split_config = {
        "train": _find_first_matching_dir(source_root, "VisDrone2019-DET-train"),
        "val": _find_first_matching_dir(source_root, "VisDrone2019-DET-val"),
    }
    notes: list[str] = []
    split_counts: dict[str, int] = {}
    class_counts: Counter[str] = Counter()
    for split_name, split_root in split_config.items():
        if split_root is None:
            notes.append(f"missing_split:{split_name}")
            split_counts[split_name] = 0
            continue
        images_dir = split_root / "images"
        annotations_dir = split_root / "annotations"
        if not images_dir.exists() or not annotations_dir.exists():
            notes.append(f"unsupported_visdrone_layout:{split_root}")
            split_counts[split_name] = 0
            continue
        split_counts[split_name] = _convert_visdrone_split(
            images_dir,
            annotations_dir,
            destination_root,
            split_name,
            class_counts,
        )
    (destination_root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (destination_root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    write_yolo_dataset_yaml(DATA_ROOT / "task1_public_only.yaml", destination_root, TARGET_CLASS_NAMES)
    return {
        "dataset_root": str(destination_root),
        "split_counts": split_counts,
        "class_counts": dict(sorted(class_counts.items(), key=lambda item: int(item[0]))),
        "notes": notes,
        "status": "ready" if split_counts.get("train", 0) and split_counts.get("val", 0) else "blocked",
    }


def build_combined_yolo_dataset(
    public_root: str | Path,
    local_root: str | Path,
    combined_root: str | Path,
    *,
    local_train_oversample_factor: int = 2,
) -> dict[str, Any]:
    public_dataset_root = Path(public_root)
    local_dataset_root = Path(local_root)
    destination = Path(combined_root)
    _reset_dataset_root(destination)
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for split_name in ("train", "val", "test"):
        for source_root, prefix in ((public_dataset_root, "public"), (local_dataset_root, "local")):
            images_dir = source_root / "images" / split_name
            labels_dir = source_root / "labels" / split_name
            if not images_dir.exists() or not labels_dir.exists():
                continue
            for image_path in sorted(images_dir.iterdir()):
                if not image_path.is_file():
                    continue
                target_image = destination / "images" / split_name / f"{prefix}_{image_path.name}"
                target_label = destination / "labels" / split_name / f"{prefix}_{image_path.stem}.txt"
                target_image.parent.mkdir(parents=True, exist_ok=True)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, target_image)
                shutil.copy2(labels_dir / f"{image_path.stem}.txt", target_label)
                split_counts[split_name] += 1
                if prefix == "local" and split_name == "train":
                    for duplicate_index in range(1, max(local_train_oversample_factor, 1)):
                        dup_image = destination / "images" / split_name / f"{prefix}_dup{duplicate_index}_{image_path.name}"
                        dup_label = destination / "labels" / split_name / f"{prefix}_dup{duplicate_index}_{image_path.stem}.txt"
                        shutil.copy2(image_path, dup_image)
                        shutil.copy2(labels_dir / f"{image_path.stem}.txt", dup_label)
                        split_counts[split_name] += 1
    write_yolo_dataset_yaml(DATA_ROOT / "task1_combined.yaml", destination, TARGET_CLASS_NAMES)
    return {"dataset_root": str(destination), "split_counts": split_counts}


def stage_unlabeled_local_videos(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root)
    staged_dir = TASK1_STAGING_ROOT / "unlabeled_local"
    staged_dir.mkdir(parents=True, exist_ok=True)
    videos = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and "task1_raw" not in str(path)
    ]
    staged_files: list[str] = []
    for video_path in videos:
        target = staged_dir / video_path.name
        if not target.exists():
            shutil.copy2(video_path, target)
        staged_files.append(str(target))
    return {"count": len(staged_files), "files": staged_files}


def create_offline_weather_augmented_subset(
    dataset_root: str | Path,
    *,
    split_name: str = "train",
    fraction: float = 0.15,
) -> dict[str, Any]:
    if cv2 is None:
        return {"status": "blocked", "reason": "missing_dependency:cv2", "generated": 0}
    root = Path(dataset_root)
    images_dir = root / "images" / split_name
    labels_dir = root / "labels" / split_name
    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file()) if images_dir.exists() else []
    if not image_paths:
        return {"status": "blocked", "reason": "no_train_images", "generated": 0}
    target_count = max(1, int(math.floor(len(image_paths) * fraction)))
    generated = 0
    generated_files: list[str] = []
    for index, image_path in enumerate(image_paths[:target_count]):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        aug_name = "night" if index % 3 == 0 else "fog" if index % 3 == 1 else "rain"
        augmented = _augment_image(image, aug_name)
        target_image = images_dir / f"{image_path.stem}__{aug_name}{image_path.suffix.lower()}"
        target_label = labels_dir / f"{image_path.stem}__{aug_name}.txt"
        cv2.imwrite(str(target_image), augmented)
        shutil.copy2(labels_dir / f"{image_path.stem}.txt", target_label)
        generated += 1
        generated_files.append(str(target_image))
    return {"status": "generated", "generated": generated, "files": generated_files[:10]}


def build_data_quality_report(
    frames_dir: str | Path = LOCAL_FRAMES_EXTRACT_DIR,
    labels_dir: str | Path = LOCAL_LABELS_EXTRACT_DIR,
) -> dict[str, Any]:
    images = sorted(path for path in Path(frames_dir).rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(path for path in Path(labels_dir).rglob("*.txt") if path.is_file())
    labels_lookup = {path.stem: path for path in labels}
    decode_failures = 0
    bbox_out_of_range = 0
    tiny_boxes = 0
    total_boxes = 0
    exact_hashes: Counter[str] = Counter()
    dhash_values: list[tuple[str, str]] = []
    for image_path in images:
        if cv2 is not None:
            image = cv2.imread(str(image_path))
            if image is None:
                decode_failures += 1
            else:
                exact_hashes[_md5_file(image_path)] += 1
                dhash = compute_dhash(image)
                if dhash is not None:
                    dhash_values.append((image_path.stem, dhash))
        label_path = labels_lookup.get(image_path.stem)
        if label_path is None:
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_yolo_label_line(line)
            if parsed is None:
                continue
            total_boxes += 1
            if not (0.0 <= float(parsed["x_center"]) <= 1.0 and 0.0 <= float(parsed["y_center"]) <= 1.0):
                bbox_out_of_range += 1
            if not (0.0 <= float(parsed["width"]) <= 1.0 and 0.0 <= float(parsed["height"]) <= 1.0):
                bbox_out_of_range += 1
            if float(parsed["width"]) * float(parsed["height"]) <= 0.001:
                tiny_boxes += 1
    empty_labels = sum(1 for path in labels if not path.read_text(encoding="utf-8").strip())
    duplicates = sum(count - 1 for count in exact_hashes.values() if count > 1)
    near_duplicates = _count_near_duplicates(dhash_values)
    return {
        "image_count": len(images),
        "label_count": len(labels),
        "decode_failures": decode_failures,
        "empty_label_files": empty_labels,
        "bbox_out_of_range": bbox_out_of_range,
        "total_boxes": total_boxes,
        "tiny_box_count": tiny_boxes,
        "tiny_box_ratio": round(tiny_boxes / max(total_boxes, 1), 6),
        "exact_duplicate_images": duplicates,
        "near_duplicate_images": near_duplicates,
        "missing_paired_labels": max(len(images) - len(labels), 0),
    }


def write_yolo_dataset_yaml(yaml_path: str | Path, dataset_root: str | Path, class_names: list[str]) -> None:
    path = Path(yaml_path)
    root = Path(dataset_root)
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"names: {json.dumps(class_names, ensure_ascii=False)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_class_mapping_report() -> str:
    return "\n".join(
        [
            "# Task 1 Class Mapping Report",
            "",
            "## Inventory Ontology",
            "",
            "- insan",
            "- kara_tasiti",
            "- deniz_tasiti",
            "- hava_araci",
            "- uap_uai_ozel",
            "- ignore",
            "",
            "## Training Ontology",
            "",
            "- 0 = kara_tasiti",
            "- 1 = insan",
            "- 2 = uap_uai_ozel",
            "",
            "## Local TEKNOFEST Mapping",
            "",
            "- local class 0 -> kara_tasiti",
            "- local class 1 -> insan",
            "- local class 2 -> uap_uai_ozel",
            "- local class 3 -> uap_uai_ozel",
            "",
            "## Public Dataset Decisions",
            "",
            "- VisDrone pedestrian/person/people -> insan",
            "- VisDrone vehicle-like classes -> kara_tasiti",
            "- UAVDT vehicle classes -> kara_tasiti",
            "- HIT-UAV person -> insan; car/bicycle/other vehicle -> kara_tasiti",
            "- SeaDronesSee boat/jetski/lifesaving appliance/buoy -> deniz_tasiti; swimmer -> insan",
            "",
            "## Notes",
            "",
            "- deniz_tasiti and hava_araci stay visible in inventory truth but are ignored in the first RGB training runs.",
            "- UAP/UAI public coverage is assumed absent unless local custom labels provide it.",
        ]
    )


def render_training_plan_markdown(public_ready: bool, local_ready: bool) -> str:
    experiment_1_status = "ready" if public_ready else "blocked_by_public_rgb_dataset"
    experiment_2_status = "ready" if public_ready and local_ready else "blocked_by_missing_public_or_local_dataset"
    experiment_3_status = "waiting_for_experiment_2_checkpoint"
    return "\n".join(
        [
            "# Task 1 Training Plan",
            "",
            "## Experiment 1",
            "",
            f"- status: `{experiment_1_status}`",
            "- scope: public-only baseline",
            "- model: `yolo11s`",
            "- imgsz: `960`",
            "- epochs: `40`",
            "- batch: `8` then fallback `4` or `2`",
            "- augment: mosaic `0.7`, scale `0.5`, translate `0.1`, fliplr `0.5`, copy_paste `0.15`, offline weather/night overlay on 15% of train images",
            "",
            "## Experiment 2",
            "",
            f"- status: `{experiment_2_status}`",
            "- scope: local labeled + public combined",
            "- model: `yolo11s`",
            "- imgsz: `960`",
            "- epochs: `60`",
            "- local train oversample: `2x`",
            "",
            "## Experiment 3",
            "",
            f"- status: `{experiment_3_status}`",
            "- scope: tiled inference probe on Experiment 2 best checkpoint",
            "- tile size: `960`",
            "- overlap: `0.25`",
        ]
    )


def initial_experiment_results_payload(*, public_ready: bool, local_ready: bool) -> dict[str, Any]:
    return {
        "decision": "EXPERIMENTAL ONLY",
        "experiments": {
            "experiment_1_public_only": {
                "status": "ready" if public_ready else "blocked",
                "reason": None if public_ready else "missing_public_rgb_dataset",
            },
            "experiment_2_combined": {
                "status": "ready" if public_ready and local_ready else "blocked",
                "reason": None if public_ready and local_ready else "missing_public_or_local_dataset",
            },
            "experiment_3_tiled_probe": {
                "status": "blocked",
                "reason": "missing_experiment_2_checkpoint",
            },
        },
    }


def _convert_visdrone_split(
    images_dir: Path,
    annotations_dir: Path,
    destination_root: Path,
    split_name: str,
    class_counts: Counter[str],
) -> int:
    converted = 0
    dest_images = destination_root / "images" / split_name
    dest_labels = destination_root / "labels" / split_name
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            continue
        image = cv2.imread(str(image_path)) if cv2 is not None else None
        image_height = int(image.shape[0]) if image is not None else 1
        image_width = int(image.shape[1]) if image is not None else 1
        converted_lines: list[str] = []
        for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
            parts = [item.strip() for item in raw_line.split(",")]
            if len(parts) < 6:
                continue
            try:
                left = float(parts[0])
                top = float(parts[1])
                width = float(parts[2])
                height = float(parts[3])
                category_id = int(parts[5])
            except ValueError:
                continue
            mapped_class = VISDRONE_TO_TRAINING.get(category_id)
            if mapped_class is None or width <= 0.0 or height <= 0.0:
                continue
            x_center = (left + width / 2.0) / max(image_width, 1)
            y_center = (top + height / 2.0) / max(image_height, 1)
            normalized_width = width / max(image_width, 1)
            normalized_height = height / max(image_height, 1)
            converted_lines.append(
                f"{mapped_class} {x_center:.6f} {y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}"
            )
            class_counts[str(mapped_class)] += 1
        shutil.copy2(image_path, dest_images / image_path.name)
        (dest_labels / f"{image_path.stem}.txt").write_text("\n".join(converted_lines), encoding="utf-8")
        converted += 1
    return converted


def _infer_zip_content_type(image_count: int, label_count: int, video_count: int, total_entries: int) -> str:
    if video_count and not image_count and not label_count:
        return "video"
    if image_count and label_count and image_count + label_count == total_entries:
        return "image+label"
    if image_count and not label_count and not video_count:
        return "image-only"
    return "mixed"


def _infer_zip_label_format(archive: zipfile.ZipFile, names: list[str]) -> str:
    txt_entries = [name for name in names if Path(name).suffix.lower() == ".txt"]
    json_entries = [name for name in names if Path(name).suffix.lower() == ".json"]
    xml_entries = [name for name in names if Path(name).suffix.lower() == ".xml"]
    if txt_entries:
        for name in txt_entries[:10]:
            lines = archive.read(name).decode("utf-8", errors="ignore").splitlines()
            for line in lines:
                if parse_yolo_label_line(line) is not None:
                    return "yolo"
        return "unknown"
    if json_entries:
        try:
            payload = json.loads(archive.read(json_entries[0]).decode("utf-8", errors="ignore"))
            if isinstance(payload, dict) and {"images", "annotations"} <= set(payload.keys()):
                return "coco"
        except Exception:
            return "unknown"
        return "json"
    if xml_entries:
        return "voc"
    return "none"


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _frame_index_from_stem(stem: str) -> int:
    digits = "".join(character for character in stem if character.isdigit())
    return int(digits) if digits else -1


def _infer_constant_stride(values: list[int]) -> int | None:
    if len(values) < 3:
        return None
    deltas = [current - previous for previous, current in zip(values, values[1:])]
    return deltas[0] if deltas and all(delta == deltas[0] for delta in deltas) else None


def _reset_dataset_root(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)


def _find_first_matching_dir(root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        if path.is_dir():
            return path
    return None


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_dhash(image: Any) -> str | None:
    if cv2 is None or image is None:
        return None
    resized = cv2.resize(image, (9, 8))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    bits = []
    for row in gray:
        for left, right in zip(row, row[1:]):
            bits.append("1" if int(left) > int(right) else "0")
    return f"{int(''.join(bits), 2):016x}"


def _count_near_duplicates(values: list[tuple[str, str]], *, max_distance: int = 5) -> int:
    duplicates = 0
    for index, (_, left) in enumerate(values):
        left_int = int(left, 16)
        for _, right in values[index + 1 :]:
            distance = (left_int ^ int(right, 16)).bit_count()
            if distance <= max_distance:
                duplicates += 1
    return duplicates


def _augment_image(image: Any, mode: str) -> Any:
    if mode == "night":
        dark = cv2.convertScaleAbs(image, alpha=0.55, beta=-18)
        return cv2.GaussianBlur(dark, (3, 3), 0)
    if mode == "fog":
        overlay = image.copy()
        fog = cv2.addWeighted(overlay, 0.65, 255 * (overlay * 0 + 1).astype(overlay.dtype), 0.35, 0)
        return cv2.GaussianBlur(fog, (9, 9), 0)
    rain = image.copy()
    height, width = rain.shape[:2]
    for step in range(0, width, 120):
        cv2.line(rain, (step, 0), (min(step + 45, width - 1), min(height - 1, 120)), (170, 170, 170), 1)
    rain = cv2.convertScaleAbs(rain, alpha=0.85, beta=-8)
    return cv2.GaussianBlur(rain, (3, 3), 0)
