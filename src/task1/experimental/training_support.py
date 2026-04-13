from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from src.task1.experimental.data_pipeline import TARGET_CLASS_NAMES, parse_yolo_label_line


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

TRAINING_EXPERIMENT_DEFAULTS: dict[str, dict[str, Any]] = {
    "experiment_1_public_only": {
        "yaml_path": DATA_ROOT / "task1_public_only.yaml",
        "model": "yolo11s.pt",
        "epochs": 40,
        "batch": 8,
        "imgsz": 960,
        "patience": 15,
    },
    "experiment_2_combined": {
        "yaml_path": DATA_ROOT / "task1_combined.yaml",
        "model": "yolo11s.pt",
        "epochs": 60,
        "batch": 8,
        "imgsz": 960,
        "patience": 15,
    },
    "combined_yolo11s_full": {
        "yaml_path": DATA_ROOT / "task1_combined.yaml",
        "model": "yolo11s.pt",
        "epochs": 60,
        "batch": 8,
        "imgsz": 960,
        "patience": 15,
    },
    "combined_yolo11m_full": {
        "yaml_path": DATA_ROOT / "task1_combined.yaml",
        "model": "yolo11m.pt",
        "epochs": 60,
        "batch": 4,
        "imgsz": 960,
        "patience": 12,
    },
}


def get_training_experiment_defaults(name: str) -> dict[str, Any]:
    try:
        defaults = TRAINING_EXPERIMENT_DEFAULTS[name]
    except KeyError as exc:  # pragma: no cover - trivial guard
        raise ValueError(f"unsupported_experiment:{name}") from exc
    return dict(defaults)


def load_dataset_config(dataset_yaml_path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(dataset_yaml_path).read_text(encoding="utf-8"))


def resolve_split_label_dir(dataset_yaml_path: str | Path, split: str) -> Path:
    dataset_yaml = load_dataset_config(dataset_yaml_path)
    dataset_root = Path(dataset_yaml["path"])
    split_ref = dataset_yaml.get(split)
    if split_ref is None:
        raise KeyError(f"missing_split:{split}")
    split_path = Path(str(split_ref))
    if split_path.parts and split_path.parts[0] == "labels":
        label_relative = split_path
    elif split_path.parts and split_path.parts[0] == "images":
        label_relative = Path("labels").joinpath(*split_path.parts[1:])
    else:
        label_relative = Path("labels") / split
    return dataset_root / label_relative


def collect_split_class_support(dataset_yaml_path: str | Path, split: str) -> dict[str, Any]:
    label_dir = resolve_split_label_dir(dataset_yaml_path, split)
    class_counts: dict[str, int] = {}
    file_count = 0
    empty_label_files = 0
    if not label_dir.exists():
        return {
            "split": split,
            "label_dir": str(label_dir),
            "file_count": 0,
            "empty_label_files": 0,
            "class_counts": {},
        }
    for label_path in sorted(label_dir.rglob("*.txt")):
        file_count += 1
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_label_files += 1
            continue
        for line in text.splitlines():
            parsed = parse_yolo_label_line(line)
            if parsed is None:
                continue
            class_key = str(int(parsed["class_id"]))
            class_counts[class_key] = class_counts.get(class_key, 0) + 1
    return {
        "split": split,
        "label_dir": str(label_dir),
        "file_count": file_count,
        "empty_label_files": empty_label_files,
        "class_counts": class_counts,
    }


def extract_detection_metrics(
    results: Any,
    *,
    class_names: list[str] | None = None,
    split_support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = class_names or list(TARGET_CLASS_NAMES)
    results_dict = {
        key: float(value)
        for key, value in getattr(results, "results_dict", {}).items()
        if isinstance(value, (int, float))
    }
    box = getattr(results, "box", None)
    ap_class_index = [int(item) for item in getattr(box, "ap_class_index", [])] if box is not None else []
    ap_lookup = {class_id: offset for offset, class_id in enumerate(ap_class_index)}
    ap_matrix = getattr(box, "all_ap", None)
    maps = [float(item) for item in getattr(box, "maps", [])] if box is not None else []
    precisions = [float(item) for item in getattr(box, "p", [])] if box is not None else []
    recalls = [float(item) for item in getattr(box, "r", [])] if box is not None else []
    supports = (split_support or {}).get("class_counts", {})
    per_class: dict[str, Any] = {}
    missing_from_split: list[str] = []
    for class_id, class_name in enumerate(names):
        support_count = int(supports.get(str(class_id), 0))
        if support_count <= 0:
            per_class[class_name] = {
                "class_id": class_id,
                "support": 0,
                "status": "no_ground_truth_in_split",
                "ap50": None,
                "ap50_95": None,
                "precision": None,
                "recall": None,
            }
            missing_from_split.append(class_name)
            continue
        if class_id not in ap_lookup:
            per_class[class_name] = {
                "class_id": class_id,
                "support": support_count,
                "status": "ground_truth_present_missing_eval_output",
                "ap50": None,
                "ap50_95": None,
                "precision": None,
                "recall": None,
            }
            continue
        offset = ap_lookup[class_id]
        ap50 = None
        if ap_matrix is not None and len(ap_matrix) > offset and len(ap_matrix[offset]) > 0:
            ap50 = float(ap_matrix[offset][0])
        ap50_95 = float(maps[class_id]) if class_id < len(maps) else None
        per_class[class_name] = {
            "class_id": class_id,
            "support": support_count,
            "status": "scored",
            "ap50": ap50,
            "ap50_95": ap50_95,
            "precision": float(precisions[offset]) if offset < len(precisions) else None,
            "recall": float(recalls[offset]) if offset < len(recalls) else None,
        }
    return {
        "overall": {
            "precision": results_dict.get("metrics/precision(B)"),
            "recall": results_dict.get("metrics/recall(B)"),
            "mAP50": results_dict.get("metrics/mAP50(B)"),
            "mAP50_95": results_dict.get("metrics/mAP50-95(B)"),
            "fitness": results_dict.get("fitness"),
        },
        "per_class": per_class,
        "support": split_support or {},
        "missing_from_split": missing_from_split,
    }


def load_training_history(results_csv_path: str | Path) -> list[dict[str, float]]:
    csv_path = Path(results_csv_path)
    if not csv_path.exists():
        return []
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            rows.append(parsed)
    return rows


def summarize_learning_dynamics(history: list[dict[str, float]]) -> dict[str, Any]:
    if not history:
        return {
            "status": "unavailable",
            "signals": ["no_history"],
            "best_epoch": None,
            "best_val_mAP50_95": None,
            "final_val_mAP50_95": None,
        }
    metric_key = "metrics/mAP50-95(B)"
    train_loss_keys = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    metric_series = [row.get(metric_key, 0.0) for row in history]
    loss_series = [sum(row.get(key, 0.0) for key in train_loss_keys) for row in history]
    best_index = max(range(len(metric_series)), key=lambda index: metric_series[index])
    final_index = len(metric_series) - 1
    signals: list[str] = []
    if len(metric_series) >= 10:
        trailing_gain = metric_series[-1] - metric_series[-10]
        if best_index <= len(metric_series) - 5 and metric_series[best_index] - metric_series[-1] >= 0.01 and loss_series[-1] < loss_series[max(0, len(loss_series) - 10)]:
            signals.append("overfit_risk")
        if best_index == final_index and trailing_gain >= 0.01:
            signals.append("underfit_or_budget_left")
    status = "stable"
    if "overfit_risk" in signals:
        status = "watch_overfit"
    elif "underfit_or_budget_left" in signals:
        status = "still_improving"
    return {
        "status": status,
        "signals": signals,
        "best_epoch": int(history[best_index].get("epoch", best_index + 1)),
        "best_val_mAP50_95": metric_series[best_index],
        "final_val_mAP50_95": metric_series[-1],
    }


def render_experiment_results_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Task 1 Experiment Results", "", f"**Decision:** `{payload.get('decision', 'EXPERIMENTAL ONLY')}`", ""]
    for name, result in payload.get("experiments", {}).items():
        lines.extend(
            [
                f"## `{name}`",
                "",
                f"- status: `{result.get('status')}`",
                f"- reason: `{result.get('reason')}`",
                f"- model: `{result.get('model')}`",
                f"- yaml: `{result.get('yaml_path')}`",
                f"- best weights: `{result.get('best_weights_path')}`",
                f"- batch used: `{result.get('effective_batch')}`",
                f"- training metrics: `{result.get('training_metrics')}`",
                f"- holdout metrics: `{result.get('holdout_metrics')}`",
                f"- learning dynamics: `{result.get('learning_dynamics')}`",
                "",
            ]
        )
    return "\n".join(lines)
