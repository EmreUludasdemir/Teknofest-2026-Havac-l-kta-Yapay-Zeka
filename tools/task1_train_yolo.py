from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.experimental import (
    TARGET_CLASS_NAMES,
    collect_split_class_support,
    extract_detection_metrics,
    get_training_experiment_defaults,
    load_training_history,
    render_experiment_results_markdown,
    summarize_learning_dynamics,
)

REPORTS_ROOT = PROJECT_ROOT / "reports"
LOGS_ROOT = PROJECT_ROOT / "_logs" / "task1_training"


def _load_ultralytics() -> Any:
    from ultralytics import YOLO  # type: ignore[import-not-found]

    return YOLO


def _load_torch() -> Any | None:
    try:  # pragma: no cover - optional dependency guard
        import torch
    except Exception:
        return None
    return torch


def _load_results_payload() -> dict[str, Any]:
    path = REPORTS_ROOT / "task1_experiment_results.json"
    if not path.exists():
        return {"decision": "EXPERIMENTAL ONLY", "experiments": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_results_payload(payload: dict[str, Any]) -> None:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_ROOT / "task1_experiment_results.json"
    md_path = REPORTS_ROOT / "task1_experiment_results.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_experiment_results_markdown(payload), encoding="utf-8")


def _device_index_from_arg(device_arg: str) -> int | None:
    if device_arg.isdigit():
        return int(device_arg)
    return None


def _torch_device_from_arg(torch_module: Any, device_arg: str) -> Any | None:
    device_index = _device_index_from_arg(device_arg)
    if device_index is None:
        return None
    return torch_module.device("cuda", device_index)


def _capture_peak_vram_mb(torch_module: Any | None, device_arg: str) -> float | None:
    if torch_module is None or not getattr(torch_module, "cuda", None) or not torch_module.cuda.is_available():
        return None
    torch_device = _torch_device_from_arg(torch_module, device_arg)
    if torch_device is None:
        return None
    torch_module.cuda.set_device(torch_device)
    allocated = float(torch_module.cuda.max_memory_allocated(torch_device)) / (1024 * 1024)
    reserved = float(torch_module.cuda.max_memory_reserved(torch_device)) / (1024 * 1024)
    return max(allocated, reserved)


def _reset_peak_vram(torch_module: Any | None, device_arg: str) -> None:
    if torch_module is None or not getattr(torch_module, "cuda", None) or not torch_module.cuda.is_available():
        return
    torch_device = _torch_device_from_arg(torch_module, device_arg)
    if torch_device is None:
        return
    torch_module.cuda.set_device(torch_device)
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats(torch_device)


def _run_eval(
    weights_path: Path,
    *,
    dataset_yaml_path: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    YOLO = _load_ultralytics()
    model = YOLO(str(weights_path))
    results = model.val(
        data=str(dataset_yaml_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )
    support = collect_split_class_support(dataset_yaml_path, split)
    payload = extract_detection_metrics(results, class_names=list(TARGET_CLASS_NAMES), split_support=support)
    payload["split"] = split
    return payload


def _train_once(
    *,
    model_name: str,
    yaml_path: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    patience: int,
    device: str,
    workers: int,
    run_name: str,
) -> tuple[Any, Path | None, float | None]:
    YOLO = _load_ultralytics()
    torch_module = _load_torch()
    _reset_peak_vram(torch_module, device)
    model = YOLO(model_name)
    train_results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        device=device,
        workers=workers,
        project=str(LOGS_ROOT),
        name=run_name,
        exist_ok=True,
        mosaic=0.7,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        copy_paste=0.15,
    )
    save_dir = Path(train_results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    peak_vram_mb = _capture_peak_vram_mb(torch_module, device)
    return train_results, (best_weights if best_weights.exists() else None), peak_vram_mb


def _batch_fallback_sequence(default_batch: int, explicit_batch: int | None) -> list[int]:
    if explicit_batch is not None:
        return [explicit_batch]
    fallbacks = [default_batch]
    if default_batch > 4:
        fallbacks.append(4)
    if default_batch > 2:
        fallbacks.append(2)
    deduped: list[int] = []
    for item in fallbacks:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _train_with_batch_fallback(
    *,
    model_name: str,
    yaml_path: Path,
    epochs: int,
    default_batch: int,
    explicit_batch: int | None,
    imgsz: int,
    patience: int,
    device: str,
    workers: int,
    run_name: str,
) -> tuple[Any, Path | None, int, float | None]:
    attempted_batches = _batch_fallback_sequence(default_batch, explicit_batch)
    last_exc: Exception | None = None
    for batch in attempted_batches:
        try:
            train_results, best_weights, peak_vram_mb = _train_once(
                model_name=model_name,
                yaml_path=yaml_path,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                patience=patience,
                device=device,
                workers=workers,
                run_name=run_name,
            )
            return train_results, best_weights, batch, peak_vram_mb
        except RuntimeError as exc:
            last_exc = exc
            if "out of memory" not in str(exc).lower():
                raise
            torch_module = _load_torch()
            if torch_module is not None and getattr(torch_module, "cuda", None):
                torch_module.cuda.empty_cache()
            gc.collect()
    assert last_exc is not None  # noqa: S101
    raise last_exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Task 1 YOLO training experiments.")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "experiment_1_public_only",
            "experiment_2_combined",
            "combined_yolo11s_full",
            "combined_yolo11m_full",
        ],
    )
    parser.add_argument("--result-key")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--model")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run-name")
    parser.add_argument("--patience", type=int)
    args = parser.parse_args(argv)

    defaults = get_training_experiment_defaults(args.experiment)
    yaml_path = Path(defaults["yaml_path"])
    result_payload = _load_results_payload()
    result_key = args.result_key or args.experiment
    result_entry: dict[str, Any] = {
        "status": "blocked",
        "reason": None,
        "yaml_path": str(yaml_path),
        "model": args.model or defaults["model"],
        "requested_epochs": int(args.epochs or defaults["epochs"]),
        "requested_batch": int(args.batch or defaults["batch"]),
        "effective_batch": None,
        "best_weights_path": None,
        "training_metrics": None,
        "holdout_metrics": None,
        "learning_dynamics": None,
        "peak_vram_mb": None,
    }
    if not yaml_path.exists():
        result_entry["reason"] = "missing_yaml"
        result_payload["experiments"][result_key] = result_entry
        _write_results_payload(result_payload)
        print(json.dumps(result_entry, indent=2))
        return 0
    try:
        _load_ultralytics()
    except Exception as exc:
        result_entry["reason"] = f"missing_dependency:{exc.__class__.__name__}"
        result_payload["experiments"][result_key] = result_entry
        _write_results_payload(result_payload)
        print(json.dumps(result_entry, indent=2))
        return 0

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or result_key
    try:
        train_results, best_weights, effective_batch, peak_vram_mb = _train_with_batch_fallback(
            model_name=args.model or defaults["model"],
            yaml_path=yaml_path,
            epochs=int(args.epochs or defaults["epochs"]),
            default_batch=int(defaults["batch"]),
            explicit_batch=args.batch,
            imgsz=int(args.imgsz or defaults["imgsz"]),
            patience=int(args.patience or defaults["patience"]),
            device=args.device,
            workers=args.workers,
            run_name=run_name,
        )
    except RuntimeError as exc:
        result_entry["reason"] = f"training_failed:{exc.__class__.__name__}"
        result_entry["status"] = "failed"
        result_payload["experiments"][result_key] = result_entry
        _write_results_payload(result_payload)
        print(json.dumps(result_entry, indent=2))
        return 0

    save_dir = Path(train_results.save_dir)
    results_csv = save_dir / "results.csv"
    training_metrics = _run_eval(
        best_weights or save_dir / "weights" / "last.pt",
        dataset_yaml_path=yaml_path,
        split="val",
        imgsz=int(args.imgsz or defaults["imgsz"]),
        batch=effective_batch,
        device=args.device,
    )
    holdout_metrics = _run_eval(
        best_weights or save_dir / "weights" / "last.pt",
        dataset_yaml_path=yaml_path,
        split="test",
        imgsz=int(args.imgsz or defaults["imgsz"]),
        batch=effective_batch,
        device=args.device,
    )
    learning_dynamics = summarize_learning_dynamics(load_training_history(results_csv))
    result_entry.update(
        {
            "status": "completed",
            "reason": None,
            "best_weights_path": str(best_weights) if best_weights else None,
            "effective_batch": effective_batch,
            "training_metrics": training_metrics,
            "holdout_metrics": holdout_metrics,
            "learning_dynamics": learning_dynamics,
            "peak_vram_mb": peak_vram_mb,
            "save_dir": str(save_dir),
        }
    )
    result_payload["experiments"][result_key] = result_entry
    _write_results_payload(result_payload)
    print(json.dumps(result_entry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
