from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.evaluation.task1_trt_validation import validate_task1_trt_export
from src.tools.report_paths import EXPORT_REPORTS_DIR
from src.tools.runtime_package import prepare_runtime_package


def _default_model_dir() -> Path:
    return Path.home() / ".teknofest_models"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TEKNOFEST Faz 8 TRT validation")
    parser.add_argument("--output-dir", default=str(EXPORT_REPORTS_DIR))
    parser.add_argument("--primary", default="yolo26n")
    parser.add_argument("--fallback", default="yolo11n")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--primary-path")
    parser.add_argument("--fallback-path")
    args = parser.parse_args(argv)

    model_dir = _default_model_dir()
    primary_path = Path(args.primary_path) if args.primary_path else model_dir / f"{args.primary}.pt"
    fallback_path = Path(args.fallback_path) if args.fallback_path else model_dir / f"{args.fallback}.pt"
    export_dir = Path(args.output_dir) / "models"
    settings = MvpRuntimeSettings(
        task1_detector_backend=args.primary,
        task1_model_runtime="tensorrt",
        task1_device=args.device,
        task1_model_fallback_candidate=args.fallback,
        task1_candidate_paths={
            args.primary: str(primary_path),
            args.fallback: str(fallback_path),
        },
        task1_onnx_candidate_paths={args.primary: str(export_dir / f"{args.primary}.onnx")},
        task1_trt_candidate_paths={args.primary: str(export_dir / f"{args.primary}.engine")},
        task1_export_dir=export_dir,
    )
    payload = validate_task1_trt_export(
        settings,
        primary_candidate=args.primary,
        fallback_candidate=args.fallback,
        output_dir=args.output_dir,
    )
    prepare_runtime_package(settings)
    print(json.dumps(payload["comparison"], indent=2))
    return 0 if bool(payload["comparison"].get("accepted")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
