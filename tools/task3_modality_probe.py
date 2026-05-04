from __future__ import annotations

"""Probe utility for standalone Task 3 reference modality detection."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task3.modality_detection import detect_modality


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Task 3 reference modality detection.")
    parser.add_argument("images", nargs="+", type=Path, help="One or more image paths to classify.")
    args = parser.parse_args()

    for image_path in args.images:
        modality, diagnostics = detect_modality(image_path)
        exif_signals = ",".join(diagnostics["exif_signals"]) or "-"
        pixel_signals = diagnostics["pixel_signals"]
        summary = (
            f"sat={pixel_signals.get('mean_saturation', '-')}, "
            f"corr={pixel_signals.get('min_channel_correlation', '-')}, "
            f"bimodal={pixel_signals.get('histogram_bimodal', '-')}"
        )
        print(
            f"{image_path.name:<18} {modality.value.upper():<8} {diagnostics['method']:<5} "
            f"{diagnostics['confidence']:<6} exif=[{exif_signals}] {summary}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
