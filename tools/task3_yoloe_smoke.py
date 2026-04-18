from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_test_images():
    import numpy as np

    reference = np.zeros((128, 128, 3), dtype=np.uint8)
    reference[24:104, 32:96] = 255
    target = np.zeros((256, 256, 3), dtype=np.uint8)
    target[80:208, 96:160] = 255
    prompts = {
        "bboxes": np.array([[32, 24, 95, 103]], dtype=np.float32),
        "cls": np.array([0], dtype=np.int64),
    }
    return reference, target, prompts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual YOLOE visual-prompt smoke gate")
    parser.add_argument("--weight", default="data/weights/task3/yoloe-11m-seg.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args(argv)

    weight_path = Path(args.weight)
    if not weight_path.exists():
        raise FileNotFoundError(f"Staged weight not found: {weight_path}")

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    reference, target, prompts = build_test_images()
    model = YOLOE(str(weight_path))
    results = model.predict(
        target,
        refer_image=reference,
        visual_prompts=prompts,
        predictor=YOLOEVPSegPredictor,
        device=args.device,
        imgsz=args.imgsz,
        conf=0.05,
        verbose=False,
    )
    if not results:
        raise RuntimeError("Smoke predict returned no results")
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("Smoke predict returned zero boxes")

    print("SMOKE_OK")
    print(f"weight={weight_path}")
    print(f"device={args.device}")
    print(f"boxes={len(boxes)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - manual gate
        print(f"SMOKE_FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
