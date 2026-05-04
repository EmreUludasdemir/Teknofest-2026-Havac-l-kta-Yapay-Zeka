from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic Task 3 reference bank from a video crop spec")
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--crops-spec", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    import cv2

    source_video = Path(args.source_video)
    spec_path = Path(args.crops_spec)
    out_dir = Path(args.out_dir)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    items = list(payload.get("items", []))
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video acilamadi: {source_video}")

    manifest_items: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(items, start=1):
            reference_id = str(item.get("reference_id") or f"ref_{index:02d}")
            frame_idx = int(item["frame_idx"])
            x1, y1, x2, y2 = [int(value) for value in item["bbox"]]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Frame okunamadi: {frame_idx}")

            height, width = frame.shape[:2]
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, width)
            y2 = min(y2, height)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Gecersiz bbox for {reference_id}: {item['bbox']}")
            crop = frame[y1:y2, x1:x2]
            output_path = out_dir / f"{reference_id}.jpg"
            cv2.imwrite(str(output_path), crop)
            manifest_items.append(
                {
                    "reference_id": reference_id,
                    "frame_idx": frame_idx,
                    "bbox": [x1, y1, x2, y2],
                    "difficulty_tag": str(item.get("difficulty_tag", "unspecified")),
                    "source_video": str(source_video),
                    "output_file": output_path.name,
                }
            )
    finally:
        capture.release()

    manifest = {
        "source_video": str(source_video),
        "spec_path": str(spec_path),
        "reference_count": len(manifest_items),
        "items": manifest_items,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
