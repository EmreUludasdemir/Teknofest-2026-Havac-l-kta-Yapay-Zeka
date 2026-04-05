from __future__ import annotations

from typing import Any

from src.core.frame_state import CanonicalDetection, DecodedFrame, FrameEnvelope
from src.core.utils import extract_frame_index, infer_modality

try:
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - ortama bagimli
    cv2 = None
    np = None


def is_cv2_available() -> bool:
    return cv2 is not None and np is not None


def decode_image_bytes(frame: FrameEnvelope, image_bytes: bytes) -> DecodedFrame:
    frame_index = int(frame.metadata.get("frame_index", extract_frame_index(frame.frame_url)))
    fallback_width = int(frame.metadata.get("image_width", 640))
    fallback_height = int(frame.metadata.get("image_height", 512))
    fallback_modality = infer_modality(
        frame.video_name,
        width=fallback_width,
        height=fallback_height,
        camera_mode=str(frame.metadata.get("camera_mode", "")) or None,
    )

    if not is_cv2_available():
        return DecodedFrame(
            bgr=None,
            gray=None,
            width=fallback_width,
            height=fallback_height,
            channel_count=0,
            modality=fallback_modality,
            frame_index=frame_index,
            metadata={"decode_mode": "fallback_no_cv2"},
        )

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return DecodedFrame(
            bgr=None,
            gray=None,
            width=fallback_width,
            height=fallback_height,
            channel_count=0,
            modality=fallback_modality,
            frame_index=frame_index,
            metadata={"decode_mode": "fallback_invalid_image"},
        )

    height, width = image.shape[:2]
    if width < 32 or height < 32:
        return DecodedFrame(
            bgr=None,
            gray=None,
            width=fallback_width,
            height=fallback_height,
            channel_count=0,
            modality=fallback_modality,
            frame_index=frame_index,
            metadata={"decode_mode": "fallback_tiny_image"},
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    modality = infer_modality(
        frame.video_name,
        width=int(width),
        height=int(height),
        camera_mode=str(frame.metadata.get("camera_mode", "")) or None,
    )
    return DecodedFrame(
        bgr=image,
        gray=gray,
        width=int(width),
        height=int(height),
        channel_count=int(image.shape[2]) if len(image.shape) == 3 else 1,
        modality=modality,
        frame_index=frame_index,
        metadata={"decode_mode": "opencv"},
    )


def build_detection_mask(decoded_frame: DecodedFrame, detections: list[CanonicalDetection]) -> tuple[Any | None, dict[str, object]]:
    if not is_cv2_available() or decoded_frame.gray is None:
        return None, {"mask_coverage_ok": False, "masked_boxes": len(detections)}

    mask = np.full((decoded_frame.height, decoded_frame.width), 255, dtype=np.uint8)
    masked_pixels = 0
    for detection in detections:
        x1 = max(int(detection.top_left_x) - 2, 0)
        y1 = max(int(detection.top_left_y) - 2, 0)
        x2 = min(int(detection.bottom_right_x) + 2, decoded_frame.width)
        y2 = min(int(detection.bottom_right_y) + 2, decoded_frame.height)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 0
        masked_pixels += (x2 - x1) * (y2 - y1)

    total_pixels = max(decoded_frame.width * decoded_frame.height, 1)
    coverage = 1.0 - (masked_pixels / float(total_pixels))
    return mask, {
        "mask_coverage_ok": coverage >= 0.25,
        "mask_coverage_ratio": round(coverage, 4),
        "masked_boxes": len(detections),
    }
