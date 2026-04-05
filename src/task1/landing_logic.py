from __future__ import annotations

from src.core.frame_state import CanonicalDetection, DecodedFrame, FrameEnvelope


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def assign_landing_status(
    detections: list[CanonicalDetection],
    frame: FrameEnvelope,
    *,
    decoded_frame: DecodedFrame | None = None,
    frame_width: float | None = None,
    frame_height: float | None = None,
    margin_px: float = 12.0,
) -> list[CanonicalDetection]:
    """UAP/UAI alanlari icin geometrik MVP inis karari."""

    if decoded_frame is not None:
        effective_width = float(decoded_frame.width)
        effective_height = float(decoded_frame.height)
    else:
        effective_width = float(frame_width or frame.metadata.get("image_width", 640.0))
        effective_height = float(frame_height or frame.metadata.get("image_height", 512.0))

    for detection in detections:
        if detection.class_id not in {2, 3}:
            detection.landing_status = -1
            continue

        detection.landing_status = 1
        bbox = (
            float(detection.top_left_x),
            float(detection.top_left_y),
            float(detection.bottom_right_x),
            float(detection.bottom_right_y),
        )
        in_frame = bbox[0] >= 0.0 and bbox[1] >= 0.0 and bbox[2] <= effective_width and bbox[3] <= effective_height
        if not in_frame:
            detection.landing_status = 0
            continue

        expanded = (
            bbox[0] - margin_px,
            bbox[1] - margin_px,
            bbox[2] + margin_px,
            bbox[3] + margin_px,
        )
        for other in detections:
            if other is detection:
                continue
            other_box = (
                float(other.top_left_x),
                float(other.top_left_y),
                float(other.bottom_right_x),
                float(other.bottom_right_y),
            )
            if _overlaps(expanded, other_box):
                detection.landing_status = 0
                break
    return detections
