from __future__ import annotations

from src.core.frame_state import CanonicalDetection, FrameEnvelope


def assign_motion_status(
    detections: list[CanonicalDetection],
    frame: FrameEnvelope,
    *,
    threshold_px: float = 8.0,
) -> list[CanonicalDetection]:
    """Yalnizca tasitlar icin basit centroid farkina dayali hareket mantigi."""

    for detection in detections:
        if detection.class_id != 0:
            detection.motion_status = -1
            continue
        previous = detection.metadata.get("previous_centroid")
        current = detection.metadata.get("current_centroid")
        if previous is None or current is None:
            detection.motion_status = 0
            continue
        dx = float(current[0]) - float(previous[0])
        dy = float(current[1]) - float(previous[1])
        pixel_shift = (dx * dx + dy * dy) ** 0.5
        detection.motion_status = 1 if pixel_shift >= threshold_px else 0
    return detections
