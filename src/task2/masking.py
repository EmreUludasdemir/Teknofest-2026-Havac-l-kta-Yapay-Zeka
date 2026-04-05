from __future__ import annotations

from typing import Any

from src.core.frame_state import CanonicalDetection
from src.core.frame_state import DecodedFrame
from src.core.vision import build_detection_mask


def mask_dynamic_objects(
    decoded_frame: DecodedFrame,
    detections: list[CanonicalDetection],
) -> tuple[Any | None, dict[str, object]]:
    """Bir onceki kareden gelen dinamik bbox'lari maske olarak kullanir."""

    return build_detection_mask(decoded_frame, detections)
