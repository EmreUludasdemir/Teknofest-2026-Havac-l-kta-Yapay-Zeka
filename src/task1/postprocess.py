from __future__ import annotations

from src.core.frame_state import CanonicalDetection
from src.core.utils import compute_iou


def deduplicate_detections(
    detections: list[CanonicalDetection],
    *,
    max_objects_per_frame: int = 16,
    iou_threshold: float = 0.6,
) -> list[CanonicalDetection]:
    """Class-aware dedup ve max object limit uygular."""

    scored = sorted(
        detections,
        key=lambda item: float(item.metadata.get("score", 0.0)),
        reverse=True,
    )
    kept: list[CanonicalDetection] = []
    for candidate in scored:
        candidate_box = (
            float(candidate.top_left_x),
            float(candidate.top_left_y),
            float(candidate.bottom_right_x),
            float(candidate.bottom_right_y),
        )
        is_duplicate = False
        for existing in kept:
            if existing.class_id != candidate.class_id:
                continue
            existing_box = (
                float(existing.top_left_x),
                float(existing.top_left_y),
                float(existing.bottom_right_x),
                float(existing.bottom_right_y),
            )
            if compute_iou(candidate_box, existing_box) >= iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
        if len(kept) >= max_objects_per_frame:
            break
    return kept
