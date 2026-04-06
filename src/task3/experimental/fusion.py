from __future__ import annotations

from src.core.frame_state import CanonicalUndefinedObject
from src.task3.no_match_logic import filter_no_match_candidates


def fuse_candidates(
    candidates: list[CanonicalUndefinedObject],
    *,
    min_score: float,
    ambiguity_margin: float,
) -> list[CanonicalUndefinedObject]:
    """Detector, tracker ve verifier puanlarini tek bir karar skoruna indirger."""

    for item in candidates:
        detector_score = float(item.metadata.get("detection_score", item.metadata.get("match_score", 0.0)))
        tracker_score = float(item.metadata.get("tracking_score", 0.0))
        verifier_score = float(item.metadata.get("verifier_score", detector_score))
        fused = (detector_score * 0.55) + (tracker_score * 0.15) + (verifier_score * 0.30)
        item.metadata["fused_score"] = round(fused, 4)
        item.metadata["match_score"] = round(fused, 4)
    return filter_no_match_candidates(
        candidates,
        min_score=min_score,
        ambiguity_margin=ambiguity_margin,
    )
