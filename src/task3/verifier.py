from __future__ import annotations

from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame, FrameEnvelope
from src.core.utils import bbox_area


def verify_matches(
    frame: FrameEnvelope,
    matches: list[CanonicalUndefinedObject],
    *,
    decoded_frame: DecodedFrame | None = None,
    min_inliers: int = 4,
    min_inlier_ratio: float = 0.35,
    min_similarity: float = 0.78,
    min_corroboration: float = 0.10,
) -> list[CanonicalUndefinedObject]:
    """Geometric verifier; placeholder adaylari da kontrollu sekilde gecirir."""

    if not matches:
        return []

    width = float(decoded_frame.width if decoded_frame is not None else frame.metadata.get("image_width", 640))
    height = float(decoded_frame.height if decoded_frame is not None else frame.metadata.get("image_height", 512))
    frame_area = max(width * height, 1.0)
    verified: list[CanonicalUndefinedObject] = []
    for match in matches:
        source = str(match.metadata.get("matcher_source", ""))
        if source.startswith("task3_placeholder"):
            match.metadata["verification_status"] = "placeholder_pass"
            verified.append(match)
            continue

        box = (
            float(match.top_left_x),
            float(match.top_left_y),
            float(match.bottom_right_x),
            float(match.bottom_right_y),
        )
        box_area = bbox_area(box)
        bbox_sane = box[2] > box[0] and box[3] > box[1]
        area_ratio = box_area / frame_area
        scale_ok = 0.0005 <= area_ratio <= 0.8
        inlier_count = int(match.metadata.get("inlier_count", 0))
        inlier_ratio = float(match.metadata.get("inlier_ratio", 0.0))
        score = float(match.metadata.get("match_score", 0.0))
        if source.startswith("task3_learned_descriptor"):
            similarity = float(match.metadata.get("similarity", 0.0))
            corroboration = float(match.metadata.get("corroboration", 0.0))
            passed = bbox_sane and scale_ok and similarity >= min_similarity and score >= 0.70 and corroboration >= min_corroboration
            match.metadata["similarity_ok"] = similarity >= min_similarity
            match.metadata["corroboration_ok"] = corroboration >= min_corroboration
        else:
            passed = bbox_sane and scale_ok and inlier_count >= min_inliers and inlier_ratio >= min_inlier_ratio and score >= 0.70
        match.metadata["verification_status"] = "verified" if passed else "rejected"
        match.metadata["bbox_sane"] = bbox_sane
        match.metadata["scale_ok"] = scale_ok
        if passed:
            verified.append(match)
    return verified
