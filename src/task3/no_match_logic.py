from __future__ import annotations

from src.core.frame_state import CanonicalUndefinedObject

ORB_SCORE_CONFIDENCE_WEIGHT = 0.70
ORB_SCORE_MATCHES_WEIGHT = 0.30


def compute_mode_candidate_score(
    *,
    confidence: float,
    normalized_matches: float,
    inlier_ratio: float = 0.0,
    mode: str,
    yoloe_confidence_weight: float,
    yoloe_matches_weight: float,
    yoloe_inlier_weight: float = 0.0,
) -> float:
    if mode == "yoloe_vp_lightglue":
        return (
            (float(confidence) * float(yoloe_confidence_weight))
            + (float(normalized_matches) * float(yoloe_matches_weight))
            + (float(inlier_ratio) * float(yoloe_inlier_weight))
        )
    return (float(confidence) * ORB_SCORE_CONFIDENCE_WEIGHT) + (float(normalized_matches) * ORB_SCORE_MATCHES_WEIGHT)


def normalize_yoloe_match_count(
    *,
    match_count: int,
    normalization_scale: int,
) -> float:
    bounded_matches = max(int(match_count), 0)
    bounded_scale = max(int(normalization_scale), 1)
    normalized = float(bounded_matches) / float(bounded_scale)
    return min(max(normalized, 0.0), 1.0)


def resolve_min_score_for_mode(
    *,
    mode: str,
    min_score: float,
    yoloe_min_score: float | None = None,
    modality: str | None = None,
    yoloe_thermal_min_score: float | None = None,
) -> float:
    if mode == "yoloe_vp_lightglue" and modality == "thermal" and yoloe_thermal_min_score is not None:
        return float(yoloe_thermal_min_score)
    if mode == "yoloe_vp_lightglue" and yoloe_min_score is not None:
        return float(yoloe_min_score)
    return float(min_score)


def filter_no_match_candidates(
    matches: list[CanonicalUndefinedObject],
    *,
    min_score: float = 0.70,
    mode: str = "orb_template",
    yoloe_min_score: float | None = None,
    modality: str | None = None,
    yoloe_thermal_min_score: float | None = None,
    ambiguity_margin: float = 0.05,
) -> list[CanonicalUndefinedObject]:
    """Belirsiz durumda kutu basmaz; tek guvenilir adayi birakir."""

    scored = sorted(matches, key=lambda item: float(item.metadata.get("match_score", 0.0)), reverse=True)
    resolved_min_score = resolve_min_score_for_mode(
        mode=mode,
        min_score=min_score,
        yoloe_min_score=yoloe_min_score,
        modality=modality,
        yoloe_thermal_min_score=yoloe_thermal_min_score,
    )
    filtered = [item for item in scored if float(item.metadata.get("match_score", 0.0)) >= resolved_min_score]
    if not filtered:
        return []
    if len(filtered) > 1:
        best = float(filtered[0].metadata.get("match_score", 0.0))
        second = float(filtered[1].metadata.get("match_score", 0.0))
        if (best - second) < ambiguity_margin:
            return []
    return [filtered[0]]
