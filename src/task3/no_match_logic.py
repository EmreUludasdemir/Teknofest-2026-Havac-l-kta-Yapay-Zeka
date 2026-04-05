from __future__ import annotations

from src.core.frame_state import CanonicalUndefinedObject


def filter_no_match_candidates(
    matches: list[CanonicalUndefinedObject],
    *,
    min_score: float = 0.70,
    ambiguity_margin: float = 0.05,
) -> list[CanonicalUndefinedObject]:
    """Belirsiz durumda kutu basmaz; tek guvenilir adayi birakir."""

    scored = sorted(matches, key=lambda item: float(item.metadata.get("match_score", 0.0)), reverse=True)
    filtered = [item for item in scored if float(item.metadata.get("match_score", 0.0)) >= min_score]
    if not filtered:
        return []
    if len(filtered) > 1:
        best = float(filtered[0].metadata.get("match_score", 0.0))
        second = float(filtered[1].metadata.get("match_score", 0.0))
        if (best - second) < ambiguity_margin:
            return []
    return [filtered[0]]
