from __future__ import annotations

from dataclasses import replace

from src.core.frame_state import CanonicalTranslation
from src.core.utils import clamp


def apply_drift_guard(
    candidate: CanonicalTranslation,
    previous_output: CanonicalTranslation | None,
    previous_reliable: CanonicalTranslation | None,
    *,
    anchor_translation: CanonicalTranslation | None = None,
    confidence: float = 1.0,
    confidence_floor: float = 0.30,
    estimation_mode: str = "unknown",
    mask_coverage_ok: bool = True,
    max_step_xy: float = 5.0,
    max_step_z: float = 2.0,
    long_drift_limit: float = 4.0,
    anchor_distance_limit_xy: float | None = None,
    anchor_distance_limit_z: float | None = None,
) -> tuple[CanonicalTranslation, dict[str, object]]:
    """Confidence-aware kisa ve uzun sicrama filtresi uygular."""

    reference = previous_output or previous_reliable
    if reference is None:
        guarded = replace(candidate, source=candidate.source)
        return guarded, {
            "drift_clamped": False,
            "long_drift_clamped": False,
            "anchor_distance_clamped": False,
            "confidence": round(float(confidence), 4),
            "estimation_mode": estimation_mode,
            "mask_coverage_ok": mask_coverage_ok,
            "fallback_source": None,
        }

    dx = candidate.translation_x - reference.translation_x
    dy = candidate.translation_y - reference.translation_y
    dz = candidate.translation_z - reference.translation_z

    clamped_x = reference.translation_x + clamp(dx, -max_step_xy, max_step_xy)
    clamped_y = reference.translation_y + clamp(dy, -max_step_xy, max_step_xy)
    clamped_z = reference.translation_z + clamp(dz, -max_step_z, max_step_z)
    did_clamp = any(
        (
            abs(dx) > max_step_xy,
            abs(dy) > max_step_xy,
            abs(dz) > max_step_z,
        )
    )

    guarded_candidate = replace(
        candidate,
        translation_x=clamped_x,
        translation_y=clamped_y,
        translation_z=clamped_z,
    )

    long_clamped = False
    if previous_reliable is not None:
        long_dx = guarded_candidate.translation_x - previous_reliable.translation_x
        long_dy = guarded_candidate.translation_y - previous_reliable.translation_y
        long_dz = guarded_candidate.translation_z - previous_reliable.translation_z
        long_limit_xy = max_step_xy * max(long_drift_limit, 1.0)
        long_limit_z = max_step_z * max(long_drift_limit, 1.0)
        if any(
            (
                abs(long_dx) > long_limit_xy,
                abs(long_dy) > long_limit_xy,
                abs(long_dz) > long_limit_z,
            )
        ):
            guarded_candidate = replace(
                guarded_candidate,
                translation_x=previous_reliable.translation_x + clamp(long_dx, -long_limit_xy, long_limit_xy),
                translation_y=previous_reliable.translation_y + clamp(long_dy, -long_limit_xy, long_limit_xy),
                translation_z=previous_reliable.translation_z + clamp(long_dz, -long_limit_z, long_limit_z),
            )
            long_clamped = True

    anchor_distance_clamped = False
    if anchor_translation is not None:
        anchor_limit_xy = anchor_distance_limit_xy if anchor_distance_limit_xy is not None else max_step_xy * max(long_drift_limit, 1.0)
        anchor_limit_z = anchor_distance_limit_z if anchor_distance_limit_z is not None else max_step_z * max(long_drift_limit, 1.0)
        anchor_dx = guarded_candidate.translation_x - anchor_translation.translation_x
        anchor_dy = guarded_candidate.translation_y - anchor_translation.translation_y
        anchor_dz = guarded_candidate.translation_z - anchor_translation.translation_z
        if any(
            (
                abs(anchor_dx) > anchor_limit_xy,
                abs(anchor_dy) > anchor_limit_xy,
                abs(anchor_dz) > anchor_limit_z,
            )
        ):
            guarded_candidate = replace(
                guarded_candidate,
                translation_x=anchor_translation.translation_x + clamp(anchor_dx, -anchor_limit_xy, anchor_limit_xy),
                translation_y=anchor_translation.translation_y + clamp(anchor_dy, -anchor_limit_xy, anchor_limit_xy),
                translation_z=anchor_translation.translation_z + clamp(anchor_dz, -anchor_limit_z, anchor_limit_z),
            )
            anchor_distance_clamped = True

    fallback_source: str | None = None
    guarded = guarded_candidate
    if confidence < confidence_floor:
        if previous_output is not None:
            guarded = replace(
                previous_output,
                source="task2_estimated_low_confidence",
            )
            fallback_source = "previous_output"
        elif previous_reliable is not None:
            guarded = replace(
                previous_reliable,
                source="task2_estimated_low_confidence",
            )
            fallback_source = "previous_reliable"

    return guarded, {
        "drift_clamped": did_clamp,
        "long_drift_clamped": long_clamped,
        "anchor_distance_clamped": anchor_distance_clamped,
        "confidence": round(float(confidence), 4),
        "estimation_mode": estimation_mode,
        "mask_coverage_ok": mask_coverage_ok,
        "fallback_source": fallback_source,
    }
