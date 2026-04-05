from __future__ import annotations

from src.core.frame_state import CanonicalTranslation, FrameEnvelope, FrameResult


def build_empty_result(frame: FrameEnvelope) -> FrameResult:
    return FrameResult(frame_url=frame.frame_url)


def build_protocol_placeholder_result(frame: FrameEnvelope, _: bytes | None = None) -> FrameResult:
    """Gercek inference baglanana kadar protokol smoke testi icin kullanilir."""
    result = build_empty_result(frame)
    if str(frame.health_status) == "1":
        result.detected_translations.append(frame.ground_truth_translation(source="echo_ground_truth"))
    else:
        result.detected_translations.append(
            CanonicalTranslation(
                translation_x=0.0,
                translation_y=0.0,
                translation_z=0.0,
                source="placeholder_fallback",
            )
        )
    result.diagnostics["processor"] = "placeholder"
    return result
