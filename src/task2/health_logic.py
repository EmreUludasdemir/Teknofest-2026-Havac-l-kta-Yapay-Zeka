from __future__ import annotations

from src.core.frame_state import CanonicalTranslation, CanonicalDetection, DecodedFrame, FrameEnvelope
from src.task2.calibration import select_calibration_profile
from src.task2.estimator import Task2Estimator


def should_use_reference_translation(frame: FrameEnvelope) -> bool:
    """Saglik biti 1 ise referans deger kullanma kararini verir."""

    return str(frame.health_status) == "1"


def resolve_task2_translation(
    frame: FrameEnvelope,
    decoded_frame: DecodedFrame,
    estimator: Task2Estimator,
    *,
    dynamic_detections: list[CanonicalDetection] | None = None,
) -> tuple[CanonicalTranslation, dict[str, object]]:
    """Task 2 icin referans/estimator ayrimini acik ve test edilebilir sekilde kurar."""

    if should_use_reference_translation(frame):
        translation = frame.ground_truth_translation(source="task2_reference")
        calibration_profile = select_calibration_profile(estimator.calibration_bundle, frame, decoded_frame)
        estimator.observe_reference(
            translation,
            decoded_frame=decoded_frame,
            calibration_profile=calibration_profile,
        )
        return translation, {
            "task2_branch": "reference",
            "health_status": str(frame.health_status),
            "drift_clamped": False,
            "confidence": 1.0,
            "calibration_modality": calibration_profile.modality,
        }

    translation, diagnostics = estimator.estimate(
        frame,
        decoded_frame,
        dynamic_detections=dynamic_detections,
    )
    diagnostics["task2_branch"] = "estimated"
    diagnostics["health_status"] = str(frame.health_status)
    return translation, diagnostics
