from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalDetection, CanonicalTranslation, DecodedFrame, FrameEnvelope
from src.core.utils import extract_frame_index
from src.core.vision import is_cv2_available
from src.task2.calibration import (
    CalibrationBundle,
    CameraCalibrationProfile,
    load_calibration_bundle,
    pixel_shift_to_translation_delta,
    select_calibration_profile,
)
from src.task2.drift_control import apply_drift_guard
from src.task2.masking import mask_dynamic_objects

if is_cv2_available():  # pragma: no branch - import durumu ortama bagli
    from src.core.vision import cv2, np
else:  # pragma: no cover - cv2 yoksa
    cv2 = None
    np = None


@dataclass(slots=True)
class Task2Estimator:
    """Task 2 icin anchor-temelli, goruntu kullanan estimator."""

    runtime_settings: MvpRuntimeSettings
    calibration_bundle: CalibrationBundle = field(init=False)
    last_reliable_translation: CanonicalTranslation | None = None
    last_output_translation: CanonicalTranslation | None = None
    last_velocity: CanonicalTranslation | None = None
    anchor_translation: CanonicalTranslation | None = None
    last_decoded_frame: DecodedFrame | None = None
    last_reference_frame: DecodedFrame | None = None
    anchor_frame: DecodedFrame | None = None
    last_confidence: float = 0.0
    last_calibration_profile: CameraCalibrationProfile | None = None
    anchor_calibration_profile: CameraCalibrationProfile | None = None
    last_health_status: str | None = None
    health_epoch: int = 0
    health0_frame_count: int = 0

    def __post_init__(self) -> None:
        self.calibration_bundle = load_calibration_bundle(self.runtime_settings.task2_calibration_path)

    def observe_reference(
        self,
        reference_translation: CanonicalTranslation,
        *,
        decoded_frame: DecodedFrame | None = None,
        calibration_profile: CameraCalibrationProfile | None = None,
    ) -> None:
        if self.last_reliable_translation is not None:
            self.last_velocity = CanonicalTranslation(
                translation_x=reference_translation.translation_x - self.last_reliable_translation.translation_x,
                translation_y=reference_translation.translation_y - self.last_reliable_translation.translation_y,
                translation_z=reference_translation.translation_z - self.last_reliable_translation.translation_z,
                source="task2_velocity",
            )

        if self.last_health_status == "0":
            self.health_epoch += 1

        self.last_reliable_translation = CanonicalTranslation(
            translation_x=reference_translation.translation_x,
            translation_y=reference_translation.translation_y,
            translation_z=reference_translation.translation_z,
            source=reference_translation.source,
        )
        self.last_output_translation = CanonicalTranslation(
            translation_x=reference_translation.translation_x,
            translation_y=reference_translation.translation_y,
            translation_z=reference_translation.translation_z,
            source=reference_translation.source,
        )
        self.anchor_translation = CanonicalTranslation(
            translation_x=reference_translation.translation_x,
            translation_y=reference_translation.translation_y,
            translation_z=reference_translation.translation_z,
            source=reference_translation.source,
        )
        self.last_reference_frame = decoded_frame
        self.last_decoded_frame = decoded_frame
        self.anchor_frame = decoded_frame
        self.last_calibration_profile = calibration_profile
        self.anchor_calibration_profile = calibration_profile
        self.last_confidence = 1.0
        self.last_health_status = "1"
        self.health0_frame_count = 0

    def estimate(
        self,
        frame: FrameEnvelope,
        decoded_frame: DecodedFrame,
        *,
        dynamic_detections: list[CanonicalDetection] | None = None,
    ) -> tuple[CanonicalTranslation, dict[str, object]]:
        entering_health0 = self.last_health_status != "0"
        if entering_health0:
            self.health_epoch += 1
            self.health0_frame_count = 0
            if self.last_reference_frame is not None:
                self.anchor_frame = self.last_reference_frame
            if self.last_reliable_translation is not None:
                self.anchor_translation = CanonicalTranslation(
                    translation_x=self.last_reliable_translation.translation_x,
                    translation_y=self.last_reliable_translation.translation_y,
                    translation_z=self.last_reliable_translation.translation_z,
                    source=self.last_reliable_translation.source,
                )
            self.anchor_calibration_profile = self.last_calibration_profile
        self.last_health_status = "0"
        self.health0_frame_count += 1

        current_profile = select_calibration_profile(self.calibration_bundle, frame, decoded_frame)
        self.last_calibration_profile = current_profile
        thermal_guard_active = current_profile.modality == "thermal"
        phase_primary_response_min = (
            self.runtime_settings.task2_phase_primary_response_min_thermal
            if thermal_guard_active
            else self.runtime_settings.task2_phase_primary_response_min
        )
        confidence_floor = (
            self.runtime_settings.task2_confidence_floor_thermal
            if thermal_guard_active
            else self.runtime_settings.task2_confidence_floor
        )
        z_update_scale = (
            self.runtime_settings.task2_z_update_scale_thermal
            if thermal_guard_active
            else self.runtime_settings.task2_z_update_scale
        )
        sensor_hint_max_weight = (
            self.runtime_settings.task2_sensor_hint_max_weight_thermal
            if thermal_guard_active
            else 0.50
        )
        base_output = self.last_output_translation or self.last_reliable_translation
        anchor_translation = self.anchor_translation or self.last_reliable_translation or self.last_output_translation
        sensor_hint = CanonicalTranslation(
            translation_x=frame.translation_x,
            translation_y=frame.translation_y,
            translation_z=frame.translation_z,
            source="task2_sensor_hint",
        )
        if base_output is None or anchor_translation is None:
            estimated = CanonicalTranslation(0.0, 0.0, 0.0, source="task2_zero_fallback")
            self.last_output_translation = estimated
            self.last_decoded_frame = decoded_frame
            return estimated, {
                "estimator_branch": "zero_fallback",
                "used_last_velocity": False,
                "confidence": 0.0,
                "calibration_modality": current_profile.modality,
                "health_epoch": self.health_epoch,
                "health0_frame_count": self.health0_frame_count,
                "health_window_id": self.health_epoch,
                "hold_mode_reason": "missing_anchor_state",
                "thermal_guard_active": thermal_guard_active,
                "sensor_hint_seen": True,
                "sensor_hint_used": False,
                "sensor_hint_weight": 0.0,
            }

        if (
            self.anchor_calibration_profile is not None
            and self.anchor_calibration_profile.modality != current_profile.modality
        ):
            held = self._build_hold_translation(anchor_translation, base_output)
            guarded, diagnostics = apply_drift_guard(
                held,
                self.last_output_translation,
                self.last_reliable_translation,
                anchor_translation=anchor_translation,
                confidence=min(self.last_confidence, confidence_floor * 0.5),
                confidence_floor=confidence_floor,
                estimation_mode="hold_mode_modality_mismatch",
                mask_coverage_ok=False,
                max_step_xy=self.runtime_settings.task2_max_step_xy,
                max_step_z=self.runtime_settings.task2_max_step_z,
                long_drift_limit=self.runtime_settings.task2_long_drift_limit,
                anchor_distance_limit_xy=self.runtime_settings.task2_anchor_distance_limit_xy,
                anchor_distance_limit_z=self.runtime_settings.task2_anchor_distance_limit_z,
            )
            self.last_output_translation = guarded
            self.last_decoded_frame = decoded_frame
            self.last_confidence = float(diagnostics.get("confidence", 0.0))
            diagnostics.update(
                {
                    "estimator_branch": "hold_mode",
                    "calibration_modality": current_profile.modality,
                    "health_epoch": self.health_epoch,
                    "health0_frame_count": self.health0_frame_count,
                    "health_window_id": self.health_epoch,
                    "hold_mode_reason": "modality_mismatch",
                    "thermal_guard_active": thermal_guard_active,
                    "sensor_hint_seen": True,
                    "sensor_hint_used": False,
                    "sensor_hint_weight": 0.0,
                    "anchor_refreshed": False,
                }
            )
            return guarded, diagnostics

        mask, mask_info = mask_dynamic_objects(decoded_frame, dynamic_detections or [])
        anchor_frame = self.anchor_frame or self.last_reference_frame or self.last_decoded_frame
        previous_frame = self.last_decoded_frame or anchor_frame

        phase_x, phase_y, phase_response = self._estimate_phase_shift(anchor_frame, decoded_frame, mask)
        flow_x, flow_y, flow_info = self._estimate_flow_shift(previous_frame, decoded_frame, mask)
        flow_point_count = int(flow_info.get("point_count", 0))
        flow_confidence = float(flow_info.get("confidence", 0.0))
        mask_coverage_ok = bool(mask_info.get("mask_coverage_ok", False))

        phase_abs = self._absolute_from_anchor(
            anchor_translation,
            anchor_frame,
            decoded_frame,
            shift_x=phase_x,
            shift_y=phase_y,
            calibration_profile=current_profile,
            z_confidence=phase_response,
            z_update_scale=z_update_scale,
        )
        incremental_abs = self._absolute_from_increment(
            base_output,
            previous_frame,
            decoded_frame,
            shift_x=flow_x,
            shift_y=flow_y,
            calibration_profile=current_profile,
            z_confidence=flow_confidence,
            z_update_scale=z_update_scale,
        )
        velocity_prior = self._build_velocity_prior(base_output)

        phase_motion_strength = self._motion_strength(phase_x, phase_y)
        flow_motion_strength = self._motion_strength(flow_x, flow_y)
        phase_primary = phase_response >= phase_primary_response_min and mask_coverage_ok
        flow_primary = flow_point_count >= self.runtime_settings.task2_flow_primary_points
        flow_recoverable = (
            flow_point_count >= self.runtime_settings.task2_flow_min_points and flow_confidence > 0.0
        )
        estimation_mode = "hold_mode"
        hold_mode_reason = "weak_visual_signal"
        weights = self._normalize_weights(0.30, 0.35, 0.35)
        confidence = max(min(self.last_confidence * 0.45, 0.22), 0.08)
        candidate = self._blend_states(
            anchor_translation,
            base_output,
            velocity_prior,
            weights=weights,
            source="task2_estimated",
        )

        if phase_primary:
            estimation_mode = "phase_fusion"
            hold_mode_reason = None
            phase_weight = 0.28 + (0.30 * phase_motion_strength)
            flow_weight = 0.12 + (0.15 * flow_motion_strength)
            velocity_weight = 1.0 - phase_weight - flow_weight
            weights = self._normalize_weights(phase_weight, flow_weight, max(velocity_weight, 0.25))
            confidence = clamp_ratio(
                (phase_response * 0.45)
                + (flow_confidence * 0.10)
                + (phase_motion_strength * 0.20)
                + (flow_motion_strength * 0.10)
                + (self.last_confidence * 0.15)
            )
            candidate = self._blend_states(
                phase_abs,
                incremental_abs,
                velocity_prior,
                weights=weights,
                source="task2_estimated",
            )
        elif flow_primary or (not phase_primary and flow_recoverable):
            estimation_mode = "lucas_kanade"
            hold_mode_reason = None
            phase_weight = 0.12 + (0.15 * phase_motion_strength)
            flow_weight = 0.35 + (0.30 * flow_motion_strength)
            velocity_weight = 1.0 - phase_weight - flow_weight
            weights = self._normalize_weights(phase_weight, flow_weight, max(velocity_weight, 0.25))
            confidence = clamp_ratio(
                (flow_confidence * 0.40)
                + (flow_motion_strength * 0.25)
                + (phase_response * 0.10)
                + (phase_motion_strength * 0.10)
                + (self.last_confidence * 0.15)
            )
            candidate = self._blend_states(
                phase_abs,
                incremental_abs,
                velocity_prior,
                weights=weights,
                source="task2_estimated",
            )
        elif thermal_guard_active:
            hold_mode_reason = "thermal_low_response"

        candidate, sensor_hint_info = self._apply_sensor_hint(
            candidate,
            sensor_hint,
            base_output=base_output,
            anchor_translation=anchor_translation,
            confidence=confidence,
            max_hint_weight=sensor_hint_max_weight,
        )

        guarded, guard_info = apply_drift_guard(
            candidate,
            self.last_output_translation,
            self.last_reliable_translation,
            anchor_translation=anchor_translation,
            confidence=confidence,
            confidence_floor=confidence_floor,
            estimation_mode=estimation_mode,
            mask_coverage_ok=mask_coverage_ok,
            max_step_xy=self.runtime_settings.task2_max_step_xy,
            max_step_z=self.runtime_settings.task2_max_step_z,
            long_drift_limit=self.runtime_settings.task2_long_drift_limit,
            anchor_distance_limit_xy=self.runtime_settings.task2_anchor_distance_limit_xy,
            anchor_distance_limit_z=self.runtime_settings.task2_anchor_distance_limit_z,
        )

        anchor_refreshed = False
        if (
            float(guard_info.get("confidence", confidence)) >= self.runtime_settings.task2_anchor_refresh_confidence
            and self.health0_frame_count % max(self.runtime_settings.task2_anchor_refresh_interval, 1) == 0
        ):
            self.anchor_frame = decoded_frame
            self.anchor_translation = CanonicalTranslation(
                translation_x=guarded.translation_x,
                translation_y=guarded.translation_y,
                translation_z=guarded.translation_z,
                source=guarded.source,
            )
            self.anchor_calibration_profile = current_profile
            anchor_refreshed = True

        self._update_velocity(guarded, self.last_output_translation, confidence=float(guard_info.get("confidence", confidence)))
        self.last_output_translation = guarded
        self.last_decoded_frame = decoded_frame
        self.last_confidence = float(guard_info.get("confidence", confidence))

        diagnostics: dict[str, object] = {
            "estimator_branch": "predicted",
            "used_last_velocity": self.last_velocity is not None,
            "frame_index": decoded_frame.frame_index,
            "calibration_modality": current_profile.modality,
            "health_epoch": self.health_epoch,
            "health0_frame_count": self.health0_frame_count,
            "health_window_id": self.health_epoch,
            "phase_response": round(float(phase_response), 4),
            "phase_primary": phase_primary,
            "flow_primary": flow_primary,
            "hold_mode_reason": hold_mode_reason,
            "thermal_guard_active": thermal_guard_active,
            "fusion_weights": [round(float(item), 3) for item in weights],
            "anchor_refreshed": anchor_refreshed,
        }
        diagnostics.update(mask_info)
        diagnostics.update(flow_info)
        diagnostics.update(sensor_hint_info)
        diagnostics.update(guard_info)
        return guarded, diagnostics

    def _absolute_from_anchor(
        self,
        anchor_translation: CanonicalTranslation,
        anchor_frame: DecodedFrame | None,
        decoded_frame: DecodedFrame,
        *,
        shift_x: float,
        shift_y: float,
        calibration_profile: CameraCalibrationProfile,
        z_confidence: float,
        z_update_scale: float,
    ) -> CanonicalTranslation:
        delta_x, delta_y = pixel_shift_to_translation_delta(
            shift_x,
            shift_y,
            reference_z=anchor_translation.translation_z,
            profile=calibration_profile,
        )
        z_signal = self._estimate_z_delta(anchor_frame, decoded_frame) * z_update_scale * clamp_ratio(z_confidence)
        return CanonicalTranslation(
            translation_x=anchor_translation.translation_x + delta_x,
            translation_y=anchor_translation.translation_y + delta_y,
            translation_z=anchor_translation.translation_z + z_signal,
            source="task2_estimated",
        )

    def _absolute_from_increment(
        self,
        base_output: CanonicalTranslation,
        previous_frame: DecodedFrame | None,
        decoded_frame: DecodedFrame,
        *,
        shift_x: float,
        shift_y: float,
        calibration_profile: CameraCalibrationProfile,
        z_confidence: float,
        z_update_scale: float,
    ) -> CanonicalTranslation:
        delta_x, delta_y = pixel_shift_to_translation_delta(
            shift_x,
            shift_y,
            reference_z=base_output.translation_z,
            profile=calibration_profile,
        )
        z_signal = self._estimate_z_delta(previous_frame, decoded_frame) * z_update_scale * clamp_ratio(z_confidence)
        return CanonicalTranslation(
            translation_x=base_output.translation_x + delta_x,
            translation_y=base_output.translation_y + delta_y,
            translation_z=base_output.translation_z + z_signal,
            source="task2_estimated",
        )

    def _build_velocity_prior(self, base_output: CanonicalTranslation) -> CanonicalTranslation:
        velocity = self.last_velocity or CanonicalTranslation(0.0, 0.0, 0.0, source="task2_zero_velocity")
        return CanonicalTranslation(
            translation_x=base_output.translation_x + (velocity.translation_x * self.runtime_settings.task2_velocity_decay),
            translation_y=base_output.translation_y + (velocity.translation_y * self.runtime_settings.task2_velocity_decay),
            translation_z=base_output.translation_z + (velocity.translation_z * self.runtime_settings.task2_velocity_decay),
            source="task2_estimated",
        )

    def _build_hold_translation(
        self,
        anchor_translation: CanonicalTranslation,
        base_output: CanonicalTranslation,
    ) -> CanonicalTranslation:
        velocity_prior = self._build_velocity_prior(base_output)
        return self._blend_states(
            anchor_translation,
            base_output,
            velocity_prior,
            weights=(0.45, 0.45, self.runtime_settings.task2_hold_mode_velocity_weight),
            source="task2_estimated",
        )

    def _blend_states(
        self,
        first: CanonicalTranslation,
        second: CanonicalTranslation,
        third: CanonicalTranslation,
        *,
        weights: tuple[float, float, float],
        source: str,
        ) -> CanonicalTranslation:
        weight_sum = max(sum(weights), 1e-6)
        normalized = tuple(weight / weight_sum for weight in weights)
        return CanonicalTranslation(
            translation_x=(
                (first.translation_x * normalized[0])
                + (second.translation_x * normalized[1])
                + (third.translation_x * normalized[2])
            ),
            translation_y=(
                (first.translation_y * normalized[0])
                + (second.translation_y * normalized[1])
                + (third.translation_y * normalized[2])
            ),
            translation_z=(
                (first.translation_z * normalized[0])
                + (second.translation_z * normalized[1])
                + (third.translation_z * normalized[2])
            ),
            source=source,
        )

    def _normalize_weights(self, first: float, second: float, third: float) -> tuple[float, float, float]:
        weight_sum = max(first + second + third, 1e-6)
        return (first / weight_sum, second / weight_sum, third / weight_sum)

    def _motion_strength(self, shift_x: float, shift_y: float) -> float:
        magnitude = hypot(float(shift_x), float(shift_y))
        return clamp_ratio(magnitude / 4.0)

    def _apply_sensor_hint(
        self,
        candidate: CanonicalTranslation,
        sensor_hint: CanonicalTranslation,
        *,
        base_output: CanonicalTranslation,
        anchor_translation: CanonicalTranslation,
        confidence: float,
        max_hint_weight: float,
    ) -> tuple[CanonicalTranslation, dict[str, object]]:
        candidate_distance = self._translation_distance(candidate, sensor_hint)
        base_distance = self._translation_distance(base_output, sensor_hint)
        anchor_distance = self._translation_distance(anchor_translation, sensor_hint)
        hint_weight = 0.0
        if candidate_distance <= 3.5 and anchor_distance <= 8.0:
            hint_weight = 0.35 + (0.15 * (1.0 - confidence))
        elif candidate_distance <= 6.0 and base_distance <= 10.0:
            hint_weight = 0.2 + (0.10 * (1.0 - confidence))
        hint_weight = min(hint_weight, max_hint_weight)

        if hint_weight <= 0.0:
            return candidate, {"sensor_hint_seen": True, "sensor_hint_used": False, "sensor_hint_weight": 0.0}

        blended = CanonicalTranslation(
            translation_x=(candidate.translation_x * (1.0 - hint_weight)) + (sensor_hint.translation_x * hint_weight),
            translation_y=(candidate.translation_y * (1.0 - hint_weight)) + (sensor_hint.translation_y * hint_weight),
            translation_z=(candidate.translation_z * (1.0 - hint_weight)) + (sensor_hint.translation_z * hint_weight),
            source=candidate.source,
        )
        return blended, {
            "sensor_hint_seen": True,
            "sensor_hint_used": True,
            "sensor_hint_weight": round(float(hint_weight), 4),
        }

    def _translation_distance(self, first: CanonicalTranslation, second: CanonicalTranslation) -> float:
        return hypot(
            hypot(first.translation_x - second.translation_x, first.translation_y - second.translation_y),
            first.translation_z - second.translation_z,
        )

    def _update_velocity(
        self,
        current_output: CanonicalTranslation,
        previous_output: CanonicalTranslation | None,
        *,
        confidence: float,
    ) -> None:
        if previous_output is None or confidence < self.runtime_settings.task2_confidence_floor:
            return
        instant = CanonicalTranslation(
            translation_x=current_output.translation_x - previous_output.translation_x,
            translation_y=current_output.translation_y - previous_output.translation_y,
            translation_z=current_output.translation_z - previous_output.translation_z,
            source="task2_velocity",
        )
        if self.last_velocity is None:
            self.last_velocity = instant
            return
        instant_magnitude = abs(instant.translation_x) + abs(instant.translation_y) + abs(instant.translation_z)
        new_weight = 0.15 + (0.35 * clamp_ratio(instant_magnitude / 1.5))
        historic_weight = 1.0 - new_weight
        self.last_velocity = CanonicalTranslation(
            translation_x=(self.last_velocity.translation_x * historic_weight) + (instant.translation_x * new_weight),
            translation_y=(self.last_velocity.translation_y * historic_weight) + (instant.translation_y * new_weight),
            translation_z=(self.last_velocity.translation_z * historic_weight) + (instant.translation_z * new_weight),
            source="task2_velocity",
        )

    def _estimate_phase_shift(
        self,
        previous_frame: DecodedFrame | None,
        decoded_frame: DecodedFrame,
        mask,
    ) -> tuple[float, float, float]:
        if (
            not is_cv2_available()
            or previous_frame is None
            or previous_frame.gray is None
            or decoded_frame.gray is None
            or previous_frame.width != decoded_frame.width
            or previous_frame.height != decoded_frame.height
        ):
            return 0.0, 0.0, 0.0

        prev_gray = previous_frame.gray
        curr_gray = decoded_frame.gray
        if mask is not None:
            prev_gray = cv2.bitwise_and(prev_gray, prev_gray, mask=mask)
            curr_gray = cv2.bitwise_and(curr_gray, curr_gray, mask=mask)
        shift, response = cv2.phaseCorrelate(prev_gray.astype(np.float32), curr_gray.astype(np.float32))
        return float(shift[0]), float(shift[1]), float(response)

    def _estimate_flow_shift(
        self,
        previous_frame: DecodedFrame | None,
        decoded_frame: DecodedFrame,
        mask,
    ) -> tuple[float, float, dict[str, object]]:
        if (
            not is_cv2_available()
            or previous_frame is None
            or previous_frame.gray is None
            or decoded_frame.gray is None
            or previous_frame.width != decoded_frame.width
            or previous_frame.height != decoded_frame.height
        ):
            return 0.0, 0.0, {"point_count": 0, "confidence": 0.0}

        prev_points = cv2.goodFeaturesToTrack(
            previous_frame.gray,
            maxCorners=64,
            qualityLevel=0.01,
            minDistance=6,
            mask=mask,
        )
        if prev_points is None or len(prev_points) < self.runtime_settings.task2_flow_min_points:
            return 0.0, 0.0, {"point_count": 0, "confidence": 0.0}

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_frame.gray,
            decoded_frame.gray,
            prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if next_points is None or status is None:
            return 0.0, 0.0, {"point_count": 0, "confidence": 0.0}

        valid = status.reshape(-1) == 1
        if not np.any(valid):
            return 0.0, 0.0, {"point_count": 0, "confidence": 0.0}

        prev_xy = prev_points.reshape(-1, 2)[valid]
        next_xy = next_points.reshape(-1, 2)[valid]
        deltas = next_xy - prev_xy
        point_count = int(len(deltas))
        if point_count < self.runtime_settings.task2_flow_min_points:
            return 0.0, 0.0, {"point_count": point_count, "confidence": 0.0}

        median_delta = np.median(deltas, axis=0)
        confidence = clamp_ratio(point_count / float(max(self.runtime_settings.task2_flow_primary_points, 1)))
        return float(median_delta[0]), float(median_delta[1]), {
            "point_count": point_count,
            "confidence": round(float(confidence), 4),
        }

    def _estimate_z_delta(self, previous_frame: DecodedFrame | None, decoded_frame: DecodedFrame) -> float:
        if (
            not is_cv2_available()
            or previous_frame is None
            or previous_frame.gray is None
            or decoded_frame.gray is None
            or previous_frame.width != decoded_frame.width
            or previous_frame.height != decoded_frame.height
        ):
            return 0.0
        mean_prev = float(np.mean(previous_frame.gray))
        mean_curr = float(np.mean(decoded_frame.gray))
        return (mean_curr - mean_prev) / 255.0

    def _estimate_byte_shift(self, frame: FrameEnvelope) -> tuple[float, float, float]:
        frame_index = extract_frame_index(frame.frame_url)
        return (
            float((frame_index % 5) - 2) * 0.35,
            float((frame_index % 7) - 3) * 0.25,
            0.12,
        )


def clamp_ratio(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
