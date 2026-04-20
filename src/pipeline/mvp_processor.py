from __future__ import annotations

from dataclasses import dataclass, field

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalTranslation, FrameEnvelope, FrameResult
from src.core.vision import decode_image_bytes
from src.task1.detector import Task1Detector
from src.task1.landing_logic import assign_landing_status
from src.task1.motion_logic import assign_motion_status
from src.task1.postprocess import deduplicate_detections
from src.task1.tracker import Task1Tracker
from src.task2.estimator import Task2Estimator
from src.task2.health_logic import resolve_task2_translation
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches


@dataclass(slots=True)
class MvpFrameProcessor:
    """Faz 3 MVP gorevlerini tek stateful islemcide toplar."""

    runtime_settings: MvpRuntimeSettings = field(default_factory=MvpRuntimeSettings)
    task1_detector: Task1Detector = field(init=False)
    task1_tracker: Task1Tracker = field(init=False)
    task2_estimator: Task2Estimator = field(init=False)
    reference_cache: ReferenceCache = field(init=False)
    task3_matcher: Task3Matcher = field(init=False)
    last_task1_dynamic_detections: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task1_detector = Task1Detector(runtime_settings=self.runtime_settings)
        self.task1_tracker = Task1Tracker()
        self.task2_estimator = Task2Estimator(runtime_settings=self.runtime_settings)
        self.reference_cache = ReferenceCache()
        self.reference_cache.preload_from_directory(
            self.runtime_settings.task3_reference_dir,
            orb_features=self.runtime_settings.task3_orb_features,
        )
        self.task3_matcher = Task3Matcher(
            reference_cache=self.reference_cache,
            runtime_settings=self.runtime_settings,
        )

    def __call__(self, frame: FrameEnvelope, image_bytes: bytes) -> FrameResult:
        decoded_frame = decode_image_bytes(frame, image_bytes)
        result = FrameResult(frame_url=frame.frame_url)
        result.diagnostics.update(
            {
                "task1_status": "pending",
                "task2_status": "pending",
                "task3_status": "pending",
                "task3_info": {},
                "fallback_mode": None,
                "reference_cache_size": len(self.reference_cache.list_ids()),
                "modality": decoded_frame.modality,
                "image_size": [decoded_frame.height, decoded_frame.width],
            }
        )

        # Task 2 once islenir; Task 1 ve Task 3 bundan bagimsiz hata izolesine sahiptir.
        try:
            translation, task2_info = resolve_task2_translation(
                frame,
                decoded_frame,
                self.task2_estimator,
                dynamic_detections=self.last_task1_dynamic_detections,
            )
            result.detected_translations.append(translation)
            result.diagnostics["task2_status"] = "ok"
            result.diagnostics["task2_info"] = task2_info
        except Exception as exc:
            fallback = self._task2_fallback(frame, decoded_frame)
            result.detected_translations.append(fallback)
            result.diagnostics["task2_status"] = "fallback"
            result.diagnostics["task2_error"] = str(exc)
            result.diagnostics["fallback_mode"] = "task2"

        try:
            detections = self.task1_detector.detect(frame, image_bytes, decoded_frame=decoded_frame)
            detections = self.task1_tracker.update(frame, detections)
            detections = assign_motion_status(
                detections,
                frame,
                threshold_px=self.runtime_settings.task1_motion_threshold_px,
            )
            detections = assign_landing_status(
                detections,
                frame,
                decoded_frame=decoded_frame,
                margin_px=self.runtime_settings.task1_landing_margin_px,
            )
            detections = deduplicate_detections(
                detections,
                max_objects_per_frame=self.runtime_settings.max_objects_per_frame,
            )
            result.detected_objects.extend(detections)
            self.last_task1_dynamic_detections = list(detections)
            result.diagnostics["task1_status"] = "ok"
        except Exception as exc:
            result.detected_objects = []
            result.diagnostics["task1_status"] = "fallback"
            result.diagnostics["task1_error"] = str(exc)
            if result.diagnostics["fallback_mode"] is None:
                result.diagnostics["fallback_mode"] = "task1"

        try:
            raw_matches = self.task3_matcher.match(
                frame,
                image_bytes,
                self.reference_cache.list_ids(),
                decoded_frame=decoded_frame,
                mode=self.runtime_settings.task3_mode,
            )
            matches = filter_no_match_candidates(
                raw_matches,
                min_score=self.runtime_settings.task3_min_score,
                mode=self.runtime_settings.task3_mode,
                yoloe_min_score=self.runtime_settings.task3_yoloe_min_score,
                modality=decoded_frame.modality,
                yoloe_thermal_min_score=self.runtime_settings.task3_yoloe_thermal_min_score,
                ambiguity_margin=self.runtime_settings.task3_ambiguity_margin,
            )
            verified_matches = verify_matches(
                frame,
                matches,
                decoded_frame=decoded_frame,
                min_inliers=self.runtime_settings.task3_match_min_inliers,
            )
            result.detected_undefined_objects.extend(verified_matches)
            task3_info = dict(self.task3_matcher.last_run_info)
            generated_count = int(task3_info.get("candidates_generated", 0) or len(raw_matches))
            task3_info["candidates_generated"] = generated_count
            task3_info["candidates_accepted"] = len(verified_matches)
            task3_info["candidates_rejected"] = max(generated_count - len(verified_matches), 0)
            result.diagnostics["task3_info"] = task3_info
            result.diagnostics["task3_status"] = "ok"
        except Exception as exc:
            result.detected_undefined_objects = []
            result.diagnostics["task3_status"] = "fallback"
            result.diagnostics["task3_error"] = str(exc)
            result.diagnostics["task3_info"] = dict(self.task3_matcher.last_run_info)
            if result.diagnostics["fallback_mode"] is None:
                result.diagnostics["fallback_mode"] = "task3"

        return result

    def _task2_fallback(self, frame: FrameEnvelope, decoded_frame) -> CanonicalTranslation:
        if str(frame.health_status) == "1":
            reference = frame.ground_truth_translation(source="task2_reference_fallback")
            self.task2_estimator.observe_reference(reference, decoded_frame=decoded_frame)
            return reference
        if self.task2_estimator.last_output_translation is not None:
            return CanonicalTranslation(
                translation_x=self.task2_estimator.last_output_translation.translation_x,
                translation_y=self.task2_estimator.last_output_translation.translation_y,
                translation_z=self.task2_estimator.last_output_translation.translation_z,
                source="task2_last_output_fallback",
            )
        if self.task2_estimator.last_reliable_translation is not None:
            return CanonicalTranslation(
                translation_x=self.task2_estimator.last_reliable_translation.translation_x,
                translation_y=self.task2_estimator.last_reliable_translation.translation_y,
                translation_z=self.task2_estimator.last_reliable_translation.translation_z,
                source="task2_last_reliable_fallback",
            )
        return CanonicalTranslation(
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            source="task2_zero_fallback",
        )
