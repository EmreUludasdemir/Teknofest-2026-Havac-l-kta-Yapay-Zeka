from __future__ import annotations

from dataclasses import dataclass, field

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope, FrameResult
from src.core.vision import decode_image_bytes
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches


@dataclass(slots=True)
class MvpFrameProcessor:
    """Task 3-only frame processor used by the batch and sequential adapters."""

    runtime_settings: MvpRuntimeSettings = field(default_factory=MvpRuntimeSettings)
    reference_cache: ReferenceCache = field(init=False)
    task3_matcher: Task3Matcher = field(init=False)

    def __post_init__(self) -> None:
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
                "task3_status": "pending",
                "task3_info": {},
                "fallback_mode": None,
                "reference_cache_size": len(self.reference_cache.list_ids()),
                "modality": decoded_frame.modality,
                "image_size": [decoded_frame.height, decoded_frame.width],
            }
        )

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
                suppression_mode=self.reference_cache.get_candidate_suppression_mode(),
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
            result.diagnostics["task3_info"] = dict(getattr(self.task3_matcher, "last_run_info", {}))
            result.diagnostics["fallback_mode"] = "task3"

        return result


Task3OnlyProcessor = MvpFrameProcessor
