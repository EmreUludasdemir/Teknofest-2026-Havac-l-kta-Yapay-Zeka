from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame, FrameEnvelope
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches


@dataclass(slots=True)
class Task3Pipeline:
    """Task 3 baseline ve deneysel akislarini tek noktada toplar."""

    runtime_settings: MvpRuntimeSettings = field(default_factory=MvpRuntimeSettings)
    reference_cache: ReferenceCache = field(default_factory=ReferenceCache)
    baseline_matcher: Task3Matcher = field(init=False)
    experimental_pipeline: Any | None = field(init=False, default=None)
    experimental_init_error: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.reference_cache.list_ids():
            self.reference_cache.preload_from_directory(
                self.runtime_settings.task3_reference_dir,
                orb_features=self.runtime_settings.task3_orb_features,
            )
        self.baseline_matcher = Task3Matcher(
            reference_cache=self.reference_cache,
            runtime_settings=self.runtime_settings,
        )

    def process(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        *,
        decoded_frame: DecodedFrame | None = None,
    ) -> tuple[list[CanonicalUndefinedObject], dict[str, Any]]:
        requested_mode = self.runtime_settings.task3_experimental_mode
        if not self.runtime_settings.task3_experimental_enabled or requested_mode == "baseline":
            matches, diagnostics = self._run_baseline(frame, image_bytes, decoded_frame=decoded_frame)
            diagnostics.update(
                {
                    "requested_mode": requested_mode,
                    "active_mode": "baseline",
                    "experimental_enabled": False,
                }
            )
            return matches, diagnostics

        experimental = self._get_experimental_pipeline()
        if experimental is None:
            matches, diagnostics = self._run_baseline(frame, image_bytes, decoded_frame=decoded_frame)
            diagnostics.update(
                {
                    "requested_mode": requested_mode,
                    "active_mode": "baseline",
                    "experimental_enabled": True,
                    "experimental_init_failed": True,
                    "experimental_error": self.experimental_init_error,
                }
            )
            return matches, diagnostics

        try:
            matches, diagnostics = experimental.process(
                frame,
                image_bytes,
                decoded_frame=decoded_frame,
            )
            diagnostics.setdefault("requested_mode", requested_mode)
            diagnostics.setdefault("experimental_enabled", True)
            return matches, diagnostics
        except Exception as exc:
            matches, diagnostics = self._run_baseline(frame, image_bytes, decoded_frame=decoded_frame)
            diagnostics.update(
                {
                    "requested_mode": requested_mode,
                    "active_mode": "baseline",
                    "experimental_enabled": True,
                    "experimental_frame_failed": True,
                    "experimental_error": str(exc),
                }
            )
            return matches, diagnostics

    def reset_state(self) -> None:
        if self.experimental_pipeline is not None and hasattr(self.experimental_pipeline, "reset_state"):
            self.experimental_pipeline.reset_state()

    def force_track_loss(self) -> None:
        if self.experimental_pipeline is not None and hasattr(self.experimental_pipeline, "force_track_loss"):
            self.experimental_pipeline.force_track_loss()

    def _run_baseline(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        *,
        decoded_frame: DecodedFrame | None = None,
    ) -> tuple[list[CanonicalUndefinedObject], dict[str, Any]]:
        reference_ids = self.reference_cache.list_ids()
        matches = self.baseline_matcher.match(
            frame,
            image_bytes,
            reference_ids,
            decoded_frame=decoded_frame,
            mode="orb_template",
        )
        filtered = filter_no_match_candidates(
            matches,
            min_score=self.runtime_settings.task3_min_score,
            ambiguity_margin=self.runtime_settings.task3_ambiguity_margin,
        )
        verified = verify_matches(
            frame,
            filtered,
            decoded_frame=decoded_frame,
            min_inliers=self.runtime_settings.task3_match_min_inliers,
        )
        diagnostics = {
            "active_mode": "baseline",
            "integration_status": "baseline",
            "reference_count": len(reference_ids),
            "raw_candidate_count": len(matches),
            "filtered_candidate_count": len(filtered),
            "verified_candidate_count": len(verified),
        }
        return verified, diagnostics

    def _get_experimental_pipeline(self) -> Any | None:
        if self.experimental_pipeline is not None:
            return self.experimental_pipeline
        if self.experimental_init_error is not None:
            return None
        try:
            from src.task3.experimental.pipeline import ExperimentalTask3Pipeline

            self.experimental_pipeline = ExperimentalTask3Pipeline(
                reference_cache=self.reference_cache,
                runtime_settings=self.runtime_settings,
                baseline_matcher=self.baseline_matcher,
            )
            return self.experimental_pipeline
        except Exception as exc:
            self.experimental_init_error = str(exc)
            return None
