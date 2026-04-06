from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame, FrameEnvelope
from src.task3.experimental.detector import ExperimentalPromptDetector
from src.task3.experimental.fusion import fuse_candidates
from src.task3.experimental.tracker import ExperimentalTask3Tracker
from src.task3.experimental.verifier import ExperimentalVerifier
from src.task3.matcher import Task3Matcher
from src.task3.reference_cache import ReferenceCache


@dataclass(slots=True)
class ExperimentalTask3Pipeline:
    """Config-gated deneysel Task 3 akisi."""

    reference_cache: ReferenceCache
    runtime_settings: MvpRuntimeSettings
    baseline_matcher: Task3Matcher
    detector: ExperimentalPromptDetector = field(init=False)
    tracker: ExperimentalTask3Tracker = field(init=False)
    verifier: ExperimentalVerifier = field(init=False)
    frame_counter: int = field(init=False, default=0)
    reject_streaks: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.detector = ExperimentalPromptDetector(
            runtime_settings=self.runtime_settings,
            reference_cache=self.reference_cache,
            baseline_matcher=self.baseline_matcher,
        )
        self.tracker = ExperimentalTask3Tracker(
            mode=self.runtime_settings.task3_experimental_tracking,
            lost_patience=self.runtime_settings.task3_experimental_track_lost_patience,
        )
        self.verifier = ExperimentalVerifier(
            runtime_settings=self.runtime_settings,
            reference_cache=self.reference_cache,
        )

    def process(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        *,
        decoded_frame: DecodedFrame | None = None,
    ) -> tuple[list[CanonicalUndefinedObject], dict[str, Any]]:
        del image_bytes
        if decoded_frame is None:
            raise RuntimeError("task3_experimental_requires_decoded_frame")

        reference_ids = self.reference_cache.list_ids()
        frame_index = int(frame.metadata.get("frame_index", decoded_frame.frame_index))
        if not reference_ids:
            return [], {
                "requested_mode": self.runtime_settings.task3_experimental_mode,
                "active_mode": "experimental",
                "integration_status": "no_references",
                "frame_index": frame_index,
                "searched_reference_ids": [],
                "accepted_candidate_count": 0,
            }

        track_candidates = self.tracker.build_track_candidates(frame_index=frame_index)
        searched_reference_ids = self.tracker.select_reference_ids(
            reference_ids,
            max_per_frame=max(self.runtime_settings.task3_experimental_max_references_per_frame, 1),
            frame_index=frame_index,
            fullframe_redetect_every_n=self.runtime_settings.task3_experimental_fullframe_redetect_every_n,
        )
        detector_candidates, detector_diag = self.detector.search(frame, decoded_frame, searched_reference_ids)
        accepted, accept_diag = self._accept_candidates(
            frame,
            decoded_frame,
            track_candidates + detector_candidates,
        )

        unique_searched = list(dict.fromkeys(searched_reference_ids))
        redetect_triggered = False
        redetect_reason = ""
        redetect_diag: dict[str, Any] = {}
        if self._should_redetect(unique_searched, accepted):
            redetect_triggered = True
            redetect_reason = self._redetect_reason()
            redetect_ids = self._select_redetect_reference_ids(reference_ids, unique_searched)
            if redetect_ids:
                extra_candidates, extra_detector_diag = self.detector.search(frame, decoded_frame, redetect_ids)
                extra_accepted, extra_accept_diag = self._accept_candidates(
                    frame,
                    decoded_frame,
                    extra_candidates,
                )
                redetect_diag = {
                    "searched_reference_ids": list(redetect_ids),
                    "detector": extra_detector_diag,
                    "accept": extra_accept_diag,
                }
                if extra_accepted:
                    accepted = extra_accepted
                unique_searched = list(dict.fromkeys(unique_searched + redetect_ids))

        self._update_reject_streaks(unique_searched, accepted)
        self.tracker.update_after_frame(
            frame_index=frame_index,
            searched_reference_ids=unique_searched,
            accepted_matches=accepted,
        )
        self.frame_counter += 1

        verifier_mode, verifier_exact = self.verifier.effective_mode()
        diagnostics = {
            "requested_mode": self.runtime_settings.task3_experimental_mode,
            "active_mode": "experimental",
            "integration_status": detector_diag.get("integration_status", "prompt_approx"),
            "frame_index": frame_index,
            "reference_count": len(reference_ids),
            "searched_reference_ids": unique_searched,
            "track_candidate_count": len(track_candidates),
            "detector_candidate_count": len(detector_candidates),
            "accepted_candidate_count": len(accepted),
            "tracker_requested_mode": self.runtime_settings.task3_experimental_tracking,
            "tracker_active_mode": self.tracker.effective_mode(),
            "verifier_requested_mode": self.runtime_settings.task3_experimental_verifier,
            "verifier_active_mode": verifier_mode,
            "verifier_exact": verifier_exact,
            "detector": detector_diag,
            "accept": accept_diag,
            "redetect_triggered": redetect_triggered,
            "redetect_reason": redetect_reason or None,
            "redetect": redetect_diag,
            "tiled_inference_enabled": self.runtime_settings.task3_experimental_tiled_inference,
            "multiscale_enabled": self.runtime_settings.task3_experimental_multiscale,
            "reject_streaks": dict(self.reject_streaks),
        }
        return accepted, diagnostics

    def reset_state(self) -> None:
        self.frame_counter = 0
        self.reject_streaks.clear()
        self.tracker.reset()

    def force_track_loss(self) -> None:
        self.tracker.force_track_loss()

    def _accept_candidates(
        self,
        frame: FrameEnvelope,
        decoded_frame: DecodedFrame,
        candidates: list[CanonicalUndefinedObject],
    ) -> tuple[list[CanonicalUndefinedObject], dict[str, Any]]:
        verified, verifier_diag = self.verifier.verify(frame, decoded_frame, candidates)
        fused = fuse_candidates(
            verified,
            min_score=self.runtime_settings.task3_min_score,
            ambiguity_margin=self.runtime_settings.task3_ambiguity_margin,
        )
        return fused, {
            "input_candidate_count": len(candidates),
            "verified_candidate_count": len(verified),
            "fused_candidate_count": len(fused),
            "verifier": verifier_diag,
        }

    def _should_redetect(
        self,
        searched_reference_ids: list[str],
        accepted: list[CanonicalUndefinedObject],
    ) -> bool:
        if accepted:
            return False
        if not searched_reference_ids:
            return False
        return self._tracker_collapsed() or any(count >= 2 for count in self.reject_streaks.values())

    def _redetect_reason(self) -> str:
        if self._tracker_collapsed():
            return "tracker_collapsed"
        if any(count >= 2 for count in self.reject_streaks.values()):
            return "repeated_verifier_reject"
        return "periodic"

    def _tracker_collapsed(self) -> bool:
        for state in self.tracker.states.values():
            if state.lost_count >= self.tracker.lost_patience or state.confidence <= 0.15:
                return True
        return False

    def _select_redetect_reference_ids(
        self,
        reference_ids: list[str],
        searched_reference_ids: list[str],
    ) -> list[str]:
        remaining = [reference_id for reference_id in reference_ids if reference_id not in searched_reference_ids]
        ordered = remaining if remaining else reference_ids
        return ordered[: max(self.runtime_settings.task3_experimental_max_references_per_frame, 1)]

    def _update_reject_streaks(
        self,
        searched_reference_ids: list[str],
        accepted: list[CanonicalUndefinedObject],
    ) -> None:
        accepted_ids = {item.object_id for item in accepted}
        for reference_id in searched_reference_ids:
            if reference_id in accepted_ids:
                self.reject_streaks[reference_id] = 0
            else:
                self.reject_streaks[reference_id] = self.reject_streaks.get(reference_id, 0) + 1

