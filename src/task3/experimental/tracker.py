from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.frame_state import CanonicalUndefinedObject


@dataclass(slots=True)
class TrackState:
    object_id: str
    bbox: tuple[float, float, float, float]
    confidence: float
    lost_count: int = 0
    last_seen_frame: int = -1
    source: str = "detector"


@dataclass(slots=True)
class ExperimentalTask3Tracker:
    """Task 3 icin hafif, bounded takip yardimcisi."""

    mode: str = "off"
    lost_patience: int = 3
    round_robin_index: int = 0
    states: dict[str, TrackState] = field(default_factory=dict)

    def effective_mode(self) -> str:
        if self.mode in {"off", "light", "botsort", "samurai_like"}:
            return "light" if self.mode in {"botsort", "samurai_like"} else self.mode
        return "off"

    def select_reference_ids(
        self,
        reference_ids: list[str],
        *,
        max_per_frame: int,
        frame_index: int,
        fullframe_redetect_every_n: int,
    ) -> list[str]:
        if not reference_ids:
            return []
        tracked_ids = [
            object_id
            for object_id, state in self.states.items()
            if state.confidence > 0.0 and state.lost_count < self.lost_patience
        ]
        selected: list[str] = []
        if frame_index % max(fullframe_redetect_every_n, 1) == 0:
            selected.extend(tracked_ids)
        if len(selected) < max_per_frame:
            remaining = [item for item in reference_ids if item not in selected]
            if remaining:
                start = self.round_robin_index % len(remaining)
                ordered = remaining[start:] + remaining[:start]
                selected.extend(ordered[: max(0, max_per_frame - len(selected))])
                self.round_robin_index = (self.round_robin_index + 1) % max(len(remaining), 1)
        return selected[:max_per_frame]

    def build_track_candidates(self, *, frame_index: int) -> list[CanonicalUndefinedObject]:
        if self.effective_mode() == "off":
            return []
        candidates: list[CanonicalUndefinedObject] = []
        for state in self.states.values():
            if state.lost_count >= self.lost_patience or state.confidence <= 0.0:
                continue
            decayed_confidence = max(0.0, state.confidence - (0.12 * state.lost_count))
            if decayed_confidence < 0.25:
                continue
            candidates.append(
                CanonicalUndefinedObject(
                    object_id=state.object_id,
                    top_left_x=state.bbox[0],
                    top_left_y=state.bbox[1],
                    bottom_right_x=state.bbox[2],
                    bottom_right_y=state.bbox[3],
                    metadata={
                        "matcher_source": "task3_experimental_track",
                        "candidate_name": f"tracker_{self.effective_mode()}",
                        "tracking_source": state.source,
                        "tracking_lost_count": state.lost_count,
                        "tracking_score": round(decayed_confidence, 4),
                        "match_score": round(decayed_confidence, 4),
                        "frame_index": frame_index,
                    },
                )
            )
        return candidates

    def update_after_frame(
        self,
        *,
        frame_index: int,
        searched_reference_ids: list[str],
        accepted_matches: list[CanonicalUndefinedObject],
    ) -> None:
        accepted_map = {item.object_id: item for item in accepted_matches}
        for object_id in searched_reference_ids:
            accepted = accepted_map.get(object_id)
            if accepted is None:
                self.mark_reference_missed(object_id)
                continue
            self.states[object_id] = TrackState(
                object_id=object_id,
                bbox=(
                    float(accepted.top_left_x),
                    float(accepted.top_left_y),
                    float(accepted.bottom_right_x),
                    float(accepted.bottom_right_y),
                ),
                confidence=float(accepted.metadata.get("tracking_score", accepted.metadata.get("match_score", 0.75))),
                lost_count=0,
                last_seen_frame=frame_index,
                source=str(accepted.metadata.get("candidate_name", "detector")),
            )

    def mark_reference_missed(self, object_id: str) -> None:
        state = self.states.get(object_id)
        if state is None:
            return
        state.lost_count += 1
        state.confidence = max(0.0, state.confidence - 0.2)

    def reset(self) -> None:
        self.states.clear()
        self.round_robin_index = 0

    def force_track_loss(self) -> None:
        for state in self.states.values():
            state.lost_count = self.lost_patience
            state.confidence = 0.0
