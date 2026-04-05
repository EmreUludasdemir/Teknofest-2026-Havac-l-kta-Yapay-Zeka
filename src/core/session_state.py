from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.frame_state import FrameEnvelope


@dataclass(slots=True)
class SessionState:
    session_name: str
    video_name: str
    frames: list[FrameEnvelope]
    protocol_name: str = "official-repo-batch-v1"
    raw_frames: list[dict[str, Any]] = field(default_factory=list)
    raw_translations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReplaySummary:
    session_name: str
    frames_seen: int = 0
    frames_submitted: int = 0
    validation_failures: int = 0
    submission_failures: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
