from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CanonicalTranslation:
    translation_x: float
    translation_y: float
    translation_z: float
    source: str = "estimated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation_x": self.translation_x,
            "translation_y": self.translation_y,
            "translation_z": self.translation_z,
            "source": self.source,
        }


@dataclass(slots=True)
class CanonicalDetection:
    class_id: int
    landing_status: int = -1
    motion_status: int | None = None
    top_left_x: float = 0.0
    top_left_y: float = 0.0
    bottom_right_x: float = 0.0
    bottom_right_y: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "landing_status": self.landing_status,
            "motion_status": self.motion_status,
            "top_left_x": self.top_left_x,
            "top_left_y": self.top_left_y,
            "bottom_right_x": self.bottom_right_x,
            "bottom_right_y": self.bottom_right_y,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CanonicalUndefinedObject:
    object_id: str
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "top_left_x": self.top_left_x,
            "top_left_y": self.top_left_y,
            "bottom_right_x": self.bottom_right_x,
            "bottom_right_y": self.bottom_right_y,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DecodedFrame:
    bgr: Any | None
    gray: Any | None
    width: int
    height: int
    channel_count: int
    modality: str
    frame_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrameEnvelope:
    frame_url: str
    image_url: str
    video_name: str
    translation_x: float
    translation_y: float
    translation_z: float
    health_status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def ground_truth_translation(self, source: str = "ground_truth") -> CanonicalTranslation:
        return CanonicalTranslation(
            translation_x=self.translation_x,
            translation_y=self.translation_y,
            translation_z=self.translation_z,
            source=source,
        )


@dataclass(slots=True)
class FrameResult:
    frame_url: str
    detected_objects: list[CanonicalDetection] = field(default_factory=list)
    detected_translations: list[CanonicalTranslation] = field(default_factory=list)
    # Gorev 3 alanlari canonical modelde tutulur, resmi repo wire formatina dokulmez.
    detected_undefined_objects: list[CanonicalUndefinedObject] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_url,
            "detected_objects": [item.to_dict() for item in self.detected_objects],
            "detected_translations": [item.to_dict() for item in self.detected_translations],
            "detected_undefined_objects": [item.to_dict() for item in self.detected_undefined_objects],
            "diagnostics": self.diagnostics,
        }
