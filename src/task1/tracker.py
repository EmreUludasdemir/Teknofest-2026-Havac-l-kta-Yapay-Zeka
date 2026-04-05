from __future__ import annotations

from dataclasses import dataclass, field

from src.core.frame_state import CanonicalDetection, FrameEnvelope


@dataclass(slots=True)
class Task1Tracker:
    """Gorev 1 icin basit track state tutar."""

    last_centroids: dict[str, tuple[float, float]] = field(default_factory=dict)

    def update(self, frame: FrameEnvelope, detections: list[CanonicalDetection]) -> list[CanonicalDetection]:
        for detection in detections:
            track_key = str(detection.metadata.get("track_key", f"class-{detection.class_id}"))
            current_centroid = (
                (float(detection.top_left_x) + float(detection.bottom_right_x)) / 2.0,
                (float(detection.top_left_y) + float(detection.bottom_right_y)) / 2.0,
            )
            previous = self.last_centroids.get(track_key)
            detection.metadata["track_key"] = track_key
            detection.metadata["current_centroid"] = current_centroid
            detection.metadata["previous_centroid"] = previous
            self.last_centroids[track_key] = current_centroid
        return detections
