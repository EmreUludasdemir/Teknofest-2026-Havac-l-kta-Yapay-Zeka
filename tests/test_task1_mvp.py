from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalDetection, FrameEnvelope
from src.task1.detector import Task1Detector
from src.task1.landing_logic import assign_landing_status
from src.task1.motion_logic import assign_motion_status
from src.task1.postprocess import deduplicate_detections
from src.task1.tracker import Task1Tracker


class Task1MvpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = MvpRuntimeSettings()
        self.detector = Task1Detector(runtime_settings=self.settings)
        self.tracker = Task1Tracker()
        self.image_bytes = b"img"

    def _frame(self, frame_id: int) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url=f"http://mock/frames/{frame_id}/",
            image_url="/frame.jpg",
            video_name="session",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
        )

    def test_detector_is_deterministic(self) -> None:
        first = self.detector.detect(self._frame(4), self.image_bytes)
        second = self.detector.detect(self._frame(4), self.image_bytes)
        self.assertEqual([item.to_dict() for item in first], [item.to_dict() for item in second])

    def test_motion_state_changes_after_tracker_updates(self) -> None:
        detections_1 = self.tracker.update(self._frame(1), self.detector.detect(self._frame(1), self.image_bytes))
        assign_motion_status(detections_1, self._frame(1), threshold_px=self.settings.task1_motion_threshold_px)
        detections_2 = self.tracker.update(self._frame(2), self.detector.detect(self._frame(2), self.image_bytes))
        assign_motion_status(detections_2, self._frame(2), threshold_px=self.settings.task1_motion_threshold_px)
        vehicle = next(item for item in detections_2 if item.class_id == 0)
        self.assertEqual(vehicle.motion_status, 1)

    def test_landing_logic_marks_overlap_as_not_suitable(self) -> None:
        detections = [
            CanonicalDetection(2, top_left_x=100.0, top_left_y=100.0, bottom_right_x=180.0, bottom_right_y=180.0),
            CanonicalDetection(0, top_left_x=110.0, top_left_y=110.0, bottom_right_x=150.0, bottom_right_y=150.0),
        ]
        assign_landing_status(detections, self._frame(8), margin_px=self.settings.task1_landing_margin_px)
        self.assertEqual(detections[0].landing_status, 0)

    def test_dedup_and_limit_apply(self) -> None:
        detections = self.detector.detect(self._frame(10), self.image_bytes)
        deduped = deduplicate_detections(detections, max_objects_per_frame=2)
        self.assertLessEqual(len(deduped), 2)


if __name__ == "__main__":
    unittest.main()
