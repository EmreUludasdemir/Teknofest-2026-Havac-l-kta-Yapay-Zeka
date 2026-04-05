from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope
from src.task1.detector import Task1Detector
from src.task1.landing_logic import assign_landing_status
from src.task1.motion_logic import assign_motion_status
from src.task1.postprocess import deduplicate_detections
from src.task1.tracker import Task1Tracker


class Task1BackendInterfaceTests(unittest.TestCase):
    def _frame(self) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url="http://mock/frames/4/",
            image_url="/frame.jpg",
            video_name="session",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
        )

    def test_synthetic_backend_is_default(self) -> None:
        detector = Task1Detector(runtime_settings=MvpRuntimeSettings(task1_detector_backend="synthetic"))
        detections = detector.detect(self._frame(), b"img")
        self.assertTrue(all(item.metadata.get("backend_name") == "synthetic" for item in detections))

    def test_external_backend_falls_back_without_breaking_downstream_chain(self) -> None:
        settings = MvpRuntimeSettings(task1_detector_backend="yolo11n")
        detector = Task1Detector(runtime_settings=settings)
        tracker = Task1Tracker()
        detections = detector.detect(self._frame(), b"img")
        detections = tracker.update(self._frame(), detections)
        detections = assign_motion_status(detections, self._frame(), threshold_px=settings.task1_motion_threshold_px)
        detections = assign_landing_status(detections, self._frame(), margin_px=settings.task1_landing_margin_px)
        detections = deduplicate_detections(detections)
        self.assertTrue(any(item.metadata.get("backend_unavailable") for item in detections))
        self.assertTrue(all(item.metadata.get("backend_name") == "yolo11n" for item in detections))
        self.assertTrue(any("missing_model_path" in str(item.metadata.get("backend_error", "")) for item in detections))


if __name__ == "__main__":
    unittest.main()
