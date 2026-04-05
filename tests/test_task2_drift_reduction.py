from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.task2.estimator import Task2Estimator
from src.task2.health_logic import resolve_task2_translation


class Task2DriftReductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = MvpRuntimeSettings()
        self.estimator = Task2Estimator(runtime_settings=self.settings)

    def _frame(self, frame_id: int, health: str, x: float = 0.0, y: float = 0.0, z: float = 5.0) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url=f"http://mock/frames/{frame_id}/",
            image_url="/frame.jpg",
            video_name="session",
            translation_x=x,
            translation_y=y,
            translation_z=z,
            health_status=health,
        )

    def _decoded(self, frame_id: int) -> DecodedFrame:
        return DecodedFrame(None, None, 640, 512, 0, "thermal", frame_id)

    def test_anchor_estimator_stays_bounded_during_long_health_zero_period(self) -> None:
        resolve_task2_translation(self._frame(1, "1", 2.0, 2.0, 6.0), self._decoded(1), self.estimator)
        resolve_task2_translation(self._frame(2, "1", 2.2, 2.1, 6.1), self._decoded(2), self.estimator)
        outputs = []
        for frame_id in range(3, 24):
            translation, diagnostics = resolve_task2_translation(self._frame(frame_id, "0"), self._decoded(frame_id), self.estimator)
            outputs.append((translation, diagnostics))
        self.assertTrue(all(item[0].source != "task2_reference" for item in outputs))
        self.assertLessEqual(abs(outputs[-1][0].translation_x - 2.2), self.settings.task2_anchor_distance_limit_xy)
        self.assertLessEqual(abs(outputs[-1][0].translation_y - 2.1), self.settings.task2_anchor_distance_limit_xy)

    def test_low_confidence_hold_mode_prefers_previous_state(self) -> None:
        resolve_task2_translation(self._frame(1, "1", 1.0, 1.0, 5.0), self._decoded(1), self.estimator)
        self.estimator.anchor_calibration_profile = None
        self.estimator.last_confidence = 0.05
        translation, diagnostics = resolve_task2_translation(self._frame(2, "0"), self._decoded(2), self.estimator)
        self.assertIn(diagnostics["estimation_mode"], {"hold_mode", "hold_mode_modality_mismatch"})
        self.assertIn(translation.source, {"task2_estimated", "task2_estimated_low_confidence"})


if __name__ == "__main__":
    unittest.main()
