from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.task2.estimator import Task2Estimator
from src.task2.health_logic import resolve_task2_translation


class Task2LongSequenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = Task2Estimator(runtime_settings=MvpRuntimeSettings())

    def _frame(self, frame_id: int, health: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> FrameEnvelope:
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

    def test_long_health_zero_sequence_stays_bounded(self) -> None:
        resolve_task2_translation(self._frame(1, "1", 5.0, 5.0, 10.0), self._decoded(1), self.estimator)
        resolve_task2_translation(self._frame(2, "1", 6.0, 6.0, 10.5), self._decoded(2), self.estimator)
        outputs = []
        for frame_id in range(3, 12):
            translation, diagnostics = resolve_task2_translation(self._frame(frame_id, "0"), self._decoded(frame_id), self.estimator)
            outputs.append((translation, diagnostics))
        last_translation, last_diagnostics = outputs[-1]
        self.assertLessEqual(abs(last_translation.translation_x - 6.0), 5.0)
        self.assertLessEqual(abs(last_translation.translation_y - 6.0), 5.0)
        self.assertIn("confidence", last_diagnostics)
        self.assertTrue(all(item[0].source != "task2_reference" for item in outputs))


if __name__ == "__main__":
    unittest.main()
