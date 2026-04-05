from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalTranslation, DecodedFrame, FrameEnvelope
from src.task2.drift_control import apply_drift_guard
from src.task2.estimator import Task2Estimator
from src.task2.health_logic import resolve_task2_translation


class Task2MvpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = MvpRuntimeSettings()
        self.estimator = Task2Estimator(runtime_settings=self.settings)
        self.image_bytes = b"img"

    def _frame(self, frame_id: int, health_status: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url=f"http://mock/frames/{frame_id}/",
            image_url="/frame.jpg",
            video_name="session",
            translation_x=x,
            translation_y=y,
            translation_z=z,
            health_status=health_status,
        )

    def _decoded(self, frame_id: int) -> DecodedFrame:
        return DecodedFrame(
            bgr=None,
            gray=None,
            width=640,
            height=512,
            channel_count=0,
            modality="thermal",
            frame_index=frame_id,
        )

    def test_health_one_uses_reference_directly(self) -> None:
        frame = self._frame(1, "1", 10.0, 20.0, 30.0)
        translation, diagnostics = resolve_task2_translation(frame, self._decoded(1), self.estimator)
        self.assertEqual(translation.source, "task2_reference")
        self.assertEqual((translation.translation_x, translation.translation_y, translation.translation_z), (10.0, 20.0, 30.0))
        self.assertEqual(diagnostics["task2_branch"], "reference")

    def test_health_zero_uses_estimator_after_reference(self) -> None:
        resolve_task2_translation(self._frame(1, "1", 10.0, 10.0, 5.0), self._decoded(1), self.estimator)
        resolve_task2_translation(self._frame(2, "1", 12.0, 12.0, 6.0), self._decoded(2), self.estimator)
        translation, diagnostics = resolve_task2_translation(self._frame(3, "0", 99.0, 99.0, 99.0), self._decoded(3), self.estimator)
        self.assertNotEqual(translation.source, "task2_reference")
        self.assertEqual(diagnostics["task2_branch"], "estimated")
        self.assertTrue(isinstance(translation, CanonicalTranslation))

    def test_drift_guard_clamps_large_jump(self) -> None:
        previous = CanonicalTranslation(0.0, 0.0, 0.0, source="prev")
        candidate = CanonicalTranslation(20.0, -20.0, 10.0, source="candidate")
        guarded, diagnostics = apply_drift_guard(
            candidate,
            previous,
            previous,
            confidence=0.9,
            max_step_xy=5.0,
            max_step_z=2.0,
        )
        self.assertEqual((guarded.translation_x, guarded.translation_y, guarded.translation_z), (5.0, -5.0, 2.0))
        self.assertTrue(diagnostics["drift_clamped"])

    def test_zero_fallback_when_no_reliable_state_exists(self) -> None:
        translation, diagnostics = resolve_task2_translation(self._frame(7, "0"), self._decoded(7), self.estimator)
        self.assertEqual(translation.source, "task2_zero_fallback")
        self.assertEqual(diagnostics["task2_branch"], "estimated")


if __name__ == "__main__":
    unittest.main()
