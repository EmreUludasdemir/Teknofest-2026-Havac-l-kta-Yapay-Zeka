from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.task2.estimator import Task2Estimator
from src.task2.health_logic import resolve_task2_translation


class Task2HealthTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = MvpRuntimeSettings()
        self.estimator = Task2Estimator(runtime_settings=self.settings)

    def _frame(self, frame_id: int, health: str, x: float, y: float, z: float) -> FrameEnvelope:
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
        return DecodedFrame(
            bgr=None,
            gray=None,
            width=640,
            height=512,
            channel_count=0,
            modality="thermal",
            frame_index=frame_id,
        )

    def test_reference_branch_bypasses_estimator(self) -> None:
        class FailingEstimateTask2Estimator(Task2Estimator):
            def estimate(self, frame, decoded_frame, *, dynamic_detections=None):  # type: ignore[override]
                raise AssertionError("estimate should not run")

        estimator = FailingEstimateTask2Estimator(runtime_settings=self.settings)
        translation, diagnostics = resolve_task2_translation(
            self._frame(1, "1", 2.0, 3.0, 4.0),
            self._decoded(1),
            estimator,
        )
        self.assertEqual(translation.source, "task2_reference")
        self.assertEqual(diagnostics["task2_branch"], "reference")

    def test_health_transition_preserves_continuity_and_resets_on_recovery(self) -> None:
        resolve_task2_translation(self._frame(1, "1", 1.0, 1.0, 5.0), self._decoded(1), self.estimator)
        resolve_task2_translation(self._frame(2, "1", 2.0, 2.0, 6.0), self._decoded(2), self.estimator)
        estimated, _ = resolve_task2_translation(self._frame(3, "0", 99.0, 99.0, 99.0), self._decoded(3), self.estimator)
        recovered, diagnostics = resolve_task2_translation(self._frame(4, "1", 3.0, 3.0, 7.0), self._decoded(4), self.estimator)
        self.assertNotEqual(estimated.source, "task2_reference")
        self.assertEqual(recovered.source, "task2_reference")
        self.assertEqual(diagnostics["task2_branch"], "reference")
        self.assertEqual(self.estimator.last_health_status, "1")

    def test_low_phase_response_falls_back_to_flow_branch(self) -> None:
        class FlowFallbackEstimator(Task2Estimator):
            def _estimate_phase_shift(self, previous_frame, decoded_frame, mask):  # type: ignore[override]
                return 0.0, 0.0, 0.01

            def _estimate_flow_shift(self, previous_frame, decoded_frame, mask):  # type: ignore[override]
                return 2.0, -1.0, {"point_count": 8, "confidence": 0.6}

        estimator = FlowFallbackEstimator(runtime_settings=self.settings)
        resolve_task2_translation(self._frame(1, "1", 10.0, 10.0, 5.0), self._decoded(1), estimator)
        resolve_task2_translation(self._frame(2, "1", 11.0, 11.0, 5.5), self._decoded(2), estimator)
        translation, diagnostics = resolve_task2_translation(
            self._frame(3, "0", 99.0, 99.0, 99.0),
            self._decoded(3),
            estimator,
        )
        self.assertEqual(diagnostics["estimation_mode"], "lucas_kanade")
        self.assertEqual(diagnostics["task2_branch"], "estimated")
        self.assertNotEqual(translation.source, "task2_reference")


if __name__ == "__main__":
    unittest.main()
