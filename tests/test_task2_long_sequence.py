from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.evaluation.task2_long_sequence import assess_phase10_task2_stability
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

    def test_thermal_health_zero_logs_guard_and_transition_metadata(self) -> None:
        reference_1 = resolve_task2_translation(self._frame(1, "1", 4.0, 4.0, 9.5), self._decoded(1), self.estimator)
        estimated = resolve_task2_translation(self._frame(2, "0", 100.0, 100.0, 100.0), self._decoded(2), self.estimator)
        reference_2 = resolve_task2_translation(self._frame(3, "1", 4.5, 4.5, 9.6), self._decoded(3), self.estimator)

        self.assertEqual(reference_1[1]["task2_branch"], "reference")
        self.assertEqual(reference_2[1]["task2_branch"], "reference")
        self.assertEqual(estimated[1]["task2_branch"], "estimated")
        self.assertTrue(estimated[1]["thermal_guard_active"])
        self.assertIn("health_window_id", estimated[1])
        self.assertIn("sensor_hint_weight", estimated[1])
        self.assertLessEqual(
            float(estimated[1]["sensor_hint_weight"]),
            self.estimator.runtime_settings.task2_sensor_hint_max_weight_thermal,
        )

    def test_phase10_assessment_marks_observability_only_when_drift_not_improved(self) -> None:
        assessment = assess_phase10_task2_stability(
            {
                "health0_drift_accumulation": 5.0,
                "recovery_error_after_health_returns_to_1": 0.0,
            },
            {
                "health0_drift_accumulation": 5.5,
                "recovery_error_after_health_returns_to_1": 0.0,
                "phase9_stability": {
                    "thermal_guard_frames_mean": 3.0,
                    "hold_mode_frames_mean": 2.0,
                },
            },
        )
        self.assertEqual(assessment["conclusion"], "observability_only")


if __name__ == "__main__":
    unittest.main()
