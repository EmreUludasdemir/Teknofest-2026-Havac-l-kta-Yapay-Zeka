from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.pipeline.mvp_processor import MvpFrameProcessor
from src.task2.estimator import Task2Estimator


class ExplodingTask2Estimator(Task2Estimator):
    def estimate(self, frame, decoded_frame, *, dynamic_detections=None):  # type: ignore[override]
        raise RuntimeError("forced-task2-error")


class Task2ProcessorFallbackTests(unittest.TestCase):
    def test_task2_exception_still_returns_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            runtime_settings = MvpRuntimeSettings(task3_reference_dir=temp_path)
            processor = MvpFrameProcessor(runtime_settings=runtime_settings)
            processor.task2_estimator = ExplodingTask2Estimator(runtime_settings=runtime_settings)

            frame = FrameEnvelope(
                frame_url="http://mock/frames/11/",
                image_url="/frame.jpg",
                video_name="session",
                translation_x=1.0,
                translation_y=2.0,
                translation_z=3.0,
                health_status="0",
            )
            result = processor(frame, b"img")

            self.assertEqual(result.diagnostics["task2_status"], "fallback")
            self.assertEqual(len(result.detected_translations), 1)
            self.assertEqual(result.detected_translations[0].source, "task2_zero_fallback")


if __name__ == "__main__":
    unittest.main()
