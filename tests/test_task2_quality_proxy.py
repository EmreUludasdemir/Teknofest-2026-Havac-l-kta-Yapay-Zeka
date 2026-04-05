from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame
from src.core.vision import is_cv2_available
from src.evaluation.task2_long_sequence import Task2CsvRecord, evaluate_records, evaluate_task2_long_sequences

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import np
else:  # pragma: no cover - cv2 yoksa
    np = None


@unittest.skipUnless(is_cv2_available(), "Task2 quality proxy tests require cv2")
class Task2QualityProxyTests(unittest.TestCase):
    def _decoded(self, frame_index: int) -> DecodedFrame:
        image = np.zeros((64, 64), dtype=np.uint8)
        image[16:32, 16 + frame_index : 32 + frame_index] = 255
        return DecodedFrame(
            bgr=None,
            gray=image,
            width=64,
            height=64,
            channel_count=1,
            modality="thermal",
            frame_index=frame_index,
        )

    def test_evaluate_records_reports_reference_and_estimated_periods_separately(self) -> None:
        records = [
            Task2CsvRecord("frame_000000", 0.0, 0.0, 1.0, "1"),
            Task2CsvRecord("frame_000001", 0.1, 0.0, 1.0, "1"),
            Task2CsvRecord("frame_000002", 0.2, 0.1, 1.0, "0"),
            Task2CsvRecord("frame_000003", 0.3, 0.1, 1.0, "0"),
            Task2CsvRecord("frame_000004", 0.4, 0.2, 1.0, "1"),
        ]
        decoded_frames = [self._decoded(index) for index in range(len(records))]
        summary = evaluate_records(
            records,
            decoded_frames,
            runtime_settings=MvpRuntimeSettings(),
            sequence_name="mini-seq",
            video_name="mini-seq",
        )
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["reference_frames"], 3)
        self.assertEqual(summary["estimated_frames"], 2)
        self.assertIn("continuity_error", summary)
        self.assertIn("recovery_error_after_health_returns_to_1", summary)

    def test_missing_video_is_reported_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            csv_path = temp_root / "Example-RGB-translation.csv"
            csv_path.write_text(
                "translation_x,translation_y,translation_z,frame_numbers\n"
                "0.0,0.0,1.0,frame_000000\n"
                "0.1,0.0,1.0,frame_000001\n",
                encoding="utf-8",
            )
            payload = evaluate_task2_long_sequences(
                runtime_settings=MvpRuntimeSettings(task2_eval_sequence_limit=2),
                root_dir=temp_root,
                output_dir=temp_root,
            )
            self.assertEqual(payload["results"][0]["status"], "missing_video")
            self.assertTrue((temp_root / "task2_long_sequence_summary.json").exists())
            self.assertTrue((temp_root / "task2_long_sequence_table.md").exists())


if __name__ == "__main__":
    unittest.main()
