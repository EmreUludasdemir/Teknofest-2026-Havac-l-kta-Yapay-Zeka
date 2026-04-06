from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame
from src.evaluation.task3_experimental import compare_task3_baseline_vs_experimental


class Task3ExperimentalEvalTests(unittest.TestCase):
    def test_comparison_runner_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            manifest = {
                "scenarios": [
                    {
                        "scenario_id": "synthetic_absent",
                        "video_path": "video.mp4",
                        "reference_mode": "synthetic_absent",
                        "frame_stride": 1,
                        "frame_limit": 3,
                        "target_expected": False,
                        "tags": ["absent_target"],
                    }
                ]
            }
            manifest_path = temp_path / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (temp_path / "video.mp4").write_bytes(b"video")

            fake_frames = [
                DecodedFrame(
                    bgr=None,
                    gray=None,
                    width=640,
                    height=512,
                    channel_count=0,
                    modality="rgb",
                    frame_index=index,
                )
                for index in range(3)
            ]

            with patch("src.evaluation.task3_experimental.iter_video_frames", return_value=iter(fake_frames)):
                payload = compare_task3_baseline_vs_experimental(
                    manifest_path=manifest_path,
                    runtime_settings=MvpRuntimeSettings(),
                    output_dir=temp_path,
                )

            self.assertIn("comparison", payload)
            self.assertTrue((temp_path / "task3_baseline_vs_experimental.json").exists())
            self.assertTrue((temp_path / "task3_baseline_vs_experimental.md").exists())
            self.assertTrue((temp_path / "task3_experimental_design.md").exists())
            self.assertTrue((temp_path / "task3_integration_risks.md").exists())


if __name__ == "__main__":
    unittest.main()
