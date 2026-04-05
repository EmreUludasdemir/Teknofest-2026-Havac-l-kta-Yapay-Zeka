from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.tools.profiling_harness import build_candidate_registry, evaluate_candidates


class ProfilingHarnessTests(unittest.TestCase):
    def test_candidate_registry_and_output_files_exist(self) -> None:
        self.assertGreaterEqual(len(build_candidate_registry()), 3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = evaluate_candidates(output_dir=tmp_dir, mode="smoke")
            summary_path = Path(tmp_dir) / "profiling_summary.json"
            table_path = Path(tmp_dir) / "profiling_table.md"
            self.assertTrue(summary_path.exists())
            self.assertTrue(table_path.exists())
            self.assertIn("results", payload)
            on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(on_disk["results"]), len(payload["results"]))

    def test_task1_filtered_profile_writes_real_profile_files_even_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = evaluate_candidates(
                runtime_settings=MvpRuntimeSettings(task1_detector_backend="yolo11n", task1_env_name="cpu", task1_device="cpu"),
                output_dir=tmp_dir,
                mode="smoke",
                task="task1",
                candidate="yolo11n",
            )
            self.assertEqual(len(payload["results"]), 1)
            self.assertEqual(payload["results"][0]["candidate_name"], "yolo11n")
            self.assertTrue((Path(tmp_dir) / "task1_real_profile.json").exists())
            self.assertTrue((Path(tmp_dir) / "task1_real_profile.md").exists())
            self.assertTrue((Path(tmp_dir) / "task1_real_profile_cpu.json").exists())
            self.assertEqual(payload["env_name"], "cpu")


if __name__ == "__main__":
    unittest.main()
