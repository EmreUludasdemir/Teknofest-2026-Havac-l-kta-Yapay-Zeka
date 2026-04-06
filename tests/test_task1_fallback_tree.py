from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.evaluation.task1_trt_validation import build_task1_fallback_tree


class Task1FallbackTreeTests(unittest.TestCase):
    def test_fallback_tree_locks_yolo11n_as_native_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            (reports_dir / "task1_yolo26n_onnx_export_summary.json").write_text(
                json.dumps({"validation_passed": True}),
                encoding="utf-8",
            )
            (reports_dir / "task1_yolo11n_onnx_export_summary.json").write_text(
                json.dumps({"validation_passed": False}),
                encoding="utf-8",
            )
            (reports_dir / "task1_trt_export_summary.json").write_text(
                json.dumps({"validation_passed": False}),
                encoding="utf-8",
            )
            payload = build_task1_fallback_tree(
                runtime_settings=MvpRuntimeSettings(),
                reports_dir=reports_dir,
            )
        stages = [item["stage"] for item in payload["stages"]]
        self.assertEqual(
            stages,
            [
                "tensorrt:yolo26n",
                "onnxruntime:yolo26n",
                "ultralytics:yolo26n",
                "ultralytics:yolo11n",
                "synthetic",
            ],
        )
        self.assertNotIn("onnxruntime:yolo11n", stages)


if __name__ == "__main__":
    unittest.main()
