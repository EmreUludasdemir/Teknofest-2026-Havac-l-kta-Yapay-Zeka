from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import MvpRuntimeSettings
from tools.run_phase10_gpu_validation import ensure_trt_engine_ready


class Phase10GpuValidationTests(unittest.TestCase):
    def test_missing_gpu_name_in_metadata_triggers_rebuild_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            onnx_path = root / "yolo26n.onnx"
            onnx_path.write_bytes(b"onnx")
            engine_path = root / "yolo26n.engine"
            engine_path.write_bytes(b"engine")
            metadata_path = root / "yolo26n.engine.metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "input_shape": [1, 3, 640, 640],
                        "precision": "fp16",
                        "builder_version": "10.16.0.72",
                    }
                ),
                encoding="utf-8",
            )
            settings = MvpRuntimeSettings(
                task1_onnx_candidate_paths={"yolo26n": str(onnx_path)},
                task1_trt_candidate_paths={"yolo26n": str(engine_path)},
                task1_export_dir=root,
            )
            with patch("tools.run_phase10_gpu_validation.resolve_gpu_name", return_value="RTX 5060"), patch(
                "tools.run_phase10_gpu_validation.export_task1_candidate_to_trt",
                return_value={"success": False},
            ):
                payload = ensure_trt_engine_ready(settings, logger=None)  # type: ignore[arg-type]
            self.assertTrue(payload["rebuild_attempted"])
            self.assertIn("missing_gpu_name", payload["mismatch_reasons"])


if __name__ == "__main__":
    unittest.main()
