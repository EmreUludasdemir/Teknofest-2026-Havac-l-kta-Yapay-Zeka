from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.tools.runtime_package import prepare_runtime_package
from tools.run_runtime_smoke import main as runtime_smoke_main


class RuntimePackageSmokeTests(unittest.TestCase):
    def test_prepare_runtime_package_writes_manifest_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir = root / "reports" / "export"
            models_dir = reports_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            (models_dir / "yolo26n.onnx").write_bytes(b"onnx")
            (models_dir / "yolo11n.onnx").write_bytes(b"onnx")
            (reports_dir / "task1_yolo26n_onnx_export_summary.json").write_text(
                json.dumps({"validation_passed": True}),
                encoding="utf-8",
            )
            settings = MvpRuntimeSettings(
                task1_candidate_paths={
                    "yolo26n": str(root / "yolo26n.pt"),
                    "yolo11n": str(root / "yolo11n.pt"),
                },
                task1_onnx_candidate_paths={
                    "yolo26n": str(models_dir / "yolo26n.onnx"),
                    "yolo11n": str(models_dir / "yolo11n.onnx"),
                },
            )
            payload = prepare_runtime_package(settings, base_dir=root / "final_runtime", reports_dir=reports_dir)
            self.assertTrue(Path(payload["config_path"]).exists())
            self.assertTrue(Path(payload["manifest_path"]).exists())
            self.assertTrue((root / "final_runtime" / "artifacts" / "onnx" / "yolo26n.onnx").exists())

    def test_runtime_smoke_cli_runs_single_batch_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rc = runtime_smoke_main(
                [
                    "--mode",
                    "batch",
                    "--frames",
                    "1",
                    "--base-dir",
                    str(root / "final_runtime"),
                    "--reports-dir",
                    str(root / "reports" / "export"),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((root / "final_runtime" / "logs" / "runtime_smoke_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
