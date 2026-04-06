from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import MvpRuntimeSettings
from src.exports.export_trt import build_task1_trt_export_spec, export_task1_candidate_to_trt
from src.task1.detector import Task1Detector, TensorRtModelDetectorBackend


class _FakeTrtBridge:
    created_with: dict | None = None

    def __init__(self, **kwargs) -> None:
        type(self).created_with = kwargs
        self.provider_name = "TensorRT"

    def load(self) -> None:
        return

    def unload(self) -> None:
        return

    def predict(self, decoded_frame):
        return [], {"provider": "TensorRT"}


class Task1TrtContractTests(unittest.TestCase):
    def test_build_trt_export_spec_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            onnx_path = root / "yolo26n.onnx"
            onnx_path.write_bytes(b"onnx")
            settings = MvpRuntimeSettings(
                task1_onnx_candidate_paths={"yolo26n": str(onnx_path)},
                task1_export_dir=root / "exports",
                task1_trt_precision="fp16",
                task1_trt_workspace_mb=2048,
            )
            spec = build_task1_trt_export_spec(settings, candidate_name="yolo26n")
            self.assertEqual(spec.source_onnx_path, onnx_path)
            self.assertEqual(spec.output_engine_path, root / "exports" / "yolo26n.engine")
            self.assertEqual(spec.metadata_path, root / "exports" / "yolo26n.engine.metadata.json")
            self.assertEqual(spec.precision, "fp16")
            self.assertEqual(spec.workspace_mb, 2048)

    def test_trt_export_returns_graceful_unavailable_when_dependency_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            onnx_path = root / "yolo26n.onnx"
            onnx_path.write_bytes(b"onnx")
            metadata_path = root / "yolo26n.metadata.json"
            metadata_path.write_text('{"class_names": ["person"]}', encoding="utf-8")
            settings = MvpRuntimeSettings(
                task1_onnx_candidate_paths={"yolo26n": str(onnx_path)},
                task1_export_dir=root,
            )
            with patch("src.exports.export_trt.is_tensorrt_available", return_value=False):
                payload = export_task1_candidate_to_trt(settings, candidate_name="yolo26n")
            self.assertFalse(payload["success"])
            self.assertIn("missing_dependency:tensorrt", payload["error"])

    def test_runtime_order_is_config_driven_for_requested_stage(self) -> None:
        settings = MvpRuntimeSettings(
            task1_detector_backend="yolo26n",
            task1_model_runtime="onnxruntime",
        )
        detector = Task1Detector(runtime_settings=settings)
        stage_ids = [spec.stage_id for spec, _ in detector.backend_chain]
        self.assertEqual(
            stage_ids,
            [
                "onnxruntime:yolo26n",
                "ultralytics:yolo26n",
                "ultralytics:yolo11n",
                "synthetic",
            ],
        )

    def test_trt_backend_loads_bridge_with_warmup_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            engine_path = root / "yolo26n.engine"
            engine_path.write_bytes(b"engine")
            metadata_path = root / "yolo26n.engine.metadata.json"
            metadata_path.write_text('{"input_shape":[1,3,640,640],"class_names":["person"]}', encoding="utf-8")
            settings = MvpRuntimeSettings(
                task1_model_runtime="tensorrt",
                task1_trt_candidate_paths={"yolo26n": str(engine_path)},
                task1_export_dir=root,
                task1_trt_warmup_runs=5,
            )
            backend = TensorRtModelDetectorBackend(runtime_settings=settings, backend_name="yolo26n")
            with patch("importlib.util.find_spec", return_value=object()), patch(
                "src.exports.trt_bridge.TensorRtTask1Bridge",
                _FakeTrtBridge,
            ):
                backend.load()
            self.assertEqual(_FakeTrtBridge.created_with["warmup_runs"], 5)


if __name__ == "__main__":
    unittest.main()
