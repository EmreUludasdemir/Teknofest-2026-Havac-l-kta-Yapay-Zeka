from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame
from src.core.vision import is_cv2_available
from src.exports import onnx_bridge
from src.exports.export_onnx import build_task1_export_spec

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import np
else:  # pragma: no cover - cv2 yoksa
    np = None


class _FakeOrt:
    @staticmethod
    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider", "CUDAExecutionProvider"]


class Task1OnnxContractTests(unittest.TestCase):
    def test_build_export_spec_resolves_candidate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "yolo26n.pt"
            model_path.write_bytes(b"pt")
            settings = MvpRuntimeSettings(
                task1_candidate_paths={"yolo26n": str(model_path)},
                task1_export_dir=root / "exports",
                task1_model_input_size=640,
                task1_export_opset=17,
                task1_model_runtime="ultralytics",
                task1_device="cuda:0",
            )
            spec = build_task1_export_spec(settings, candidate_name="yolo26n")
            self.assertEqual(spec.source_model_path, model_path)
            self.assertEqual(spec.output_model_path, root / "exports" / "yolo26n.onnx")
            self.assertEqual(spec.metadata_path, root / "exports" / "yolo26n.metadata.json")
            self.assertEqual(spec.imgsz, 640)
            self.assertEqual(spec.opset, 17)
            self.assertEqual(spec.device, "cuda:0")

    @unittest.skipUnless(is_cv2_available(), "opencv gerekli")
    def test_preprocess_shape_contract_is_fixed_1x3x640x640(self) -> None:
        assert np is not None
        decoded_frame = DecodedFrame(
            bgr=np.zeros((1080, 1920, 3), dtype=np.uint8),
            gray=np.zeros((1080, 1920), dtype=np.uint8),
            width=1920,
            height=1080,
            channel_count=3,
            modality="rgb",
            frame_index=1,
        )
        tensor, context = onnx_bridge.preprocess_task1_onnx_input(decoded_frame, input_size=640)
        self.assertEqual(tuple(tensor.shape), (1, 3, 640, 640))
        self.assertEqual(context.original_width, 1920)
        self.assertEqual(context.original_height, 1080)
        self.assertGreaterEqual(context.pad_y, 0.0)

    def test_provider_resolution_prefers_gpu_then_cpu(self) -> None:
        with patch.object(onnx_bridge, "ort", _FakeOrt()):
            providers = onnx_bridge.resolve_onnx_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
