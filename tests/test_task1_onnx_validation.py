from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope
from src.evaluation.task1_onnx_validation import compare_detection_runs
from src.task1.detector import Task1Detector


class Task1OnnxValidationTests(unittest.TestCase):
    def _frame(self) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url="http://validation/frames/8/",
            image_url="/frame.png",
            video_name="THYZ_2026_Ornek_Veri_1",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
            metadata={"image_width": 1920, "image_height": 1080},
        )

    def test_compare_detection_runs_accepts_balanced_similarity(self) -> None:
        native = {
            "frame_results": [
                {"frame_index": 1, "bbox_count": 3, "class_histogram": {"0": 2, "2": 1}, "latency_ms": 20.0},
                {"frame_index": 2, "bbox_count": 0, "class_histogram": {}, "latency_ms": 18.0},
            ],
            "aggregate_class_histogram": {"0": 2, "2": 1},
            "latency_p50_ms": 19.0,
        }
        onnx = {
            "frame_results": [
                {"frame_index": 1, "bbox_count": 2, "class_histogram": {"0": 1, "2": 1}, "latency_ms": 24.0},
                {"frame_index": 2, "bbox_count": 0, "class_histogram": {}, "latency_ms": 23.0},
            ],
            "aggregate_class_histogram": {"0": 1, "2": 1},
            "latency_p50_ms": 24.0,
        }
        comparison = compare_detection_runs(native, onnx)
        self.assertTrue(comparison["accepted"])
        self.assertTrue(comparison["onnx_smoke_passed"])

    def test_compare_detection_runs_rejects_latency_regression(self) -> None:
        native = {
            "frame_results": [{"frame_index": 1, "bbox_count": 1, "class_histogram": {"0": 1}, "latency_ms": 10.0}],
            "aggregate_class_histogram": {"0": 1},
            "latency_p50_ms": 10.0,
        }
        onnx = {
            "frame_results": [{"frame_index": 1, "bbox_count": 1, "class_histogram": {"0": 1}, "latency_ms": 30.0}],
            "aggregate_class_histogram": {"0": 1},
            "latency_p50_ms": 30.0,
        }
        comparison = compare_detection_runs(native, onnx)
        self.assertFalse(comparison["accepted"])
        self.assertFalse(comparison["latency_ok"])

    def test_onnx_runtime_falls_back_without_breaking_detector_chain(self) -> None:
        settings = MvpRuntimeSettings(
            task1_detector_backend="yolo26n",
            task1_model_runtime="onnxruntime",
            task1_model_fallback_candidate="yolo11n",
        )
        detector = Task1Detector(runtime_settings=settings)
        detections = detector.detect(self._frame(), b"img")
        self.assertTrue(detections)
        self.assertTrue(any(item.metadata.get("backend_unavailable") for item in detections))
        self.assertTrue(all(item.metadata.get("backend_name") == "yolo26n" for item in detections))
        errors = " | ".join(str(item.metadata.get("backend_error", "")) for item in detections)
        self.assertTrue(
            ("missing_dependency:onnxruntime" in errors)
            or ("missing_onnx_path" in errors)
            or ("missing_onnx_file" in errors)
        )


if __name__ == "__main__":
    unittest.main()
