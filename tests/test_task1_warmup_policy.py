from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalDetection, FrameEnvelope
from src.task1.detector import DetectorStageSpec, Task1Detector, Task1DetectorBackend


class _FakeBackend(Task1DetectorBackend):
    def __init__(self, backend_name: str, outcomes: list[object]) -> None:
        self.backend_name = backend_name
        self.outcomes = list(outcomes)

    @property
    def name(self) -> str:
        return self.backend_name

    def is_available(self) -> bool:
        return True

    def detect(self, frame, image_bytes, decoded_frame=None):
        outcome = self.outcomes.pop(0) if self.outcomes else []
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Task1WarmupPolicyTests(unittest.TestCase):
    def _frame(self) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url="http://mock/frames/1/",
            image_url="/frame.jpg",
            video_name="mock",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
            metadata={"image_width": 640, "image_height": 512},
        )

    def test_warmup_falls_through_to_next_eligible_stage(self) -> None:
        detector = Task1Detector(
            runtime_settings=MvpRuntimeSettings(
                task1_detector_backend="yolo26n",
                task1_model_runtime="tensorrt",
                task1_enabled_stage_ids=["tensorrt:yolo26n", "onnxruntime:yolo26n"],
            )
        )
        trt_spec = DetectorStageSpec("tensorrt", "yolo26n", "tensorrt:yolo26n")
        onnx_spec = DetectorStageSpec("onnxruntime", "yolo26n", "onnxruntime:yolo26n")
        detector.backend_chain = [
            (trt_spec, _FakeBackend("yolo26n", [RuntimeError("trt failed")])),
            (onnx_spec, _FakeBackend("yolo26n", [[]])),
            (DetectorStageSpec("synthetic", None, "synthetic"), detector.fallback_backend),
        ]
        detector.preferred_stage_id = trt_spec.stage_id

        payload = detector.warm_up_stage(self._frame(), b"img")
        self.assertEqual(payload["selected_stage"], "onnxruntime:yolo26n")
        self.assertEqual(detector.active_stage_id, "onnxruntime:yolo26n")

    def test_first_frame_failure_retries_same_frame_on_later_stage(self) -> None:
        detector = Task1Detector(
            runtime_settings=MvpRuntimeSettings(
                task1_detector_backend="yolo26n",
                task1_model_runtime="onnxruntime",
                task1_enabled_stage_ids=["onnxruntime:yolo26n", "ultralytics:yolo26n"],
            )
        )
        onnx_spec = DetectorStageSpec("onnxruntime", "yolo26n", "onnxruntime:yolo26n")
        native_spec = DetectorStageSpec("ultralytics", "yolo26n", "ultralytics:yolo26n")
        detector.backend_chain = [
            (onnx_spec, _FakeBackend("yolo26n", [RuntimeError("onnx failed")])),
            (
                native_spec,
                _FakeBackend(
                    "yolo26n",
                    [[CanonicalDetection(class_id=0, metadata={"score": 0.9})]],
                ),
            ),
            (DetectorStageSpec("synthetic", None, "synthetic"), detector.fallback_backend),
        ]
        detector.preferred_stage_id = onnx_spec.stage_id

        detections = detector.detect(self._frame(), b"img")
        self.assertTrue(detections)
        self.assertEqual(detections[0].metadata["active_stage_id"], "ultralytics:yolo26n")
        self.assertEqual(detector.active_stage_id, "ultralytics:yolo26n")


if __name__ == "__main__":
    unittest.main()
