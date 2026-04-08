from __future__ import annotations

import gc
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import replace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalDetection, DecodedFrame, FrameEnvelope
from src.core.utils import extract_frame_index
from src.core.vision import decode_image_bytes
from src.task1.class_mapping import canonical_task1_class_from_model_name


def _resolve_runtime_device() -> str | int | None:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    return 0 if bool(torch.cuda.is_available()) else "cpu"


def _canonical_class_from_model_name(class_name: str) -> int | None:
    """Map model labels to explicit TEKNOFEST 2026 Task 1 class IDs."""
    return canonical_task1_class_from_model_name(class_name)


def _resolve_model_label(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _validation_eligibility_for_stage(runtime_name: str) -> str:
    if runtime_name == "tensorrt":
        return "requires_trt_validation"
    if runtime_name == "onnxruntime":
        return "requires_onnx_validation"
    if runtime_name == "ultralytics":
        return "native_runtime"
    return "operational_only"


@dataclass(frozen=True, slots=True)
class DetectorStageSpec:
    runtime_name: str
    candidate_name: str | None
    stage_id: str


def _parse_detector_stage(stage_text: str) -> DetectorStageSpec:
    normalized = stage_text.strip().lower()
    if normalized == "synthetic":
        return DetectorStageSpec(runtime_name="synthetic", candidate_name=None, stage_id="synthetic")
    runtime_name, candidate_name = normalized.split(":", maxsplit=1)
    return DetectorStageSpec(
        runtime_name=runtime_name,
        candidate_name=candidate_name,
        stage_id=f"{runtime_name}:{candidate_name}",
    )


class Task1DetectorBackend(ABC):
    name: str

    def load(self) -> None:
        return

    def unload(self) -> None:
        gc.collect()

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError


@dataclass(slots=True)
class UnavailableDetectorBackend(Task1DetectorBackend):
    runtime_settings: MvpRuntimeSettings
    backend_name: str
    reason: str

    @property
    def name(self) -> str:
        return self.backend_name

    def is_available(self) -> bool:
        return False

    def detect(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        decoded_frame: DecodedFrame | None = None,
    ) -> list[CanonicalDetection]:
        raise RuntimeError(self.reason)

    @abstractmethod
    def detect(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        decoded_frame: DecodedFrame | None = None,
    ) -> list[CanonicalDetection]:
        raise NotImplementedError


@dataclass(slots=True)
class SyntheticDetectorBackend(Task1DetectorBackend):
    runtime_settings: MvpRuntimeSettings
    name: str = "synthetic"

    def is_available(self) -> bool:
        return True

    def detect(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        decoded_frame: DecodedFrame | None = None,
    ) -> list[CanonicalDetection]:
        frame_index = extract_frame_index(frame.frame_url)
        detections: list[CanonicalDetection] = []

        frame_width = float(decoded_frame.width if decoded_frame is not None else frame.metadata.get("image_width", 640))
        frame_height = float(decoded_frame.height if decoded_frame is not None else frame.metadata.get("image_height", 512))
        car_x = 20.0 + float((frame_index * 11) % max(int(frame_width * 0.25), 1))
        car_y = 30.0 + float((frame_index * 7) % max(int(frame_height * 0.2), 1))
        car = CanonicalDetection(
            class_id=0,
            landing_status=-1,
            motion_status=-1,
            top_left_x=car_x,
            top_left_y=car_y,
            bottom_right_x=min(car_x + 70.0, frame_width - 1.0),
            bottom_right_y=min(car_y + 40.0, frame_height - 1.0),
            metadata={
                "track_key": "vehicle-main",
                "score": 0.92,
                "detector_source": "task1_synthetic",
                "backend_name": self.name,
            },
        )
        detections.append(car)

        if frame_index % 2 == 0:
            detections.append(
                CanonicalDetection(
                    class_id=0,
                    landing_status=-1,
                    motion_status=-1,
                    top_left_x=car_x + 2.0,
                    top_left_y=car_y + 1.0,
                    bottom_right_x=min(car_x + 72.0, frame_width - 1.0),
                    bottom_right_y=min(car_y + 41.0, frame_height - 1.0),
                    metadata={
                        "track_key": "vehicle-main-dup",
                        "score": 0.41,
                        "detector_source": "task1_synthetic_duplicate",
                        "backend_name": self.name,
                    },
                )
            )

        if frame_index % 3 == 0:
            detections.append(
                CanonicalDetection(
                    class_id=1,
                    landing_status=-1,
                    motion_status=-1,
                    top_left_x=min(frame_width * 0.4, frame_width - 30.0),
                    top_left_y=min(110.0 + float((frame_index * 5) % 40), frame_height - 60.0),
                    bottom_right_x=min(frame_width * 0.4 + 25.0, frame_width - 1.0),
                    bottom_right_y=min(165.0 + float((frame_index * 5) % 40), frame_height - 1.0),
                    metadata={
                        "track_key": "human-main",
                        "score": 0.83,
                        "detector_source": "task1_synthetic",
                        "backend_name": self.name,
                    },
                )
            )

        if frame_index % 4 == 0:
            detections.append(
                CanonicalDetection(
                    class_id=2,
                    landing_status=-1,
                    motion_status=-1,
                    top_left_x=min(frame_width * 0.5, frame_width - 72.0),
                    top_left_y=min(frame_height * 0.38, frame_height - 72.0),
                    bottom_right_x=min(frame_width * 0.5 + 70.0, frame_width - 1.0),
                    bottom_right_y=min(frame_height * 0.38 + 70.0, frame_height - 1.0),
                    metadata={
                        "track_key": "uap-main",
                        "score": 0.88,
                        "detector_source": "task1_synthetic",
                        "backend_name": self.name,
                    },
                )
            )

        if frame_index % 5 == 0:
            detections.append(
                CanonicalDetection(
                    class_id=3,
                    landing_status=-1,
                    motion_status=-1,
                    top_left_x=min(frame_width * 0.67, frame_width - 72.0),
                    top_left_y=min(frame_height * 0.46, frame_height - 72.0),
                    bottom_right_x=min(frame_width * 0.67 + 70.0, frame_width - 1.0),
                    bottom_right_y=min(frame_height * 0.46 + 70.0, frame_height - 1.0),
                    metadata={
                        "track_key": "uai-main",
                        "score": 0.86,
                        "detector_source": "task1_synthetic",
                        "backend_name": self.name,
                    },
                )
            )

        return detections


@dataclass(slots=True)
class LocalModelDetectorBackend(Task1DetectorBackend):
    runtime_settings: MvpRuntimeSettings
    backend_name: str
    model_path: Path | None = None
    model: Any | None = None
    availability_error: str | None = None
    runtime_device: str | int | None = None

    @property
    def name(self) -> str:
        return self.backend_name

    def is_available(self) -> bool:
        if self.runtime_settings.task1_model_runtime != "ultralytics":
            self.availability_error = f"unsupported_runtime:{self.runtime_settings.task1_model_runtime}"
            return False
        if importlib.util.find_spec("ultralytics") is None:
            self.availability_error = "missing_dependency:ultralytics"
            return False
        resolved = self._resolve_model_path()
        if resolved is None:
            self.availability_error = "missing_model_path"
            return False
        if not resolved.exists():
            self.availability_error = f"missing_model_file:{resolved}"
            return False
        return True

    def _resolve_model_path(self) -> Path | None:
        if self.model_path is not None:
            return self.model_path
        self.model_path = self.runtime_settings.resolve_task1_model_path(self.backend_name)
        return self.model_path

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.is_available():
            raise RuntimeError(self.availability_error or "local_model_unavailable")
        from ultralytics import YOLO  # type: ignore[import-not-found]

        self.runtime_device = self.runtime_settings.task1_device or _resolve_runtime_device()
        self.model = YOLO(str(self._resolve_model_path()))

    def unload(self) -> None:
        self.model = None
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found]

            if bool(torch.cuda.is_available()):
                torch.cuda.empty_cache()
        except Exception:
            return

    def detect(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        decoded_frame: DecodedFrame | None = None,
    ) -> list[CanonicalDetection]:
        self.load()
        effective_frame = decoded_frame or decode_image_bytes(frame, image_bytes)
        if effective_frame.bgr is None:
            raise RuntimeError("local_model_decode_failed")

        predict_kwargs: dict[str, Any] = {
            "source": effective_frame.bgr,
            "verbose": False,
            "imgsz": self.runtime_settings.task1_model_input_size,
            "conf": self.runtime_settings.task1_conf_threshold,
            "iou": self.runtime_settings.task1_iou_threshold,
            "max_det": self.runtime_settings.max_objects_per_frame,
        }
        if self.runtime_device is not None:
            predict_kwargs["device"] = self.runtime_device
        predictions = self.model.predict(**predict_kwargs)
        if not predictions:
            return []

        result = predictions[0]
        names = getattr(result, "names", getattr(self.model, "names", {}))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: list[CanonicalDetection] = []
        xyxy_values = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else []
        conf_values = boxes.conf.tolist() if hasattr(boxes, "conf") and hasattr(boxes.conf, "tolist") else []
        cls_values = boxes.cls.tolist() if hasattr(boxes, "cls") and hasattr(boxes.cls, "tolist") else []
        for index, coords in enumerate(xyxy_values):
            if len(coords) != 4:
                continue
            model_class_id = int(cls_values[index]) if index < len(cls_values) else -1
            label = _resolve_model_label(names, model_class_id)
            canonical_class = _canonical_class_from_model_name(label)
            if canonical_class is None:
                continue
            score = float(conf_values[index]) if index < len(conf_values) else 0.0
            detections.append(
                CanonicalDetection(
                    class_id=canonical_class,
                    landing_status=-1,
                    motion_status=-1,
                    top_left_x=float(coords[0]),
                    top_left_y=float(coords[1]),
                    bottom_right_x=float(coords[2]),
                    bottom_right_y=float(coords[3]),
                    metadata={
                        "track_key": f"{self.backend_name}:{canonical_class}:{index}",
                        "score": round(score, 4),
                        "detector_source": "task1_local_model",
                        "backend_name": self.backend_name,
                        "model_class_name": label,
                        "model_path": str(self._resolve_model_path()),
                        "runtime_device": str(self.runtime_device or "auto"),
                    },
                )
            )
        return detections


@dataclass(slots=True)
class ExternalModelDetectorBackend(LocalModelDetectorBackend):
    """Geriye uyum icin saklanan legacy alias."""


@dataclass(slots=True)
class OnnxModelDetectorBackend(Task1DetectorBackend):
    runtime_settings: MvpRuntimeSettings
    backend_name: str
    model_path: Path | None = None
    metadata_path: Path | None = None
    bridge: Any | None = None
    availability_error: str | None = None
    runtime_provider: str | None = None

    @property
    def name(self) -> str:
        return self.backend_name

    def is_available(self) -> bool:
        if self.runtime_settings.task1_model_runtime != "onnxruntime":
            self.availability_error = f"unsupported_runtime:{self.runtime_settings.task1_model_runtime}"
            return False
        if importlib.util.find_spec("onnxruntime") is None:
            self.availability_error = "missing_dependency:onnxruntime"
            return False
        resolved = self._resolve_model_path()
        if resolved is None:
            self.availability_error = "missing_onnx_path"
            return False
        if not resolved.exists():
            self.availability_error = f"missing_onnx_file:{resolved}"
            return False
        metadata_path = self._resolve_metadata_path()
        if not metadata_path.exists():
            self.availability_error = f"missing_onnx_metadata:{metadata_path}"
            return False
        return True

    def _resolve_model_path(self) -> Path | None:
        if self.model_path is not None:
            return self.model_path
        self.model_path = self.runtime_settings.resolve_task1_onnx_path(self.backend_name)
        return self.model_path

    def _resolve_metadata_path(self) -> Path:
        if self.metadata_path is not None:
            return self.metadata_path
        self.metadata_path = self.runtime_settings.resolve_task1_export_metadata_path(self.backend_name)
        return self.metadata_path

    def load(self) -> None:
        if self.bridge is not None:
            return
        if not self.is_available():
            raise RuntimeError(self.availability_error or "onnx_model_unavailable")
        from src.exports.onnx_bridge import OnnxTask1Bridge

        self.bridge = OnnxTask1Bridge(
            model_path=self._resolve_model_path(),
            metadata_path=self._resolve_metadata_path(),
            providers=list(self.runtime_settings.task1_onnx_providers),
            conf_threshold=self.runtime_settings.task1_conf_threshold,
            conf_threshold_offset=self.runtime_settings.task1_onnx_conf_threshold_offset,
            iou_threshold=self.runtime_settings.task1_iou_threshold,
            max_objects_per_frame=self.runtime_settings.max_objects_per_frame,
        )
        self.bridge.load()
        self.runtime_provider = str(getattr(self.bridge, "provider_name", "unknown"))

    def unload(self) -> None:
        if self.bridge is not None:
            try:
                self.bridge.unload()
            finally:
                self.bridge = None
        gc.collect()

    def detect(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        decoded_frame: DecodedFrame | None = None,
    ) -> list[CanonicalDetection]:
        self.load()
        effective_frame = decoded_frame or decode_image_bytes(frame, image_bytes)
        detections, diagnostics = self.bridge.predict(effective_frame)
        self.runtime_provider = str(diagnostics.get("provider", self.runtime_provider or "unknown"))
        for detection in detections:
            detection.metadata.setdefault("backend_name", self.backend_name)
            detection.metadata.setdefault("detector_source", "task1_onnx_model")
            detection.metadata["runtime_device"] = self.runtime_provider or "unknown"
        return detections


@dataclass(slots=True)
class TensorRtModelDetectorBackend(Task1DetectorBackend):
    runtime_settings: MvpRuntimeSettings
    backend_name: str
    model_path: Path | None = None
    metadata_path: Path | None = None
    bridge: Any | None = None
    availability_error: str | None = None
    runtime_provider: str | None = None

    @property
    def name(self) -> str:
        return self.backend_name

    def is_available(self) -> bool:
        if self.runtime_settings.task1_model_runtime != "tensorrt":
            self.availability_error = f"unsupported_runtime:{self.runtime_settings.task1_model_runtime}"
            return False
        if importlib.util.find_spec("tensorrt") is None:
            self.availability_error = "missing_dependency:tensorrt"
            return False
        resolved = self._resolve_model_path()
        if resolved is None:
            self.availability_error = "missing_trt_path"
            return False
        if not resolved.exists():
            self.availability_error = f"missing_trt_engine:{resolved}"
            return False
        metadata_path = self._resolve_metadata_path()
        if not metadata_path.exists():
            self.availability_error = f"missing_trt_metadata:{metadata_path}"
            return False
        return True

    def _resolve_model_path(self) -> Path | None:
        if self.model_path is not None:
            return self.model_path
        self.model_path = self.runtime_settings.resolve_task1_trt_path(self.backend_name)
        return self.model_path

    def _resolve_metadata_path(self) -> Path:
        if self.metadata_path is not None:
            return self.metadata_path
        self.metadata_path = self.runtime_settings.resolve_task1_trt_metadata_path(self.backend_name)
        return self.metadata_path

    def load(self) -> None:
        if self.bridge is not None:
            return
        if not self.is_available():
            raise RuntimeError(self.availability_error or "trt_model_unavailable")
        from src.exports.trt_bridge import TensorRtTask1Bridge

        self.bridge = TensorRtTask1Bridge(
            engine_path=self._resolve_model_path(),
            metadata_path=self._resolve_metadata_path(),
            conf_threshold=self.runtime_settings.task1_conf_threshold,
            iou_threshold=self.runtime_settings.task1_iou_threshold,
            max_objects_per_frame=self.runtime_settings.max_objects_per_frame,
            warmup_runs=self.runtime_settings.task1_trt_warmup_runs,
        )
        self.bridge.load()
        self.runtime_provider = str(getattr(self.bridge, "provider_name", "TensorRT"))

    def unload(self) -> None:
        if self.bridge is not None:
            try:
                self.bridge.unload()
            finally:
                self.bridge = None
        gc.collect()

    def detect(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        decoded_frame: DecodedFrame | None = None,
    ) -> list[CanonicalDetection]:
        self.load()
        effective_frame = decoded_frame or decode_image_bytes(frame, image_bytes)
        detections, diagnostics = self.bridge.predict(effective_frame)
        self.runtime_provider = str(diagnostics.get("provider", self.runtime_provider or "TensorRT"))
        for detection in detections:
            detection.metadata.setdefault("backend_name", self.backend_name)
            detection.metadata.setdefault("detector_source", "task1_trt_model")
            detection.metadata["runtime_device"] = self.runtime_provider or "TensorRT"
            detection.metadata["trt_provider"] = self.runtime_provider or "TensorRT"
        return detections


@dataclass(slots=True)
class Task1Detector:
    """Backend secen facade detector."""

    runtime_settings: MvpRuntimeSettings | None = None
    backend: Task1DetectorBackend = field(init=False)
    fallback_backend: SyntheticDetectorBackend = field(init=False)
    backend_chain: list[tuple[DetectorStageSpec, Task1DetectorBackend]] = field(init=False, default_factory=list)
    primary_backend_name: str = field(init=False, default="synthetic")
    requested_stage: DetectorStageSpec = field(init=False)

    def __post_init__(self) -> None:
        self.runtime_settings = self.runtime_settings or MvpRuntimeSettings()
        self.fallback_backend = SyntheticDetectorBackend(runtime_settings=self.runtime_settings)
        self.requested_stage = self._resolve_requested_stage()
        self.primary_backend_name = self.requested_stage.candidate_name or "synthetic"
        self.backend_chain = [
            (spec, self._build_backend_for_stage(spec))
            for spec in self._resolve_stage_sequence()
        ]
        self.backend = self.backend_chain[0][1] if self.backend_chain else self.fallback_backend

    def detect(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        decoded_frame: DecodedFrame | None = None,
    ) -> list[CanonicalDetection]:
        errors: list[str] = []
        for spec, backend in self.backend_chain:
            try:
                detections = backend.detect(frame, image_bytes, decoded_frame=decoded_frame)
                self._annotate_success(
                    detections,
                    active_backend=backend,
                    active_spec=spec,
                    errors=errors,
                )
                return detections
            except Exception as exc:
                errors.append(f"{spec.stage_id}:{exc}")

        detections = self.fallback_backend.detect(frame, image_bytes, decoded_frame=decoded_frame)
        for detection in detections:
            detection.metadata["backend_name"] = self.primary_backend_name
            detection.metadata["active_backend_name"] = self.fallback_backend.name
            detection.metadata["requested_runtime_name"] = self.requested_stage.runtime_name
            detection.metadata["requested_stage_id"] = self.requested_stage.stage_id
            detection.metadata["active_runtime_name"] = "synthetic"
            detection.metadata["active_stage_id"] = "synthetic"
            detection.metadata["validation_eligibility"] = _validation_eligibility_for_stage("synthetic")
            detection.metadata["backend_unavailable"] = True
            detection.metadata["backend_error"] = " | ".join(errors)
            detection.metadata["fallback_stage"] = "operational_synthetic"
        return detections

    def unload(self) -> None:
        seen_backend_ids: set[int] = set()
        for _, backend in self.backend_chain:
            backend_id = id(backend)
            if backend_id in seen_backend_ids:
                continue
            seen_backend_ids.add(backend_id)
            try:
                backend.unload()
            except Exception:
                continue

    def warm_up(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        decoded_frame: DecodedFrame | None = None,
    ) -> list[CanonicalDetection]:
        return self.detect(frame, image_bytes, decoded_frame=decoded_frame)

    def _resolve_requested_stage(self) -> DetectorStageSpec:
        configured_backend = self.runtime_settings.task1_detector_backend.lower()
        if configured_backend == "synthetic":
            return DetectorStageSpec(runtime_name="synthetic", candidate_name=None, stage_id="synthetic")
        runtime_name = self.runtime_settings.task1_model_runtime.lower()
        return DetectorStageSpec(
            runtime_name=runtime_name,
            candidate_name=configured_backend,
            stage_id=f"{runtime_name}:{configured_backend}",
        )

    def _resolve_stage_sequence(self) -> list[DetectorStageSpec]:
        requested_stage_id = self.requested_stage.stage_id
        configured_order = [
            _parse_detector_stage(item)
            for item in self.runtime_settings.task1_runtime_order
        ]
        unique_specs: list[DetectorStageSpec] = []
        seen_stage_ids: set[str] = set()
        for spec in configured_order:
            if spec.stage_id in seen_stage_ids:
                continue
            seen_stage_ids.add(spec.stage_id)
            unique_specs.append(spec)

        if requested_stage_id == "synthetic":
            return [self.requested_stage]

        ordered_specs = unique_specs
        if requested_stage_id in seen_stage_ids:
            start_index = next(
                index for index, spec in enumerate(unique_specs) if spec.stage_id == requested_stage_id
            )
            ordered_specs = unique_specs[start_index:]
        else:
            ordered_specs = [self.requested_stage] + unique_specs

        if not ordered_specs or ordered_specs[-1].stage_id != "synthetic":
            ordered_specs.append(DetectorStageSpec(runtime_name="synthetic", candidate_name=None, stage_id="synthetic"))

        final_specs: list[DetectorStageSpec] = []
        seen_final: set[str] = set()
        for spec in ordered_specs:
            if spec.stage_id in seen_final:
                continue
            seen_final.add(spec.stage_id)
            final_specs.append(spec)
        return final_specs

    def _build_backend_for_stage(self, spec: DetectorStageSpec) -> Task1DetectorBackend:
        if spec.runtime_name == "synthetic" or spec.candidate_name is None:
            return self.fallback_backend
        stage_settings = replace(self.runtime_settings, task1_model_runtime=spec.runtime_name)
        if spec.runtime_name == "ultralytics":
            return LocalModelDetectorBackend(
                runtime_settings=stage_settings,
                backend_name=spec.candidate_name,
            )
        if spec.runtime_name == "onnxruntime":
            return OnnxModelDetectorBackend(
                runtime_settings=stage_settings,
                backend_name=spec.candidate_name,
            )
        if spec.runtime_name == "tensorrt":
            return TensorRtModelDetectorBackend(
                runtime_settings=stage_settings,
                backend_name=spec.candidate_name,
            )
        return UnavailableDetectorBackend(
            runtime_settings=stage_settings,
            backend_name=spec.candidate_name,
            reason=f"unsupported_runtime:{spec.runtime_name}",
        )

    def _annotate_success(
        self,
        detections: list[CanonicalDetection],
        *,
        active_backend: Task1DetectorBackend,
        active_spec: DetectorStageSpec,
        errors: list[str],
    ) -> None:
        for detection in detections:
            detection.metadata["backend_name"] = self.primary_backend_name
            detection.metadata["requested_backend_name"] = self.primary_backend_name
            detection.metadata["requested_runtime_name"] = self.requested_stage.runtime_name
            detection.metadata["requested_stage_id"] = self.requested_stage.stage_id
            detection.metadata["active_backend_name"] = getattr(active_backend, "name", self.primary_backend_name)
            detection.metadata["active_runtime_name"] = active_spec.runtime_name
            detection.metadata["active_stage_id"] = active_spec.stage_id
            detection.metadata["validation_eligibility"] = _validation_eligibility_for_stage(active_spec.runtime_name)
            if active_spec.stage_id != self.requested_stage.stage_id:
                detection.metadata["backend_unavailable"] = True
                detection.metadata["backend_error"] = " | ".join(errors)
                detection.metadata["fallback_stage"] = active_spec.stage_id
