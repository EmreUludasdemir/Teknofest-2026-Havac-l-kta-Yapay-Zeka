from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.frame_state import CanonicalDetection, DecodedFrame
from src.core.utils import clamp, normalize_box
from src.core.vision import is_cv2_available
from src.task1.class_mapping import canonical_task1_class_from_model_name
from src.task1.postprocess import deduplicate_detections

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2, np
else:  # pragma: no cover - cv2 yoksa
    cv2 = None
    np = None

try:  # pragma: no cover - opsiyonel bagimlilik
    import onnxruntime as ort  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - opsiyonel bagimlilik
    ort = None


@dataclass(slots=True)
class OnnxPreprocessContext:
    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_x: float
    pad_y: float


def is_onnxruntime_available() -> bool:
    return ort is not None and np is not None and cv2 is not None


def prepare_onnxruntime_execution_environment(preferred_providers: list[str]) -> None:
    if ort is None:
        return
    wants_cuda = "CUDAExecutionProvider" in preferred_providers
    if wants_cuda:
        try:  # pragma: no cover - opsiyonel ortam ayrintisi
            import torch  # type: ignore[import-not-found]  # noqa: F401
        except Exception:
            pass
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:  # pragma: no cover - opsiyonel ortam ayrintisi
            preload()
        except Exception:
            return


class OnnxTask1Bridge:
    def __init__(
        self,
        *,
        model_path: str | Path,
        metadata_path: str | Path,
        providers: list[str],
        conf_threshold: float,
        conf_threshold_offset: float,
        iou_threshold: float,
        max_objects_per_frame: int,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.providers = list(providers)
        self.conf_threshold = float(conf_threshold)
        self.conf_threshold_offset = float(conf_threshold_offset)
        self.iou_threshold = float(iou_threshold)
        self.max_objects_per_frame = int(max_objects_per_frame)
        self.metadata = load_onnx_metadata(self.metadata_path)
        self.session = None
        self.input_name = ""
        self.output_names: list[str] = []
        self.provider_name = "unknown"

    def load(self) -> None:
        if self.session is not None:
            return
        if not is_onnxruntime_available():
            raise RuntimeError("missing_dependency:onnxruntime")
        if not self.model_path.exists():
            raise FileNotFoundError(f"missing_onnx_file:{self.model_path}")
        prepare_onnxruntime_execution_environment(self.providers)
        selected_providers = resolve_onnx_providers(self.providers)
        if not selected_providers:
            selected_providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=selected_providers)
        self.input_name = str(self.session.get_inputs()[0].name)
        self.output_names = [str(item.name) for item in self.session.get_outputs()]
        active = list(self.session.get_providers())
        self.provider_name = active[0] if active else "unknown"

    def unload(self) -> None:
        self.session = None

    def predict(self, decoded_frame: DecodedFrame) -> tuple[list[CanonicalDetection], dict[str, Any]]:
        self.load()
        input_tensor, preprocess_context = preprocess_task1_onnx_input(
            decoded_frame,
            input_size=int(self.metadata.get("input_shape", [1, 3, 640, 640])[-1]),
        )
        outputs = self.session.run(self.output_names or None, {self.input_name: input_tensor})
        detections = decode_yolo_onnx_outputs(
            outputs[0],
            preprocess_context=preprocess_context,
            class_names=list(self.metadata.get("class_names", [])),
            conf_threshold=self.conf_threshold + self.conf_threshold_offset,
            iou_threshold=self.iou_threshold,
            max_objects_per_frame=self.max_objects_per_frame,
            provider_name=self.provider_name,
            model_path=self.model_path,
        )
        diagnostics = {
            "provider": self.provider_name,
            "input_name": self.input_name,
            "output_count": len(self.output_names),
            "input_shape": list(input_tensor.shape),
            "raw_output_shape": list(np.asarray(outputs[0]).shape) if np is not None else [],
        }
        return detections, diagnostics


def preprocess_task1_onnx_input(
    decoded_frame: DecodedFrame,
    *,
    input_size: int = 640,
) -> tuple[Any, OnnxPreprocessContext]:
    if np is None or cv2 is None or decoded_frame.bgr is None:
        raise RuntimeError("onnx_preprocess_requires_cv2_and_bgr")
    rgb = cv2.cvtColor(decoded_frame.bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = min(float(input_size) / max(height, 1), float(input_size) / max(width, 1))
    resized_width = max(int(round(width * scale)), 1)
    resized_height = max(int(round(height * scale)), 1)
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_w = max(input_size - resized_width, 0)
    pad_h = max(input_size - resized_height, 0)
    pad_x = pad_w / 2.0
    pad_y = pad_h / 2.0
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = letterboxed.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    context = OnnxPreprocessContext(
        original_width=int(width),
        original_height=int(height),
        input_size=int(input_size),
        scale=float(scale),
        pad_x=float(left),
        pad_y=float(top),
    )
    return tensor, context


def load_onnx_metadata(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"missing_onnx_metadata:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_onnx_providers(preferred_providers: list[str]) -> list[str]:
    if ort is None:
        return []
    available = set(ort.get_available_providers())
    selected = [item for item in preferred_providers if item in available]
    if "CPUExecutionProvider" not in selected and "CPUExecutionProvider" in available:
        selected.append("CPUExecutionProvider")
    return selected


def decode_yolo_onnx_outputs(
    output_tensor: Any,
    *,
    preprocess_context: OnnxPreprocessContext,
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float,
    max_objects_per_frame: int,
    provider_name: str,
    model_path: Path,
) -> list[CanonicalDetection]:
    if np is None:
        raise RuntimeError("numpy_required_for_onnx_decode")
    rows = _normalize_prediction_rows(output_tensor)
    detections: list[CanonicalDetection] = []
    for row in rows:
        detection = _decode_prediction_row(
            row,
            preprocess_context=preprocess_context,
            class_names=class_names,
            conf_threshold=conf_threshold,
            provider_name=provider_name,
            model_path=model_path,
        )
        if detection is not None:
            detections.append(detection)
    return deduplicate_detections(
        detections,
        max_objects_per_frame=max_objects_per_frame,
        iou_threshold=iou_threshold,
    )


def _normalize_prediction_rows(output_tensor: Any) -> Any:
    tensor = np.asarray(output_tensor)
    if tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise RuntimeError(f"unsupported_onnx_output_shape:{list(tensor.shape)}")
    if tensor.shape[0] <= 128 and tensor.shape[1] > tensor.shape[0]:
        tensor = tensor.T
    return tensor


def _decode_prediction_row(
    row: Any,
    *,
    preprocess_context: OnnxPreprocessContext,
    class_names: list[str],
    conf_threshold: float,
    provider_name: str,
    model_path: Path,
) -> CanonicalDetection | None:
    if len(row) < 6:
        return None

    class_count = len(class_names)
    if len(row) == 6:
        x1, y1, x2, y2, score, raw_class_id = [float(item) for item in row]
        class_id = int(raw_class_id)
        label = class_names[class_id] if 0 <= class_id < class_count else str(class_id)
    else:
        x1, y1, x2, y2, score, label = _decode_raw_yolo_candidate(row, class_names)
        if label is None:
            return None

    if score < conf_threshold:
        return None

    canonical_class = _canonical_class_from_model_name(label)
    if canonical_class is None:
        return None

    box = _rescale_box((x1, y1, x2, y2), preprocess_context)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None

    return CanonicalDetection(
        class_id=canonical_class,
        landing_status=-1,
        motion_status=-1,
        top_left_x=box[0],
        top_left_y=box[1],
        bottom_right_x=box[2],
        bottom_right_y=box[3],
        metadata={
            "score": round(float(score), 4),
            "detector_source": "task1_onnx_model",
            "onnx_provider": provider_name,
            "onnx_model_path": str(model_path),
            "model_class_name": label,
        },
    )


def _decode_raw_yolo_candidate(row: Any, class_names: list[str]) -> tuple[float, float, float, float, float, str | None]:
    values = [float(item) for item in row]
    class_count = len(class_names)
    if class_count <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, None

    if len(values) >= 5 + class_count:
        objectness = values[4]
        class_scores = values[5 : 5 + class_count]
    elif len(values) >= 4 + class_count:
        objectness = 1.0
        class_scores = values[4 : 4 + class_count]
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0, None

    if not class_scores:
        return 0.0, 0.0, 0.0, 0.0, 0.0, None
    class_index = int(np.argmax(class_scores))
    class_score = float(class_scores[class_index])
    score = float(objectness) * class_score
    cx, cy, width, height = values[0], values[1], values[2], values[3]
    x1 = cx - (width / 2.0)
    y1 = cy - (height / 2.0)
    x2 = cx + (width / 2.0)
    y2 = cy + (height / 2.0)
    label = class_names[class_index] if 0 <= class_index < len(class_names) else str(class_index)
    return x1, y1, x2, y2, score, label


def _rescale_box(box: tuple[float, float, float, float], preprocess_context: OnnxPreprocessContext) -> tuple[float, float, float, float]:
    scale = max(preprocess_context.scale, 1e-6)
    x1 = (box[0] - preprocess_context.pad_x) / scale
    y1 = (box[1] - preprocess_context.pad_y) / scale
    x2 = (box[2] - preprocess_context.pad_x) / scale
    y2 = (box[3] - preprocess_context.pad_y) / scale
    x1, y1, x2, y2 = normalize_box(
        (x1, y1, x2, y2),
        width=float(preprocess_context.original_width),
        height=float(preprocess_context.original_height),
    )
    return (
        clamp(x1, 0.0, float(preprocess_context.original_width)),
        clamp(y1, 0.0, float(preprocess_context.original_height)),
        clamp(x2, 0.0, float(preprocess_context.original_width)),
        clamp(y2, 0.0, float(preprocess_context.original_height)),
    )


def _canonical_class_from_model_name(class_name: str) -> int | None:
    return canonical_task1_class_from_model_name(class_name)
