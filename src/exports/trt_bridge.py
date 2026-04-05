from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.frame_state import CanonicalDetection, DecodedFrame
from src.exports.onnx_bridge import (
    decode_yolo_onnx_outputs,
    load_onnx_metadata,
    preprocess_task1_onnx_input,
)

try:  # pragma: no cover - opsiyonel bagimlilik
    import tensorrt as trt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - opsiyonel bagimlilik
    trt = None

try:  # pragma: no cover - opsiyonel bagimlilik
    import torch  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - opsiyonel bagimlilik
    torch = None


def is_tensorrt_runtime_available() -> bool:
    return trt is not None and torch is not None and bool(getattr(torch.cuda, "is_available", lambda: False)())


@dataclass(slots=True)
class TensorRtTask1Bridge:
    engine_path: str | Path
    metadata_path: str | Path
    conf_threshold: float
    iou_threshold: float
    max_objects_per_frame: int
    warmup_runs: int = 3
    logger_name: str = "TensorRtTask1Bridge"

    def __post_init__(self) -> None:
        self.engine_path = Path(self.engine_path)
        self.metadata_path = Path(self.metadata_path)
        self.metadata = load_onnx_metadata(self.metadata_path)
        self.engine = None
        self.runtime = None
        self.context = None
        self.provider_name = "TensorRT"
        self._io_names: dict[str, list[str]] = {"inputs": [], "outputs": []}

    def load(self) -> None:
        if self.context is not None:
            return
        if not is_tensorrt_runtime_available():
            raise RuntimeError("missing_dependency:tensorrt")
        if not self.engine_path.exists():
            raise FileNotFoundError(f"missing_trt_engine:{self.engine_path}")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(trt_logger)
        engine_bytes = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("trt_engine_deserialize_failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("trt_execution_context_failed")
        self._collect_io_names()
        self._warm_up()

    def unload(self) -> None:
        self.context = None
        self.engine = None
        self.runtime = None
        if torch is not None and bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()

    def predict(self, decoded_frame: DecodedFrame) -> tuple[list[CanonicalDetection], dict[str, Any]]:
        self.load()
        input_shape = tuple(int(item) for item in self.metadata.get("input_shape", [1, 3, 640, 640]))
        input_tensor, preprocess_context = preprocess_task1_onnx_input(decoded_frame, input_size=input_shape[-1])
        outputs = self._execute(input_tensor)
        detections = decode_yolo_onnx_outputs(
            outputs[0],
            preprocess_context=preprocess_context,
            class_names=list(self.metadata.get("class_names", [])),
            conf_threshold=float(self.conf_threshold),
            iou_threshold=float(self.iou_threshold),
            max_objects_per_frame=int(self.max_objects_per_frame),
            provider_name=self.provider_name,
            model_path=self.engine_path,
        )
        diagnostics = {
            "provider": self.provider_name,
            "input_shape": list(input_tensor.shape),
            "output_shapes": [list(item.shape) for item in outputs],
            "warmup_runs": self.warmup_runs,
        }
        return detections, diagnostics

    def _collect_io_names(self) -> None:
        if hasattr(self.engine, "num_io_tensors"):
            inputs: list[str] = []
            outputs: list[str] = []
            for index in range(int(self.engine.num_io_tensors)):
                tensor_name = str(self.engine.get_tensor_name(index))
                tensor_mode = self.engine.get_tensor_mode(tensor_name)
                if tensor_mode == trt.TensorIOMode.INPUT:
                    inputs.append(tensor_name)
                else:
                    outputs.append(tensor_name)
            self._io_names = {"inputs": inputs, "outputs": outputs}
            return
        raise RuntimeError("unsupported_tensorrt_api")

    def _warm_up(self) -> None:
        if self.context is None:
            return
        input_shape = tuple(int(item) for item in self.metadata.get("input_shape", [1, 3, 640, 640]))
        for _ in range(max(int(self.warmup_runs), 0)):
            zero_tensor = torch.zeros(input_shape, dtype=torch.float32, device="cuda")
            self._execute(zero_tensor.detach().cpu().numpy(), already_bchw=True)

    def _execute(self, input_tensor: Any, *, already_bchw: bool = False) -> list[Any]:
        if torch is None or self.context is None or self.engine is None:
            raise RuntimeError("trt_runtime_not_loaded")
        np_input = input_tensor if already_bchw else input_tensor
        torch_input = torch.from_numpy(np_input).to(device="cuda", dtype=torch.float32).contiguous()
        stream = torch.cuda.current_stream()

        input_name = self._io_names["inputs"][0]
        self.context.set_input_shape(input_name, tuple(int(item) for item in torch_input.shape))
        self.context.set_tensor_address(input_name, int(torch_input.data_ptr()))

        output_tensors: list[Any] = []
        for output_name in self._io_names["outputs"]:
            output_shape = tuple(int(item) for item in self.context.get_tensor_shape(output_name))
            output_dtype = _trt_dtype_to_torch_dtype(self.engine.get_tensor_dtype(output_name))
            output_tensor = torch.empty(output_shape, dtype=output_dtype, device="cuda")
            self.context.set_tensor_address(output_name, int(output_tensor.data_ptr()))
            output_tensors.append(output_tensor)

        ok = self.context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("trt_execute_failed")
        stream.synchronize()
        return [tensor.detach().cpu().numpy() for tensor in output_tensors]


def load_trt_metadata(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"missing_trt_metadata:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _trt_dtype_to_torch_dtype(dtype: Any) -> Any:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise RuntimeError(f"unsupported_trt_dtype:{dtype}")
    return mapping[dtype]
