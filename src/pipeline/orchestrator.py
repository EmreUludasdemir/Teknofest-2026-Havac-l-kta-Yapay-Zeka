from __future__ import annotations

from time import perf_counter
from typing import Callable

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope, FrameResult
from src.core.logger import StructuredLogger
from src.data.validators import SchemaValidator
from src.pipeline.mvp_processor import MvpFrameProcessor
from src.server.protocol import ProtocolAdapter

FrameProcessor = Callable[[FrameEnvelope, bytes], FrameResult]


class ProtocolOrchestrator:
    def __init__(
        self,
        adapter: ProtocolAdapter,
        *,
        frame_processor: FrameProcessor | None = None,
        validator: SchemaValidator | None = None,
        logger: StructuredLogger | None = None,
        runtime_settings: MvpRuntimeSettings | None = None,
    ) -> None:
        self.adapter = adapter
        self.runtime_settings = runtime_settings or MvpRuntimeSettings()
        self.validator = validator or SchemaValidator()
        self.logger = logger or StructuredLogger()
        self.frame_processor = frame_processor or MvpFrameProcessor(runtime_settings=self.runtime_settings)

    def prepare_prediction(self, frame: FrameEnvelope) -> tuple[FrameResult, dict]:
        t0 = perf_counter()
        try:
            image_bytes = self.adapter.download_image(frame)
            result = self.frame_processor(frame, image_bytes)
            canonical_payload = result.to_canonical_dict()
            self.validator.validate_canonical_result(canonical_payload)
            wire_payload = self.adapter.build_wire_prediction(result)
            self.validator.validate_official_repo_prediction(wire_payload)
            latency_ms = round((perf_counter() - t0) * 1000.0, 3)
            self.logger.log_runtime(
                event="prediction_prepared",
                adapter=type(self.adapter).__name__,
                session_name=frame.video_name,
                frame_url=frame.frame_url,
                video_name=frame.video_name,
                health_status=frame.health_status,
                latency_ms=latency_ms,
                detected_objects=len(result.detected_objects),
                detected_translations=len(result.detected_translations),
                fallback_mode=str(result.diagnostics.get("fallback_mode")) if result.diagnostics.get("fallback_mode") else None,
                diagnostics={"stage": "prepare_prediction", **result.diagnostics},
            )
            return result, wire_payload
        except Exception as exc:
            latency_ms = round((perf_counter() - t0) * 1000.0, 3)
            self.logger.log_error(
                event="prediction_prepare_failed",
                adapter=type(self.adapter).__name__,
                session_name=frame.video_name,
                frame_url=frame.frame_url,
                video_name=frame.video_name,
                health_status=frame.health_status,
                latency_ms=latency_ms,
                diagnostics={"error": str(exc)},
            )
            raise
