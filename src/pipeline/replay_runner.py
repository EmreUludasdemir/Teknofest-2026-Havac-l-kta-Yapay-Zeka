from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from src.core.session_state import ReplaySummary
from src.pipeline.orchestrator import ProtocolOrchestrator
from src.server.protocol import ProtocolAdapter


@dataclass(slots=True)
class ReplayOptions:
    max_frames: int | None = None
    submit_predictions: bool = True


class BatchManifestReplayRunner:
    def __init__(self, adapter: ProtocolAdapter, orchestrator: ProtocolOrchestrator) -> None:
        self.adapter = adapter
        self.orchestrator = orchestrator
        self.logger = orchestrator.logger

    def run(self, options: ReplayOptions | None = None) -> ReplaySummary:
        options = options or ReplayOptions()
        session = self.adapter.fetch_session_state()
        summary = ReplaySummary(session_name=session.session_name)

        for index, frame in enumerate(session.frames):
            if options.max_frames is not None and index >= options.max_frames:
                break

            t0 = perf_counter()
            summary.frames_seen += 1
            try:
                result, wire_payload = self.orchestrator.prepare_prediction(frame)
            except Exception:
                summary.validation_failures += 1
                raise
            diagnostics = {
                "frame": frame.frame_url,
                "health_status": frame.health_status,
                "detected_objects": len(result.detected_objects),
                "detected_translations": len(result.detected_translations),
            }

            status_code = None
            if options.submit_predictions:
                response = self.adapter.send_wire_prediction(wire_payload)
                status_code = response.status_code
                diagnostics["status_code"] = status_code
                if status_code == 201:
                    summary.frames_submitted += 1
                else:
                    summary.submission_failures += 1

            latency_ms = round((perf_counter() - t0) * 1000.0, 3)
            self.logger.log_runtime(
                event="frame_processed",
                adapter=type(self.adapter).__name__,
                session_name=session.session_name,
                frame_url=frame.frame_url,
                video_name=frame.video_name,
                health_status=frame.health_status,
                status_code=status_code,
                latency_ms=latency_ms,
                detected_objects=len(result.detected_objects),
                detected_translations=len(result.detected_translations),
                fallback_mode=str(result.diagnostics.get("fallback_mode")) if result.diagnostics.get("fallback_mode") else None,
                diagnostics={**diagnostics, **result.diagnostics},
            )
            summary.diagnostics.append(diagnostics)

        return summary
