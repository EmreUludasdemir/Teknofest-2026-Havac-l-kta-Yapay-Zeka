from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logger import StructuredLogger
from src.data.validators import SchemaValidator
from src.pipeline.mvp_processor import MvpFrameProcessor
from src.server.final_sequential_adapter import FinalSequentialAdapter
from src.server.protocol import ProtocolError
from src.tools.runtime_bootstrap import execute_runtime_warmup
from src.tools.runtime_package import load_runtime_bootstrap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TEKNOFEST competition runtime")
    parser.add_argument("--config", default="final_runtime/config/runtime.toml")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    args = parser.parse_args(argv)

    bootstrap = load_runtime_bootstrap(
        args.config,
        production=True,
        base_url_override=args.base_url,
        username_override=args.username,
        password_override=args.password,
    )
    if str(bootstrap.runtime_config.get("runtime", {}).get("mode", "sequential")).lower() != "sequential":
        raise SystemExit("Production runtime yalniz sequential modda calisabilir.")
    log_dir = Path(bootstrap.runtime_config.get("paths", {}).get("log_dir", "final_runtime/logs"))
    logger = StructuredLogger(log_dir=log_dir)
    validator = SchemaValidator()
    processor = MvpFrameProcessor(runtime_settings=bootstrap.runtime_settings)
    adapter = FinalSequentialAdapter(bootstrap.sequential_settings, logger=logger)
    max_frames = args.max_frames
    if max_frames is None:
        runtime_max_frames = bootstrap.runtime_config.get("runtime", {}).get("max_frames")
        max_frames = int(runtime_max_frames) if runtime_max_frames not in {None, ""} else None

    processed_frames = 0
    session_payload: dict[str, object] = {}
    warmup_payload: dict[str, object] = {}
    try:
        adapter.login()
        session_payload = adapter.open_session()
        warmup_payload = execute_runtime_warmup(
            processor=processor,
            logger=logger,
            session_payload=session_payload,
            session_name=str(session_payload.get("session_name", "competition_session")),
            adapter=adapter,
        )
        logger.log_runtime(
            event="active_task1_stage",
            adapter=type(adapter).__name__,
            session_name=str(session_payload.get("session_name", "")),
            fallback_mode=str(warmup_payload.get("active_task1_stage")),
            diagnostics=warmup_payload,
        )
        while True:
            if max_frames is not None and processed_frames >= max_frames:
                break
            frame = adapter.fetch_next_frame()
            if frame is None:
                break
            image_bytes = adapter.download_image(frame)
            result = processor(frame, image_bytes)
            validator.validate_canonical_result(result.to_canonical_dict())
            payload = adapter.build_wire_prediction(result)
            validator.validate_sequential_prediction(
                payload,
                profile=adapter.settings.wire_profile,
            )
            try:
                adapter.send_wire_prediction(payload)
            except Exception as exc:
                logger.log_error(
                    event="competition_prediction_failed",
                    adapter=type(adapter).__name__,
                    session_name=str(session_payload.get("session_name", "")),
                    frame_url=frame.frame_url,
                    health_status=frame.health_status,
                    fallback_mode="prediction_submit",
                    diagnostics={"error": str(exc), "active_task1_stage": warmup_payload.get("active_task1_stage")},
                )
                break
            processed_frames += 1
    except ProtocolError as exc:
        logger.log_error(
            event="competition_runtime_protocol_failed",
            adapter="FinalSequentialAdapter",
            session_name=str(session_payload.get("session_name", "")) if session_payload else None,
            diagnostics={"error": str(exc)},
        )
        return 1
    finally:
        try:
            adapter.close_session()
        except Exception:
            pass
        summary_path = log_dir / "competition_runtime_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "processed_frames": processed_frames,
                    "active_task1_stage": warmup_payload.get("active_task1_stage"),
                    "wire_profile": adapter.settings.wire_profile,
                    "session_id": session_payload.get("session_id"),
                    "session_name": session_payload.get("session_name"),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
