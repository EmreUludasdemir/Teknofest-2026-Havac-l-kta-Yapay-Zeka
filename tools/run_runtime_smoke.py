from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings, OfficialRepoSettings
from src.core.logger import StructuredLogger
from src.data.validators import SchemaValidator
from src.pipeline.mvp_processor import MvpFrameProcessor
from src.pipeline.orchestrator import ProtocolOrchestrator
from src.pipeline.replay_runner import BatchManifestReplayRunner, ReplayOptions
from src.server.final_sequential_adapter import FinalSequentialAdapter
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter
from src.tools.mock_server import OfficialRepoMockServer
from src.tools.runtime_bootstrap import execute_runtime_warmup
from src.tools.runtime_package import load_runtime_bootstrap, prepare_runtime_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TEKNOFEST final runtime smoke")
    parser.add_argument("--mode", choices=("batch", "sequential"), default="batch")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--base-dir", default="final_runtime")
    parser.add_argument("--reports-dir", default="reports/export")
    args = parser.parse_args(argv)

    package_settings = MvpRuntimeSettings(
        task1_candidate_paths={
            "yolo26n": str(Path.home() / ".teknofest_models" / "yolo26n.pt"),
            "yolo11n": str(Path.home() / ".teknofest_models" / "yolo11n.pt"),
        },
        task1_onnx_candidate_paths={
            "yolo26n": str(Path(args.reports_dir) / "models" / "yolo26n.onnx"),
            "yolo11n": str(Path(args.reports_dir) / "models" / "yolo11n.onnx"),
        },
        task1_trt_candidate_paths={
            "yolo26n": str(Path(args.reports_dir) / "models" / "yolo26n.engine"),
        },
    )
    package = prepare_runtime_package(package_settings, base_dir=args.base_dir, reports_dir=args.reports_dir)
    bootstrap = load_runtime_bootstrap(Path(package["config_path"]), production=True)
    logger = StructuredLogger(log_dir=Path(args.base_dir) / "logs")
    summary_path = Path(args.base_dir) / "logs" / "runtime_smoke_summary.json"

    if args.mode == "batch":
        with _mock_server(mode="batch") as server:
            adapter = OfficialRepoBatchAdapter(
                OfficialRepoSettings(base_url=server.base_url, username="team", password="password"),
                logger=logger,
            )
            orchestrator = ProtocolOrchestrator(
                adapter,
                frame_processor=MvpFrameProcessor(runtime_settings=bootstrap.runtime_settings),
                validator=SchemaValidator(),
                logger=logger,
                runtime_settings=bootstrap.runtime_settings,
            )
            runner = BatchManifestReplayRunner(adapter, orchestrator)
            summary = runner.run(ReplayOptions(max_frames=args.frames, submit_predictions=True))
            summary_path.write_text(json.dumps(asdict(summary), indent=2, default=str), encoding="utf-8")
            return 0

    with _mock_server(mode="sequential", sequential_warmup_delay_s=0.05) as server:
        sequential_settings = bootstrap.sequential_settings
        sequential_settings.base_url = server.base_url
        adapter = FinalSequentialAdapter(sequential_settings, logger=logger)
        processor = MvpFrameProcessor(runtime_settings=bootstrap.runtime_settings)
        validator = SchemaValidator()
        diagnostics: list[dict[str, object]] = []
        adapter.login()
        session_payload = adapter.open_session()
        warmup = execute_runtime_warmup(
            processor=processor,
            logger=logger,
            session_payload=session_payload,
            session_name=str(session_payload.get("session_name", "smoke_session")),
            adapter=adapter,
        )
        try:
            for _ in range(args.frames):
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
                response = adapter.send_wire_prediction(payload)
                diagnostics.append(
                    {
                        "frame": frame.frame_url,
                        "status_code": response.status_code,
                        "objects": len(result.detected_objects),
                        "translations": len(result.detected_translations),
                        "active_task1_stage": warmup.get("active_task1_stage"),
                    }
                )
        finally:
            adapter.close_session()
        summary_path.write_text(
            json.dumps({"mode": "sequential", "warmup": warmup, "diagnostics": diagnostics}, indent=2),
            encoding="utf-8",
        )
    return 0


from contextlib import contextmanager


@contextmanager
def _mock_server(**kwargs):
    server = OfficialRepoMockServer(**kwargs)
    server.start()
    try:
        yield server
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
