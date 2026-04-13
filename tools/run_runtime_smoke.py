from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings, OfficialRepoSettings, SequentialProtocolSettings
from src.core.logger import StructuredLogger
from src.data.validators import SchemaValidator
from src.pipeline.mvp_processor import MvpFrameProcessor
from src.pipeline.orchestrator import ProtocolOrchestrator
from src.pipeline.replay_runner import BatchManifestReplayRunner, ReplayOptions
from src.server.final_sequential_adapter import FinalSequentialAdapter
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter
from src.tools.mock_server import OfficialRepoMockServer
from src.tools.report_paths import EXPORT_REPORTS_DIR
from src.tools.runtime_package import prepare_runtime_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TEKNOFEST final runtime smoke")
    parser.add_argument("--mode", choices=("batch", "sequential"), default="batch")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--base-dir", default="final_runtime")
    parser.add_argument("--reports-dir", default=str(EXPORT_REPORTS_DIR))
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
    runtime_config = _load_runtime_toml(Path(package["config_path"]))
    runtime_settings = _build_runtime_settings(runtime_config, manifest_path=Path(package["manifest_path"]))
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
                frame_processor=MvpFrameProcessor(runtime_settings=runtime_settings),
                validator=SchemaValidator(),
                logger=logger,
                runtime_settings=runtime_settings,
            )
            runner = BatchManifestReplayRunner(adapter, orchestrator)
            summary = runner.run(ReplayOptions(max_frames=args.frames, submit_predictions=True))
            summary_path.write_text(json.dumps(asdict(summary), indent=2, default=str), encoding="utf-8")
            return 0

    with _mock_server(mode="sequential", sequential_warmup_delay_s=0.05) as server:
        adapter = FinalSequentialAdapter(
            SequentialProtocolSettings(base_url=server.base_url, username="team", password="password"),
            logger=logger,
        )
        processor = MvpFrameProcessor(runtime_settings=runtime_settings)
        validator = SchemaValidator()
        diagnostics: list[dict[str, object]] = []
        adapter.login()
        adapter.open_session()
        try:
            for _ in range(args.frames):
                frame = adapter.fetch_next_frame()
                if frame is None:
                    break
                image_bytes = adapter.download_image(frame)
                result = processor(frame, image_bytes)
                validator.validate_canonical_result(result.to_canonical_dict())
                payload = adapter.build_wire_prediction(result)
                validator.validate_official_repo_prediction(payload)
                response = adapter.send_wire_prediction(payload)
                diagnostics.append(
                    {
                        "frame": frame.frame_url,
                        "status_code": response.status_code,
                        "objects": len(result.detected_objects),
                        "translations": len(result.detected_translations),
                    }
                )
        finally:
            adapter.close_session()
        summary_path.write_text(json.dumps({"mode": "sequential", "diagnostics": diagnostics}, indent=2), encoding="utf-8")
    return 0


def _load_runtime_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _build_runtime_settings(runtime_config: dict, *, manifest_path: Path) -> MvpRuntimeSettings:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task1 = runtime_config.get("task1", {})
    manifest_items = manifest.get("required", []) + manifest.get("optional", [])
    task1_paths = {
        item["candidate"]: item["path"]
        for item in manifest_items
        if item.get("runtime") == "ultralytics" and item.get("path")
    }
    onnx_paths = {
        item["candidate"]: item["path"]
        for item in manifest_items
        if item.get("runtime") == "onnxruntime"
        and item.get("path")
        and item.get("validation_status") is True
    }
    trt_paths = {
        item["candidate"]: item["path"]
        for item in manifest_items
        if item.get("runtime") == "tensorrt"
        and item.get("path")
        and item.get("validation_status") is True
    }
    return MvpRuntimeSettings(
        task1_detector_backend=task1.get("detector_backend", "yolo26n"),
        task1_model_runtime=task1.get("model_runtime", "onnxruntime"),
        task1_device=task1.get("task1_device", "cuda:0"),
        task1_trt_precision=task1.get("trt_precision", "fp16"),
        task1_trt_warmup_runs=int(task1.get("trt_warmup_runs", 3)),
        task1_onnx_conf_threshold_offset=float(task1.get("onnx_conf_threshold_offset", 0.10)),
        task1_runtime_order=list(task1.get("runtime_order", [])),
        task1_candidate_paths=task1_paths,
        task1_onnx_candidate_paths=onnx_paths,
        task1_trt_candidate_paths=trt_paths,
    )


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
