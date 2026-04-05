from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config.settings import OfficialRepoSettings
from src.core.logger import StructuredLogger
from src.pipeline.orchestrator import ProtocolOrchestrator
from src.pipeline.replay_runner import BatchManifestReplayRunner, ReplayOptions
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter
from tests.helpers import running_mock_server


class LoggerTests(unittest.TestCase):
    def test_logger_writes_runtime_and_error_records_with_fixed_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = StructuredLogger(log_dir=Path(tmp_dir))
            logger.log_runtime(event="runtime_event", adapter="AdapterA", diagnostics={"ok": True})
            logger.log_error(event="error_event", adapter="AdapterA", diagnostics={"ok": False})

            runtime_records = [json.loads(line) for line in (Path(tmp_dir) / "runtime.jsonl").read_text(encoding="utf-8").splitlines()]
            error_records = [json.loads(line) for line in (Path(tmp_dir) / "error.jsonl").read_text(encoding="utf-8").splitlines()]

            expected_keys = {
                "timestamp",
                "level",
                "event",
                "adapter",
                "session_name",
                "frame_url",
                "video_name",
                "health_status",
                "status_code",
                "latency_ms",
                "detected_objects",
                "detected_translations",
                "fallback_mode",
                "diagnostics",
            }

            self.assertEqual(set(runtime_records[0].keys()), expected_keys)
            self.assertEqual(set(error_records[0].keys()), expected_keys)
            self.assertEqual(runtime_records[0]["event"], "runtime_event")
            self.assertEqual(error_records[0]["event"], "error_event")

    def test_replay_logger_integration_does_not_break_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = StructuredLogger(log_dir=Path(tmp_dir))
            with running_mock_server(frame_count=1) as server:
                adapter = OfficialRepoBatchAdapter(
                    OfficialRepoSettings(
                        base_url=server.base_url,
                        username="team",
                        password="password",
                    ),
                    logger=logger,
                )
                orchestrator = ProtocolOrchestrator(adapter, logger=logger)
                runner = BatchManifestReplayRunner(adapter, orchestrator)

                summary = runner.run(ReplayOptions(max_frames=1, submit_predictions=True))

            runtime_records = [json.loads(line) for line in (Path(tmp_dir) / "runtime.jsonl").read_text(encoding="utf-8").splitlines()]
            events = {record["event"] for record in runtime_records}
            self.assertEqual(summary.frames_submitted, 1)
            self.assertIn("frame_processed", events)
            self.assertIn("prediction_prepared", events)


if __name__ == "__main__":
    unittest.main()
