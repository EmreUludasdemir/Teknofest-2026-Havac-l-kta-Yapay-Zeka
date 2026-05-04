from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings, OfficialRepoSettings
from src.pipeline.mvp_processor import MvpFrameProcessor, Task3OnlyProcessor
from src.pipeline.orchestrator import ProtocolOrchestrator
from src.pipeline.replay_runner import BatchManifestReplayRunner, ReplayOptions
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter
from tests.helpers import running_mock_server


class _ExplodingMatcher:
    def __init__(self) -> None:
        self.last_run_info: dict[str, object] = {}

    def match(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise RuntimeError("forced-task3-error")


class Task3ProcessorTests(unittest.TestCase):
    def test_replay_smoke_with_task3_only_processor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.jpg").write_bytes(b"ref-1")
            runtime_settings = MvpRuntimeSettings(task3_reference_dir=temp_path)
            with running_mock_server(frame_count=3) as server:
                adapter = OfficialRepoBatchAdapter(
                    OfficialRepoSettings(base_url=server.base_url, username="team", password="password")
                )
                processor = MvpFrameProcessor(runtime_settings=runtime_settings)
                orchestrator = ProtocolOrchestrator(adapter, frame_processor=processor, runtime_settings=runtime_settings)
                runner = BatchManifestReplayRunner(adapter, orchestrator)
                summary = runner.run(ReplayOptions(max_frames=2, submit_predictions=True))

            self.assertEqual(summary.frames_seen, 2)
            self.assertEqual(summary.frames_submitted, 2)

    def test_task3_failure_still_returns_valid_task3_only_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.jpg").write_bytes(b"ref-1")
            runtime_settings = MvpRuntimeSettings(task3_reference_dir=temp_path)
            processor = Task3OnlyProcessor(runtime_settings=runtime_settings)
            processor.task3_matcher = _ExplodingMatcher()  # type: ignore[assignment]

            from src.core.frame_state import FrameEnvelope

            frame = FrameEnvelope(
                frame_url="http://mock/frames/2/",
                image_url="/frame.jpg",
                video_name="session",
                translation_x=1.0,
                translation_y=2.0,
                translation_z=3.0,
                health_status="1",
            )
            result = processor(frame, b"img")

            self.assertEqual(result.diagnostics["task3_status"], "fallback")
            self.assertNotIn("task1_status", result.diagnostics)
            self.assertNotIn("task2_status", result.diagnostics)
            self.assertEqual(result.detected_objects, [])
            self.assertEqual(result.detected_translations, [])
            self.assertEqual(result.detected_undefined_objects, [])


if __name__ == "__main__":
    unittest.main()
