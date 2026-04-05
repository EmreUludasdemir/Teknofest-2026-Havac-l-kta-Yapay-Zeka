from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings, OfficialRepoSettings
from src.pipeline.mvp_processor import MvpFrameProcessor
from src.pipeline.orchestrator import ProtocolOrchestrator
from src.pipeline.replay_runner import BatchManifestReplayRunner, ReplayOptions
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter
from src.task1.detector import Task1Detector
from tests.helpers import running_mock_server


class ExplodingDetector(Task1Detector):
    def detect(self, frame, image_bytes):  # type: ignore[override]
        raise RuntimeError("forced-task1-error")


class MvpProcessorTests(unittest.TestCase):
    def test_replay_smoke_with_all_tasks_connected(self) -> None:
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

    def test_task_failure_still_returns_valid_canonical_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.jpg").write_bytes(b"ref-1")
            runtime_settings = MvpRuntimeSettings(task3_reference_dir=temp_path)
            processor = MvpFrameProcessor(runtime_settings=runtime_settings)
            processor.task1_detector = ExplodingDetector(runtime_settings=runtime_settings)

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

            self.assertEqual(result.diagnostics["task1_status"], "fallback")
            self.assertEqual(result.detected_objects, [])
            self.assertEqual(len(result.detected_translations), 1)


if __name__ == "__main__":
    unittest.main()
