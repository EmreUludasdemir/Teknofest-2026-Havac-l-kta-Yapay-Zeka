from __future__ import annotations

import unittest

from src.config.settings import OfficialRepoSettings
from src.pipeline.orchestrator import ProtocolOrchestrator
from src.pipeline.replay_runner import BatchManifestReplayRunner, ReplayOptions
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter
from tests.helpers import running_mock_server


class ReplayRunnerTests(unittest.TestCase):
    def test_batch_manifest_replay_smoke(self) -> None:
        with running_mock_server(frame_count=3) as server:
            adapter = OfficialRepoBatchAdapter(
                OfficialRepoSettings(
                    base_url=server.base_url,
                    username="team",
                    password="password",
                )
            )
            orchestrator = ProtocolOrchestrator(adapter)
            runner = BatchManifestReplayRunner(adapter, orchestrator)

            summary = runner.run(ReplayOptions(max_frames=2, submit_predictions=True))

            self.assertEqual(summary.frames_seen, 2)
            self.assertEqual(summary.frames_submitted, 2)
            self.assertEqual(summary.submission_failures, 0)


if __name__ == "__main__":
    unittest.main()
