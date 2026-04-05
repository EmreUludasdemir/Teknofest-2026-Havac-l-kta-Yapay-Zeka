from __future__ import annotations

import unittest

from src.config.settings import SequentialProtocolSettings
from src.core.frame_state import FrameResult
from src.server.final_sequential_adapter import FinalSequentialAdapter
from src.server.protocol import ProtocolError
from tests.helpers import running_mock_server


class SequentialWarmupRetryTests(unittest.TestCase):
    def test_first_frame_waits_for_warmup_and_then_succeeds(self) -> None:
        with running_mock_server(
            frame_count=2,
            mode="sequential",
            sequential_warmup_delay_s=0.05,
            next_frame_retry_failures=1,
        ) as server:
            adapter = FinalSequentialAdapter(
                SequentialProtocolSettings(
                    base_url=server.base_url,
                    username="team",
                    password="password",
                    first_frame_timeout_s=0.2,
                    retry_policy={"max_retries": 4, "backoff_s": 0.02},
                )
            )
            adapter.login()
            adapter.open_session()
            frame = adapter.fetch_next_frame()
            self.assertIsNotNone(frame)
            self.assertTrue(adapter.warmup_completed)

    def test_timeout_retry_and_duplicate_prediction_guard(self) -> None:
        with running_mock_server(
            frame_count=1,
            mode="sequential",
            next_frame_timeout_failures=1,
            next_frame_timeout_delay_s=0.1,
            prediction_retry_failures=1,
        ) as server:
            adapter = FinalSequentialAdapter(
                SequentialProtocolSettings(
                    base_url=server.base_url,
                    username="team",
                    password="password",
                    first_frame_timeout_s=0.05,
                    request_timeout_s=0.05,
                    retry_policy={"max_retries": 3, "backoff_s": 0.01},
                )
            )
            adapter.login()
            adapter.open_session()
            frame = adapter.fetch_next_frame()
            self.assertIsNotNone(frame)
            payload = adapter.build_wire_prediction(
                FrameResult(
                    frame_url=frame.frame_url,
                    detected_translations=[frame.ground_truth_translation(source="test")],
                )
            )
            response = adapter.send_wire_prediction(payload)
            self.assertEqual(response.status_code, 201)
            with self.assertRaises(ProtocolError):
                adapter.send_wire_prediction(payload)


if __name__ == "__main__":
    unittest.main()
