from __future__ import annotations

import unittest

from src.config.settings import SequentialProtocolSettings
from src.core.frame_state import FrameResult
from src.server.final_sequential_adapter import FinalSequentialAdapter
from src.server.protocol import ProtocolError
from tests.helpers import running_mock_server


class SequentialMockContractTests(unittest.TestCase):
    def test_open_fetch_predict_close_contract(self) -> None:
        with running_mock_server(frame_count=2, mode="sequential") as server:
            adapter = FinalSequentialAdapter(
                SequentialProtocolSettings(base_url=server.base_url, username="team", password="password")
            )
            token = adapter.login()
            self.assertEqual(token, "mock-token")
            opened = adapter.open_session()
            self.assertIn("session_id", opened)

            frame_1 = adapter.fetch_next_frame()
            self.assertIsNotNone(frame_1)
            with self.assertRaises(ProtocolError):
                adapter.fetch_next_frame()

            result = FrameResult(
                frame_url=frame_1.frame_url,
                detected_translations=[frame_1.ground_truth_translation(source="test")],
            )
            payload = adapter.build_wire_prediction(result)
            response = adapter.send_wire_prediction(payload)
            self.assertEqual(response.status_code, 201)

            frame_2 = adapter.fetch_next_frame()
            self.assertIsNotNone(frame_2)
            adapter.close_session()


if __name__ == "__main__":
    unittest.main()
