from __future__ import annotations

import unittest

from src.config.settings import SequentialProtocolSettings
from src.core.frame_state import CanonicalUndefinedObject, FrameResult
from src.data.validators import SchemaValidator
from src.server.final_sequential_adapter import FinalSequentialAdapter
from src.server.protocol import ProtocolError
from tests.helpers import running_mock_server


class SequentialContractPhase9Tests(unittest.TestCase):
    def test_warmup_fetch_predict_fetch_order_and_exact_payload_profile(self) -> None:
        with running_mock_server(frame_count=2, mode="sequential") as server:
            adapter = FinalSequentialAdapter(
                SequentialProtocolSettings(base_url=server.base_url, username="team", password="password")
            )
            validator = SchemaValidator()
            adapter.login()
            opened = adapter.open_session()
            adapter.mark_warmup_completed({"active_task1_stage": "synthetic"})
            self.assertIn("session_id", opened)

            frame_1 = adapter.fetch_next_frame()
            self.assertIsNotNone(frame_1)
            with self.assertRaises(ProtocolError):
                adapter.fetch_next_frame()

            result = FrameResult(
                frame_url=frame_1.frame_url,
                detected_translations=[frame_1.ground_truth_translation(source="test")],
                detected_undefined_objects=[
                    CanonicalUndefinedObject(
                        object_id="ref-1",
                        top_left_x=1.0,
                        top_left_y=2.0,
                        bottom_right_x=3.0,
                        bottom_right_y=4.0,
                    )
                ],
            )
            payload = adapter.build_wire_prediction(result)
            self.assertNotIn("detected_undefined_objects", payload)
            validator.validate_sequential_prediction(payload)
            response = adapter.send_wire_prediction(payload)
            self.assertEqual(response.status_code, 201)

            frame_2 = adapter.fetch_next_frame()
            self.assertIsNotNone(frame_2)
            adapter.close_session()
            with self.assertRaises(ProtocolError):
                adapter.fetch_next_frame()

    def test_sequential_validator_rejects_undefined_objects_field(self) -> None:
        validator = SchemaValidator()
        with self.assertRaises(ValueError):
            validator.validate_sequential_prediction(
                {
                    "session_id": "seq-1",
                    "frame_id": 1,
                    "frame": "http://mock/frame/1/",
                    "detected_objects": [],
                    "detected_translations": [],
                    "detected_undefined_objects": [],
                }
            )

    def test_draft_wire_profile_allows_undefined_objects_when_enabled(self) -> None:
        with running_mock_server(frame_count=1, mode="sequential") as server:
            adapter = FinalSequentialAdapter(
                SequentialProtocolSettings(
                    base_url=server.base_url,
                    username="team",
                    password="password",
                    wire_profile="draft_with_undefined",
                )
            )
            validator = SchemaValidator()
            adapter.login()
            adapter.open_session()
            adapter.mark_warmup_completed({"active_task1_stage": "synthetic"})
            frame = adapter.fetch_next_frame()
            result = FrameResult(
                frame_url=frame.frame_url,
                detected_translations=[frame.ground_truth_translation(source="test")],
                detected_undefined_objects=[
                    CanonicalUndefinedObject(
                        object_id="ref-1",
                        top_left_x=10.0,
                        top_left_y=20.0,
                        bottom_right_x=30.0,
                        bottom_right_y=40.0,
                    )
                ],
            )
            payload = adapter.build_wire_prediction(result)
            self.assertIn("detected_undefined_objects", payload)
            validator.validate_sequential_prediction(payload, profile="draft_with_undefined")


if __name__ == "__main__":
    unittest.main()
