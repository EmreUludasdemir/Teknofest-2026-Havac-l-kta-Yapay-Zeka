from __future__ import annotations

import unittest

from src.config.settings import OfficialRepoSettings
from src.core.frame_state import CanonicalDetection, CanonicalTranslation, FrameEnvelope, FrameResult
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter
from tests.helpers import running_mock_server


class WireFormatComplianceTests(unittest.TestCase):
    def test_cls_url_and_all_string_conversion_follow_official_2025_format(self) -> None:
        adapter = OfficialRepoBatchAdapter(
            OfficialRepoSettings(base_url="http://mock/", username="team", password="password")
        )
        result = FrameResult(
            frame_url="http://mock/frames/1/",
            detected_objects=[
                CanonicalDetection(
                    class_id=0,
                    landing_status=-1,
                    motion_status=1,
                    top_left_x=10.5,
                    top_left_y=20.25,
                    bottom_right_x=30.75,
                    bottom_right_y=40.0,
                )
            ],
            detected_translations=[
                CanonicalTranslation(
                    translation_x=1.25,
                    translation_y=2.5,
                    translation_z=3.75,
                    source="test",
                )
            ],
        )

        payload = adapter.build_wire_prediction(result)

        self.assertEqual(payload["frame"], "http://mock/frames/1/")
        self.assertEqual(payload["detected_objects"][0]["cls"], "http://mock/classes/1/")
        self.assertEqual(payload["detected_objects"][0]["landing_status"], "-1")
        self.assertEqual(payload["detected_objects"][0]["top_left_x"], "10.5")
        self.assertEqual(payload["detected_translations"][0]["translation_z"], "3.75")
        self.assertNotIn("motion_status", payload["detected_objects"][0])
        self.assertNotIn("detected_undefined_objects", payload)

    def test_empty_arrays_and_health_status_field_are_batch_compatible(self) -> None:
        with running_mock_server(frame_count=1) as server:
            adapter = OfficialRepoBatchAdapter(
                OfficialRepoSettings(base_url=server.base_url, username="team", password="password")
            )
            session = adapter.fetch_session_state()
            frame = session.frames[0]
            self.assertEqual(frame.health_status, "1")

            payload = adapter.build_wire_prediction(FrameResult(frame_url=frame.frame_url))
            self.assertEqual(payload["detected_objects"], [])
            self.assertEqual(payload["detected_translations"], [])

    def test_prediction_rate_limit_guard_waits_before_second_send(self) -> None:
        with running_mock_server(frame_count=2, prediction_limit_per_minute=1000) as server:
            adapter = OfficialRepoBatchAdapter(
                OfficialRepoSettings(
                    base_url=server.base_url,
                    username="team",
                    password="password",
                    prediction_limit_per_minute=1,
                )
            )
            current_time = [0.0]
            sleeps: list[float] = []

            def fake_time() -> float:
                return current_time[0]

            def fake_sleep(seconds: float) -> None:
                sleeps.append(seconds)
                current_time[0] += seconds

            adapter._time_fn = fake_time
            adapter._sleep_fn = fake_sleep
            session = adapter.fetch_session_state()
            payload_1 = {
                "frame": session.frames[0].frame_url,
                "detected_objects": [],
                "detected_translations": [],
            }
            payload_2 = {
                "frame": session.frames[1].frame_url,
                "detected_objects": [],
                "detected_translations": [],
            }

            first = adapter.send_wire_prediction(payload_1)
            second = adapter.send_wire_prediction(payload_2)

            self.assertEqual(first.status_code, 201)
            self.assertEqual(second.status_code, 201)
            self.assertEqual(len(sleeps), 1)
            self.assertGreaterEqual(sleeps[0], 60.0)


if __name__ == "__main__":
    unittest.main()
