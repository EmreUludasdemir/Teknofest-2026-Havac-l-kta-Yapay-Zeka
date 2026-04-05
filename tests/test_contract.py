from __future__ import annotations

import unittest

from src.config.settings import OfficialRepoSettings
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter
from tests.helpers import running_mock_server


class ContractTests(unittest.TestCase):
    def test_login_fetch_and_send_prediction_against_mock_server(self) -> None:
        with running_mock_server(frame_count=2) as server:
            adapter = OfficialRepoBatchAdapter(
                OfficialRepoSettings(
                    base_url=server.base_url,
                    username="team",
                    password="password",
                )
            )
            token = adapter.login()
            self.assertEqual(token, "mock-token")

            session = adapter.fetch_session_state()
            self.assertEqual(len(session.frames), 2)
            self.assertEqual(session.frames[0].video_name, "mock_session")

            response = adapter.send_wire_prediction(
                {
                    "frame": session.frames[0].frame_url,
                    "detected_objects": [],
                    "detected_translations": [
                        {
                            "translation_x": "0.0",
                            "translation_y": "0.0",
                            "translation_z": "0.0",
                        }
                    ],
                }
            )
            self.assertEqual(response.status_code, 201)

            duplicate = adapter.send_wire_prediction(
                {
                    "frame": session.frames[0].frame_url,
                    "detected_objects": [],
                    "detected_translations": [
                        {
                            "translation_x": "0.0",
                            "translation_y": "0.0",
                            "translation_z": "0.0",
                        }
                    ],
                }
            )
            self.assertEqual(duplicate.status_code, 406)


if __name__ == "__main__":
    unittest.main()
