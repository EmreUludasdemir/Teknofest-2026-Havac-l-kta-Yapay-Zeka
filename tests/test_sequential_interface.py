from __future__ import annotations

import unittest

from src.config.settings import OfficialRepoSettings, SequentialProtocolSettings
from src.server.final_sequential_adapter import FinalSequentialAdapter, SequentialProtocolAdapter
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter


class SequentialInterfaceTests(unittest.TestCase):
    def test_final_sequential_adapter_is_import_safe(self) -> None:
        settings = SequentialProtocolSettings(base_url="http://mock/", username="team", password="password")
        adapter = FinalSequentialAdapter(settings)
        self.assertIsInstance(adapter, SequentialProtocolAdapter)
        self.assertEqual(adapter.url_open_session, "http://mock/session/open/")
        self.assertEqual(adapter.url_next_frame, "http://mock/session/next/")

    def test_sequential_profile_overrides_paths_without_touching_batch_adapter(self) -> None:
        settings = SequentialProtocolSettings(
            base_url="http://mock/",
            username="team",
            password="password",
            profile_name="final_profile",
            path_overrides={"open_session": "final/open", "next_frame": "final/next"},
        )
        adapter = FinalSequentialAdapter(settings)
        self.assertEqual(adapter.settings.profile_name, "final_profile")
        self.assertEqual(adapter.url_open_session, "http://mock/final/open/")
        self.assertEqual(adapter.url_next_frame, "http://mock/final/next/")

    def test_batch_adapter_remains_separate_from_sequential_interface(self) -> None:
        batch_adapter = OfficialRepoBatchAdapter(
            OfficialRepoSettings(base_url="http://mock/", username="team", password="password")
        )
        self.assertNotIsInstance(batch_adapter, SequentialProtocolAdapter)


if __name__ == "__main__":
    unittest.main()
