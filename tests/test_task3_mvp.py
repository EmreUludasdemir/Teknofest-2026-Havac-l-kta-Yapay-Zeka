from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings, OfficialRepoSettings
from src.core.frame_state import FrameEnvelope
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches


class Task3MvpTests(unittest.TestCase):
    def _frame(self, frame_id: int) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url=f"http://mock/frames/{frame_id}/",
            image_url="/frame.jpg",
            video_name="session",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
        )

    def test_reference_cache_preload_and_match_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.jpg").write_bytes(b"ref-1")
            (temp_path / "ref-002.jpg").write_bytes(b"ref-2")

            cache = ReferenceCache()
            loaded = cache.preload_from_directory(temp_path)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=MvpRuntimeSettings(task3_reference_dir=temp_path))
            matches = matcher.match(self._frame(1), b"img")
            filtered = filter_no_match_candidates(matches, min_score=0.70, ambiguity_margin=0.05)
            verified = verify_matches(self._frame(1), filtered)

            self.assertEqual(loaded, 2)
            self.assertGreaterEqual(len(matches), 1)
            self.assertLessEqual(len(verified), 1)
            if verified:
                self.assertEqual(verified[0].metadata["verification_status"], "placeholder_pass")

    def test_ambiguity_rule_suppresses_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.jpg").write_bytes(b"ref-1")
            (temp_path / "ref-002.jpg").write_bytes(b"ref-2")
            cache = ReferenceCache()
            cache.preload_from_directory(temp_path)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=MvpRuntimeSettings(task3_reference_dir=temp_path))
            ambiguous_matches = matcher.match(self._frame(5), b"img")
            filtered = filter_no_match_candidates(ambiguous_matches, min_score=0.70, ambiguity_margin=0.05)
            self.assertEqual(filtered, [])

    def test_official_wire_includes_undefined_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.jpg").write_bytes(b"ref-1")
            cache = ReferenceCache()
            cache.preload_from_directory(temp_path)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=MvpRuntimeSettings(task3_reference_dir=temp_path))
            matches = matcher.match(self._frame(1), b"img")
            filtered = filter_no_match_candidates(matches)
            adapter = OfficialRepoBatchAdapter(
                OfficialRepoSettings(base_url="http://mock/", username="team", password="password")
            )
            from src.core.frame_state import FrameResult

            result = FrameResult(frame_url=self._frame(1).frame_url, detected_undefined_objects=filtered)
            payload = adapter.build_wire_prediction(result)
            self.assertIn("detected_undefined_objects", payload)
            self.assertEqual(len(payload["detected_undefined_objects"]), len(filtered))
            if filtered:
                self.assertEqual(payload["detected_undefined_objects"][0]["object_id"], filtered[0].object_id)


if __name__ == "__main__":
    unittest.main()
