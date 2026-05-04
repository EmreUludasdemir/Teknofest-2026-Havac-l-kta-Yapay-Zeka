from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame, FrameEnvelope
from src.pipeline.mvp_processor import Task3OnlyProcessor
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches


class Task3ScaffoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = FrameEnvelope(
            frame_url="http://mock/frames/1/",
            image_url="/frame.jpg",
            video_name="session",
            translation_x=1.0,
            translation_y=2.0,
            translation_z=3.0,
            health_status="1",
        )
        self.image_bytes = b"placeholder-image"
        self.decoded = DecodedFrame(
            bgr=None,
            gray=None,
            width=640,
            height=512,
            channel_count=0,
            modality="thermal",
            frame_index=1,
        )

    def test_task3_scaffolds_return_expected_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.jpg").write_bytes(b"ref-data")
            cache = ReferenceCache()
            cache.preload_from_directory(temp_path)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=MvpRuntimeSettings(task3_reference_dir=temp_path))
            matches = matcher.match(self.frame, self.image_bytes, ["ref-001"])
            processor = Task3OnlyProcessor(runtime_settings=MvpRuntimeSettings(task3_reference_dir=temp_path))

        self.assertIsInstance(matches, list)
        self.assertEqual(verify_matches(self.frame, matches), matches)
        self.assertEqual(filter_no_match_candidates(matches), matches)
        cache.put("ref-001", {"source": "test"})
        self.assertEqual(cache.get("ref-001"), {"source": "test"})
        self.assertTrue(all(isinstance(item, CanonicalUndefinedObject) for item in matches))
        self.assertIsInstance(processor, Task3OnlyProcessor)


if __name__ == "__main__":
    unittest.main()
