from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope
from src.core.vision import decode_image_bytes, is_cv2_available
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches


def _pgm_bytes(width: int, height: int, pixels: bytes) -> bytes:
    return f"P5\n{width} {height}\n255\n".encode("ascii") + pixels


def _reference_pattern(width: int, height: int) -> bytearray:
    values = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            inside = 8 <= x < width - 8 and 8 <= y < height - 8
            diagonal = x == y or x + y == width - 1
            checker = (x // 4 + y // 4) % 2 == 0
            if inside and checker:
                values[y * width + x] = 200
            if diagonal:
                values[y * width + x] = 255
    return values


def _pattern_image(width: int, height: int, *, embed_at: tuple[int, int] | None = None) -> bytes:
    if embed_at is None:
        return _pgm_bytes(width, height, bytes(_reference_pattern(width, height)))

    values = bytearray(width * height)
    ref_width = 64
    ref_height = 64
    pattern = _reference_pattern(ref_width, ref_height)
    offset_x, offset_y = embed_at
    for y in range(ref_height):
        for x in range(ref_width):
            target_x = x + offset_x
            target_y = y + offset_y
            if 0 <= target_x < width and 0 <= target_y < height:
                values[target_y * width + target_x] = pattern[y * ref_width + x]
    return _pgm_bytes(width, height, bytes(values))


@unittest.skipUnless(is_cv2_available(), "OpenCV descriptor baseline tests require cv2")
class Task3DescriptorBaselineTests(unittest.TestCase):
    def test_orb_descriptor_flow_can_produce_verified_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            reference_bytes = _pattern_image(64, 64)
            frame_bytes = _pattern_image(160, 160, embed_at=(32, 40))
            (temp_path / "ref-001.pgm").write_bytes(reference_bytes)

            settings = MvpRuntimeSettings(
                task3_reference_dir=temp_path,
                task3_orb_features=512,
                task3_match_min_inliers=2,
                task3_match_ratio_threshold=0.9,
            )
            cache = ReferenceCache()
            cache.preload_from_directory(temp_path, orb_features=settings.task3_orb_features)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
            frame = FrameEnvelope(
                frame_url="http://mock/frames/1/",
                image_url="/frame.pgm",
                video_name="session",
                translation_x=0.0,
                translation_y=0.0,
                translation_z=0.0,
                health_status="1",
            )
            decoded = decode_image_bytes(frame, frame_bytes)
            matches = matcher.match(frame, frame_bytes, decoded_frame=decoded)
            filtered = filter_no_match_candidates(matches, min_score=0.70, ambiguity_margin=0.05)
            verified = verify_matches(frame, filtered, decoded_frame=decoded, min_inliers=2)

            self.assertGreaterEqual(len(matches), 1)
            self.assertNotIn("placeholder", matches[0].metadata.get("matcher_source", ""))
            self.assertLessEqual(len(verified), 1)
            if verified:
                self.assertEqual(verified[0].metadata["verification_status"], "verified")


if __name__ == "__main__":
    unittest.main()
