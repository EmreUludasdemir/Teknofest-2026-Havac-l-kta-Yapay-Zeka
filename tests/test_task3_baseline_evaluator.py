from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame
from src.core.vision import decode_image_bytes, is_cv2_available
from src.evaluation.task3_baseline import evaluate_task3_frames


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


@unittest.skipUnless(is_cv2_available(), "Task3 baseline evaluator tests require cv2")
class Task3BaselineEvaluatorTests(unittest.TestCase):
    def _decoded(self, image_bytes: bytes, frame_index: int) -> DecodedFrame:
        from src.core.frame_state import FrameEnvelope

        frame = FrameEnvelope(
            frame_url=f"http://mock/frames/{frame_index}/",
            image_url="/frame.pgm",
            video_name="session",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
        )
        return decode_image_bytes(frame, image_bytes)

    def test_evaluator_reports_no_match_and_verification_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.pgm").write_bytes(_pattern_image(64, 64))
            decoded_frames = [
                self._decoded(_pattern_image(160, 160, embed_at=(32, 40)), 1),
                self._decoded(_pattern_image(160, 160), 2),
            ]
            summary = evaluate_task3_frames(
                decoded_frames,
                runtime_settings=MvpRuntimeSettings(
                    task3_eval_reference_dir=temp_path,
                    task3_reference_dir=temp_path,
                    task3_match_min_inliers=2,
                    task3_match_ratio_threshold=0.9,
                    task3_orb_features=512,
                ),
                reference_dir=temp_path,
                video_name="descriptor-eval",
            )
            self.assertEqual(summary["status"], "ok")
            self.assertIn("false_positive_proxy_count", summary)
            self.assertIn("no_match_suppression_rate", summary)
            self.assertIn(summary["decision"], {"gerekli", "henuz_gereksiz", "belirsiz"})


if __name__ == "__main__":
    unittest.main()
