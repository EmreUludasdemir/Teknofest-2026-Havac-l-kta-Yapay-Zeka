from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings, SequentialProtocolSettings
from src.core.frame_state import FrameEnvelope
from src.core.vision import decode_image_bytes, is_cv2_available
from src.task3.pipeline import Task3Pipeline


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
            tx = x + offset_x
            ty = y + offset_y
            if 0 <= tx < width and 0 <= ty < height:
                values[ty * width + tx] = pattern[y * ref_width + x]
    return _pgm_bytes(width, height, bytes(values))


@unittest.skipUnless(is_cv2_available(), "Task3 pipeline tests require cv2")
class Task3PipelineFacadeTests(unittest.TestCase):
    def _frame(self, frame_id: int) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url=f"http://mock/frames/{frame_id}/",
            image_url=f"/frame-{frame_id}.pgm",
            video_name="task3-session",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
        )

    def test_default_settings_route_to_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.pgm").write_bytes(_pattern_image(64, 64))
            settings = MvpRuntimeSettings(task3_reference_dir=temp_path)
            pipeline = Task3Pipeline(runtime_settings=settings)
            frame = self._frame(1)
            decoded = decode_image_bytes(frame, _pattern_image(160, 160, embed_at=(24, 30)))

            matches, diagnostics = pipeline.process(frame, b"", decoded_frame=decoded)

            self.assertEqual(diagnostics["active_mode"], "baseline")
            self.assertFalse(diagnostics["experimental_enabled"])
            self.assertIn("integration_status", diagnostics)
            self.assertLessEqual(len(matches), 1)

    def test_missing_yoloe_weights_degrades_to_prompt_approx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.pgm").write_bytes(_pattern_image(64, 64))
            settings = MvpRuntimeSettings(
                task3_reference_dir=temp_path,
                task3_experimental_enabled=True,
                task3_experimental_mode="yoloe_prompted",
                task3_experimental_tracking="light",
                task3_experimental_verifier="orb_homography",
                task3_experimental_model_path=temp_path / "missing-yoloe.pt",
            )
            pipeline = Task3Pipeline(runtime_settings=settings)
            frame = self._frame(2)
            decoded = decode_image_bytes(frame, _pattern_image(160, 160, embed_at=(40, 36)))

            matches, diagnostics = pipeline.process(frame, b"", decoded_frame=decoded)

            self.assertEqual(diagnostics["active_mode"], "experimental")
            self.assertEqual(diagnostics["detector"]["active_detector_mode"], "prompt_approx")
            self.assertIn(diagnostics["integration_status"], {"yoloe_weight_missing", "prompt_approx"})
            self.assertLessEqual(len(matches), 1)

    def test_default_wire_profile_stays_official_current(self) -> None:
        self.assertEqual(SequentialProtocolSettings(base_url="http://mock/", username="a", password="b").wire_profile, "official_current")


if __name__ == "__main__":
    unittest.main()
