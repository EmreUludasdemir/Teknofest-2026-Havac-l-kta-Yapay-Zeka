from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope
from src.core.vision import decode_image_bytes, is_cv2_available
from src.task3.experimental.pipeline import ExperimentalTask3Pipeline
from src.task3.matcher import Task3Matcher
from src.task3.reference_cache import ReferenceCache


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


@unittest.skipUnless(is_cv2_available(), "Experimental Task3 tests require cv2")
class ExperimentalTask3PipelineTests(unittest.TestCase):
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

    def _build_pipeline(self, temp_path: Path) -> ExperimentalTask3Pipeline:
        settings = MvpRuntimeSettings(
            task3_reference_dir=temp_path,
            task3_experimental_enabled=True,
            task3_experimental_mode="yoloe_prompted",
            task3_experimental_tracking="light",
            task3_experimental_verifier="orb_homography",
            task3_experimental_model_path=temp_path / "missing-yoloe.pt",
        )
        cache = ReferenceCache()
        cache.preload_from_directory(temp_path, orb_features=settings.task3_orb_features)
        matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
        return ExperimentalTask3Pipeline(
            reference_cache=cache,
            runtime_settings=settings,
            baseline_matcher=matcher,
        )

    def test_force_track_loss_triggers_bounded_redetect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.pgm").write_bytes(_pattern_image(64, 64))
            pipeline = self._build_pipeline(temp_path)
            frame = self._frame(1)
            decoded = decode_image_bytes(frame, _pattern_image(160, 160, embed_at=(32, 40)))

            initial_matches, _ = pipeline.process(frame, b"", decoded_frame=decoded)
            self.assertLessEqual(len(initial_matches), 1)
            original_candidates, original_diag = pipeline.detector.search(frame, decoded, ["ref-001"])

            calls: list[list[str]] = []

            def fake_search(*args):
                reference_ids = args[-1]
                calls.append(list(reference_ids))
                if len(calls) == 1:
                    return [], {"integration_status": "prompt_approx", "active_detector_mode": "prompt_approx"}
                return original_candidates, original_diag

            pipeline.force_track_loss()
            with patch.object(type(pipeline.detector), "search", side_effect=fake_search):
                matches, diagnostics = pipeline.process(frame, b"", decoded_frame=decoded)

            self.assertTrue(diagnostics["redetect_triggered"])
            self.assertEqual(len(calls), 2)
            self.assertLessEqual(len(matches), 1)

    def test_verifier_rejection_suppresses_emission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.pgm").write_bytes(_pattern_image(64, 64))
            pipeline = self._build_pipeline(temp_path)
            frame = self._frame(2)
            decoded = decode_image_bytes(frame, _pattern_image(160, 160, embed_at=(40, 42)))

            with patch.object(
                type(pipeline.verifier),
                "verify",
                return_value=([], {"requested_verifier_mode": "orb_homography", "active_verifier_mode": "orb_homography"}),
            ):
                matches, diagnostics = pipeline.process(frame, b"", decoded_frame=decoded)

            self.assertEqual(matches, [])
            self.assertEqual(diagnostics["accepted_candidate_count"], 0)

    def test_unsupported_modes_degrade_predictably(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.pgm").write_bytes(_pattern_image(64, 64))
            settings = MvpRuntimeSettings(
                task3_reference_dir=temp_path,
                task3_experimental_enabled=True,
                task3_experimental_mode="prompt_approx",
                task3_experimental_tracking="samurai_like",
                task3_experimental_verifier="lightglue",
                task3_experimental_tiled_inference=True,
            )
            cache = ReferenceCache()
            cache.preload_from_directory(temp_path, orb_features=settings.task3_orb_features)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
            pipeline = ExperimentalTask3Pipeline(
                reference_cache=cache,
                runtime_settings=settings,
                baseline_matcher=matcher,
            )
            frame = self._frame(3)
            decoded = decode_image_bytes(frame, _pattern_image(160, 160, embed_at=(28, 26)))

            _, diagnostics = pipeline.process(frame, b"", decoded_frame=decoded)

            self.assertEqual(diagnostics["tracker_active_mode"], "light")
            self.assertEqual(diagnostics["verifier_active_mode"], "orb_homography")
            self.assertEqual(diagnostics["detector"]["tiled_status"]["reason"], "missing_dependency:sahi")


if __name__ == "__main__":
    unittest.main()
