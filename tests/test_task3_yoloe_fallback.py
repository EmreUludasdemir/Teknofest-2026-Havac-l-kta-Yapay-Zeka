from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope
from src.core.vision import decode_image_bytes, is_cv2_available
from src.pipeline.mvp_processor import MvpFrameProcessor
from src.task3.experimental.backend import Task3ExperimentalUnavailableError, YoloeVpLightGlueBackend
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
            target_x = x + offset_x
            target_y = y + offset_y
            if 0 <= target_x < width and 0 <= target_y < height:
                values[target_y * width + target_x] = pattern[y * ref_width + x]
    return _pgm_bytes(width, height, bytes(values))


class _ExplodingExperimentalBackend:
    def match(self, *, decoded_frame, reference_ids, scenario_id=None):  # type: ignore[override]
        del decoded_frame, reference_ids, scenario_id
        raise Task3ExperimentalUnavailableError("backend_exception")


class _CudaUnavailableExperimentalBackend:
    def match(self, *, decoded_frame, reference_ids, scenario_id=None):  # type: ignore[override]
        del decoded_frame, reference_ids, scenario_id
        raise Task3ExperimentalUnavailableError("cuda_required_but_unavailable")


class _FakeTorchNoCuda:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


@unittest.skipUnless(is_cv2_available(), "Task3 YOLOE fallback tests require cv2")
class Task3YoloeFallbackTests(unittest.TestCase):
    @staticmethod
    def _mock_find_spec_missing_lightglue(name: str):
        from importlib.machinery import ModuleSpec

        if name == "lightglue":
            return None
        return ModuleSpec(name, loader=None)

    def _frame(self) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url="http://mock/frames/1/",
            image_url="/frame.pgm",
            video_name="session",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
        )

    def test_missing_yoloe_weight_falls_back_to_orb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            reference_bytes = _pattern_image(64, 64)
            frame_bytes = _pattern_image(160, 160, embed_at=(32, 40))
            (temp_path / "ref-001.pgm").write_bytes(reference_bytes)

            settings = MvpRuntimeSettings(
                task3_reference_dir=temp_path,
                task3_mode="yoloe_vp_lightglue",
                task3_orb_features=512,
                task3_match_min_inliers=2,
                task3_match_ratio_threshold=0.9,
                task3_yoloe_weight_path=temp_path / "missing.pt",
            )
            cache = ReferenceCache()
            cache.preload_from_directory(temp_path, orb_features=settings.task3_orb_features)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
            decoded = decode_image_bytes(self._frame(), frame_bytes)

            matches = matcher.match(self._frame(), frame_bytes, decoded_frame=decoded, mode="yoloe_vp_lightglue")

            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(matcher.last_run_info["requested_mode"], "yoloe_vp_lightglue")
            self.assertEqual(matcher.last_run_info["effective_mode"], "orb_template")
            self.assertEqual(matcher.last_run_info["fallback_reason"], "missing_yoloe_weight")
            self.assertGreaterEqual(int(matcher.last_run_info["candidates_generated"]), 1)

    def test_backend_exception_still_returns_orb_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            reference_bytes = _pattern_image(64, 64)
            frame_bytes = _pattern_image(160, 160, embed_at=(32, 40))
            (temp_path / "ref-001.pgm").write_bytes(reference_bytes)

            settings = MvpRuntimeSettings(
                task3_reference_dir=temp_path,
                task3_mode="yoloe_vp_lightglue",
                task3_orb_features=512,
                task3_match_min_inliers=2,
                task3_match_ratio_threshold=0.9,
            )
            cache = ReferenceCache()
            cache.preload_from_directory(temp_path, orb_features=settings.task3_orb_features)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
            matcher.experimental_backend = _ExplodingExperimentalBackend()
            decoded = decode_image_bytes(self._frame(), frame_bytes)

            matches = matcher.match(self._frame(), frame_bytes, decoded_frame=decoded, mode="yoloe_vp_lightglue")

            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(matcher.last_run_info["effective_mode"], "orb_template")
            self.assertEqual(matcher.last_run_info["fallback_reason"], "backend_exception")

    def test_cuda_required_but_unavailable_falls_back_to_orb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            reference_bytes = _pattern_image(64, 64)
            frame_bytes = _pattern_image(160, 160, embed_at=(32, 40))
            (temp_path / "ref-001.pgm").write_bytes(reference_bytes)

            settings = MvpRuntimeSettings(
                task3_reference_dir=temp_path,
                task3_mode="yoloe_vp_lightglue",
                task3_orb_features=512,
                task3_match_min_inliers=2,
                task3_match_ratio_threshold=0.9,
                task3_yoloe_allow_cpu=False,
            )
            cache = ReferenceCache()
            cache.preload_from_directory(temp_path, orb_features=settings.task3_orb_features)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
            matcher.experimental_backend = _CudaUnavailableExperimentalBackend()
            decoded = decode_image_bytes(self._frame(), frame_bytes)

            matches = matcher.match(self._frame(), frame_bytes, decoded_frame=decoded, mode="yoloe_vp_lightglue")

            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(matcher.last_run_info["effective_mode"], "orb_template")
            self.assertEqual(matcher.last_run_info["fallback_reason"], "cuda_required_but_unavailable")

    def test_processor_surfaces_task3_info_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            reference_bytes = _pattern_image(64, 64)
            frame_bytes = _pattern_image(160, 160, embed_at=(32, 40))
            (temp_path / "ref-001.pgm").write_bytes(reference_bytes)
            settings = MvpRuntimeSettings(
                task3_reference_dir=temp_path,
                task3_mode="yoloe_vp_lightglue",
                task3_orb_features=512,
                task3_match_min_inliers=2,
                task3_match_ratio_threshold=0.9,
                task3_yoloe_weight_path=temp_path / "missing.pt",
            )
            processor = MvpFrameProcessor(runtime_settings=settings)

            result = processor(self._frame(), frame_bytes)

            self.assertEqual(result.diagnostics["task3_status"], "ok")
            self.assertEqual(result.diagnostics["task3_info"]["requested_mode"], "yoloe_vp_lightglue")
            self.assertEqual(result.diagnostics["task3_info"]["effective_mode"], "orb_template")
            self.assertEqual(result.diagnostics["task3_info"]["fallback_reason"], "missing_yoloe_weight")

    def test_backend_reports_missing_lightglue_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.pgm").write_bytes(_pattern_image(64, 64))
            fake_weight = temp_path / "fake.pt"
            fake_weight.write_bytes(b"weight")
            settings = MvpRuntimeSettings(
                task3_reference_dir=temp_path,
                task3_yoloe_weight_path=fake_weight,
            )
            cache = ReferenceCache()
            cache.preload_from_directory(temp_path, orb_features=settings.task3_orb_features)
            decoded = decode_image_bytes(self._frame(), _pattern_image(160, 160, embed_at=(32, 40)))
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)

            with patch("src.task3.experimental.backend.importlib.util.find_spec", side_effect=self._mock_find_spec_missing_lightglue):
                matches = matcher.match(self._frame(), b"", cache.list_ids(), decoded_frame=decoded, mode="yoloe_vp_lightglue")

            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(matcher.last_run_info["effective_mode"], "orb_template")
            self.assertEqual(matcher.last_run_info["fallback_reason"], "missing_lightglue")

    def test_backend_resolve_device_requires_cuda_when_cpu_disabled(self) -> None:
        backend = YoloeVpLightGlueBackend(
            reference_cache=ReferenceCache(),
            runtime_settings=MvpRuntimeSettings(task3_yoloe_allow_cpu=False),
        )
        with self.assertRaises(Task3ExperimentalUnavailableError) as ctx:
            backend._resolve_device(_FakeTorchNoCuda())
        self.assertEqual(ctx.exception.reason, "cuda_required_but_unavailable")


if __name__ == "__main__":
    unittest.main()
