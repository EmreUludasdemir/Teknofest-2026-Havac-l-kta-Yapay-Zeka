from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from src.task3.modality_detection import Modality, detect_modality


class Task3ModalityDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_dir = Path("data/references/2026_baseline")

    def test_official_rgb_references_are_detected_from_exif(self) -> None:
        for reference_name in ("ref_01.jpg", "ref_02.jpg", "ref_03.jpg"):
            modality, diagnostics = detect_modality(self.reference_dir / reference_name)
            self.assertEqual(modality, Modality.RGB, reference_name)
            self.assertEqual(diagnostics["method"], "exif", reference_name)
            self.assertEqual(diagnostics["confidence"], "high", reference_name)
            self.assertIn("default", diagnostics["exif_signals"], reference_name)

    def test_official_thermal_reference_is_detected_from_exif(self) -> None:
        modality, diagnostics = detect_modality(self.reference_dir / "ref_04.jpg")
        self.assertEqual(modality, Modality.THERMAL)
        self.assertEqual(diagnostics["method"], "exif")
        self.assertEqual(diagnostics["confidence"], "high")
        self.assertIn("whitehot", diagnostics["exif_signals"])

    def test_exifless_official_references_fall_back_to_pixel_analysis(self) -> None:
        for reference_name in ("ref_05.jpg", "ref_06.jpg"):
            modality, diagnostics = detect_modality(self.reference_dir / reference_name)
            self.assertEqual(diagnostics["method"], "pixel", reference_name)
            self.assertIn(modality, {Modality.RGB, Modality.UNKNOWN}, reference_name)
            self.assertEqual(diagnostics["exif_signals"], [], reference_name)
            self.assertIn("mean_saturation", diagnostics["pixel_signals"], reference_name)

    def test_pure_grayscale_image_is_classified_as_thermal_via_pixel_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "grayscale.png"
            gradient = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
            rgb = np.stack([gradient, gradient, gradient], axis=-1)
            Image.fromarray(rgb, mode="RGB").save(image_path)

            modality, diagnostics = detect_modality(image_path)

        self.assertEqual(modality, Modality.THERMAL)
        self.assertEqual(diagnostics["method"], "pixel")
        self.assertIn(diagnostics["confidence"], {"medium", "high"})
        self.assertLess(diagnostics["pixel_signals"]["mean_saturation"], 1.0)
        self.assertGreaterEqual(diagnostics["pixel_signals"]["min_channel_correlation"], 0.99)

    def test_colorful_rgb_image_is_classified_as_rgb_via_pixel_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "colorful.png"
            height = 128
            width = 128
            x = np.linspace(0, 255, width, dtype=np.uint8)
            y = np.linspace(255, 0, height, dtype=np.uint8)
            red = np.tile(x, (height, 1))
            green = np.tile(y.reshape(height, 1), (1, width))
            blue = ((red.astype(np.uint16) + green.astype(np.uint16)) // 2).astype(np.uint8)
            rgb = np.stack([red, green, blue], axis=-1)
            Image.fromarray(rgb, mode="RGB").save(image_path)

            modality, diagnostics = detect_modality(image_path)

        self.assertEqual(modality, Modality.RGB)
        self.assertEqual(diagnostics["method"], "pixel")
        self.assertIn(diagnostics["confidence"], {"medium", "high"})
        self.assertGreater(diagnostics["pixel_signals"]["mean_saturation"], 20.0)

    def test_bimodal_grayscale_image_sets_bimodal_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "bimodal.png"
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            image[:, :64, :] = 30
            image[:, 64:, :] = 220
            Image.fromarray(image, mode="RGB").save(image_path)

            modality, diagnostics = detect_modality(image_path)

        self.assertEqual(modality, Modality.THERMAL)
        self.assertEqual(diagnostics["method"], "pixel")
        self.assertTrue(diagnostics["pixel_signals"]["histogram_bimodal"])

    def test_corrupted_exif_falls_back_to_pixel_analysis_without_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "corrupted_exif.jpg"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[..., 0] = 220
            image[..., 1] = 30
            image[..., 2] = 80
            Image.fromarray(image, mode="RGB").save(image_path, format="JPEG")

            with mock.patch("PIL.Image.Image.getexif", side_effect=OSError("broken exif")):
                modality, diagnostics = detect_modality(image_path)

        self.assertEqual(modality, Modality.RGB)
        self.assertEqual(diagnostics["method"], "pixel")

    def test_nonexistent_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            detect_modality(Path("data/references/2026_baseline/does_not_exist.jpg"))

    def test_zero_byte_file_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "empty.jpg"
            image_path.write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "empty"):
                detect_modality(image_path)

    def test_tiny_image_is_handled_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "tiny.png"
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            image[..., 0] = 255
            image[..., 1] = 64
            Image.fromarray(image, mode="RGB").save(image_path)

            modality, diagnostics = detect_modality(image_path)

        self.assertIn(modality, {Modality.RGB, Modality.UNKNOWN, Modality.THERMAL})
        self.assertIn("reason", diagnostics)
        self.assertIn("pixel_signals", diagnostics)


if __name__ == "__main__":
    unittest.main()
