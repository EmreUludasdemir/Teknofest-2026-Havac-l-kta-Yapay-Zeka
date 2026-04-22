from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.task3.routing_policy import (
    EXTREME_ASPECT_RATIO_THRESHOLD,
    LARGE_REFERENCE_LONG_SIDE_PX,
    RoutingAssignment,
    assign_detector,
)


class Task3RoutingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_dir = Path("data/references/2026_baseline")

    def test_official_references_match_calibrated_policy(self) -> None:
        expected = {
            "ref_01.jpg": ("orb", ["rgb"]),
            "ref_02.jpg": ("orb", ["rgb"]),
            "ref_03.jpg": ("orb", ["rgb"]),
            "ref_04.jpg": ("yoloe", ["thermal"]),
            "ref_05.jpg": ("yoloe", ["rgb"]),
            "ref_06.jpg": ("yoloe", ["rgb"]),
        }
        for filename, (detector, modalities) in expected.items():
            assignment = assign_detector(self.reference_dir / filename)
            self.assertEqual(assignment.detector, detector, filename)
            self.assertEqual(assignment.detector_modalities, modalities, filename)

    def test_threshold_constants_are_pinned(self) -> None:
        self.assertEqual(LARGE_REFERENCE_LONG_SIDE_PX, 1800)
        self.assertAlmostEqual(EXTREME_ASPECT_RATIO_THRESHOLD, 3.0, places=6)

    def test_large_rgb_image_prefers_orb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "large_rgb.png"
            height = 2000
            width = 3000
            x = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
            y = np.tile(np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1), (1, width))
            image = np.stack(
                [x, y, ((x.astype(np.uint16) + (2 * y.astype(np.uint16))) // 3).astype(np.uint8)],
                axis=-1,
            )
            Image.fromarray(image, mode="RGB").save(image_path)

            assignment = assign_detector(image_path)

        self.assertEqual(assignment.detector, "orb")
        self.assertEqual(assignment.detector_modalities, ["rgb"])
        self.assertTrue(assignment.signals["visibility"]["is_large"])

    def test_small_centered_rgb_image_prefers_yoloe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "centered_rgb.png"
            height = 400
            width = 400
            yy, xx = np.mgrid[0:height, 0:width]
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[..., 0] = 30 + (xx % 40)
            image[..., 1] = 20 + (yy % 30)
            image[..., 2] = 15 + ((xx + yy) % 20)
            mask = ((xx - 200) ** 2 / (100**2) + (yy - 200) ** 2 / (80**2)) <= 1.0
            image[mask, 0] = (120 + ((xx[mask] * 3 + yy[mask] * 5) % 120)).astype(np.uint8)
            image[mask, 1] = (50 + ((xx[mask] * 7 + yy[mask] * 2) % 180)).astype(np.uint8)
            image[mask, 2] = (20 + ((xx[mask] * 11 + yy[mask] * 13) % 200)).astype(np.uint8)
            Image.fromarray(image, mode="RGB").save(image_path)

            assignment = assign_detector(image_path)

        self.assertEqual(assignment.detector, "yoloe")
        self.assertEqual(assignment.detector_modalities, ["rgb"])
        self.assertFalse(assignment.signals["visibility"]["is_large"])
        self.assertFalse(assignment.signals["visibility"]["is_scene_like"])

    def test_extreme_aspect_rgb_image_prefers_orb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "wide_rgb.png"
            height = 500
            width = 1500
            x = np.tile(np.linspace(20, 240, width, dtype=np.uint8), (height, 1))
            y = np.tile(np.linspace(0, 255, height, dtype=np.uint8).reshape(height, 1), (1, width))
            image = np.stack(
                [x, ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8), y],
                axis=-1,
            )
            Image.fromarray(image, mode="RGB").save(image_path)

            assignment = assign_detector(image_path)

        self.assertEqual(assignment.detector, "orb")
        self.assertTrue(assignment.signals["visibility"]["is_extreme_aspect"])

    def test_bimodal_grayscale_image_routes_to_thermal_yoloe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "thermal_like.png"
            image = np.zeros((400, 400, 3), dtype=np.uint8)
            image[:, :200, :] = 30
            image[:, 200:, :] = 220
            Image.fromarray(image, mode="RGB").save(image_path)

            assignment = assign_detector(image_path)

        self.assertEqual(assignment.detector, "yoloe")
        self.assertEqual(assignment.detector_modalities, ["thermal"])
        self.assertEqual(assignment.confidence, "high")

    def test_ambiguous_image_falls_back_to_both(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "ambiguous.png"
            gradient = np.tile(np.linspace(100, 150, 128, dtype=np.uint8), (128, 1))
            image = np.stack(
                [gradient, np.clip(gradient + 8, 0, 255), np.clip(gradient + 12, 0, 255)],
                axis=-1,
            )
            Image.fromarray(image, mode="RGB").save(image_path)

            assignment = assign_detector(image_path)

        self.assertEqual(assignment.detector, "both")
        self.assertEqual(assignment.detector_modalities, ["rgb", "thermal"])
        self.assertEqual(assignment.confidence, "low")


if __name__ == "__main__":
    unittest.main()
