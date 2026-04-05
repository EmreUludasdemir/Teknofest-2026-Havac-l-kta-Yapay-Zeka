from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.task2.calibration import build_calibration_state, load_calibration_bundle, select_calibration_profile


class Task2CalibrationTests(unittest.TestCase):
    def test_parse_calibration_file_and_build_state(self) -> None:
        content = """Termal Camera Intrinsics
                     FocalLength: [731.7965 732.0172]
                  PrincipalPoint: [319.2367 251.2424]
                       ImageSize: [512 640]

RGB Camera Intrinsics
                     FocalLength: [2792.2 2795.2]
                  PrincipalPoint: [1988.0 1562.2]
                       ImageSize: [3000 4000]
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration_path = Path(tmp_dir) / "calibration.txt"
            calibration_path.write_text(content, encoding="utf-8")
            bundle = load_calibration_bundle(calibration_path)
            frame = FrameEnvelope(
                frame_url="http://mock/frames/1/",
                image_url="/frame.jpg",
                video_name="thermal_session",
                translation_x=0.0,
                translation_y=0.0,
                translation_z=0.0,
                health_status="1",
            )
            decoded = DecodedFrame(None, None, 640, 512, 0, "thermal", 1)
            profile = select_calibration_profile(bundle, frame, decoded)
            state = build_calibration_state([frame])
            self.assertEqual(profile.modality, "thermal")
            self.assertEqual(bundle.rgb.image_width, 4000)
            self.assertIn("frame_count", state)


if __name__ == "__main__":
    unittest.main()
