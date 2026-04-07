from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame
from src.core.vision import is_cv2_available
from src.evaluation.task2_thermal_microlever import (
    apply_eval_transform,
    build_variant_settings,
    evaluate_task2_thermal_microlever,
    load_task2_thermal_microlever_manifest,
)

if is_cv2_available():  # pragma: no branch
    from src.core.vision import np
else:  # pragma: no cover
    np = None


class Task2ThermalMicroleverEvalTests(unittest.TestCase):
    def test_build_variant_settings_keeps_baseline_and_sets_candidates(self) -> None:
        base = MvpRuntimeSettings()
        baseline = build_variant_settings(base, "baseline")
        confidence_candidate = build_variant_settings(base, "thermal_confidence_floor_045")
        sensor_hint_candidate = build_variant_settings(base, "thermal_sensor_hint_weight_008")
        self.assertEqual(baseline.task2_confidence_floor_thermal, 0.40)
        self.assertEqual(confidence_candidate.task2_confidence_floor_thermal, 0.45)
        self.assertEqual(sensor_hint_candidate.task2_sensor_hint_max_weight_thermal, 0.08)

    @unittest.skipUnless(is_cv2_available(), "OpenCV required")
    def test_apply_eval_transform_weakens_sensor_hint_without_changing_shape(self) -> None:
        decoded = DecodedFrame(None, np.full((16, 16), 120, dtype=np.uint8), 16, 16, 1, "thermal", 1)
        transformed, hint, info = apply_eval_transform(
            decoded,
            transform_profile="weak_sensor_hint",
            frame_offset=3,
            health_status="0",
            sensor_hint={"translation_x": 1.0, "translation_y": 2.0, "translation_z": 3.0},
        )
        self.assertEqual(transformed.gray.shape, decoded.gray.shape)
        self.assertTrue(info["sensor_hint_weakened"])
        self.assertNotEqual(hint["translation_x"], 1.0)

    def test_manifest_loader_and_report_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp = Path(tmp_dir)
            manifest = {
                "scenarios": [
                    {
                        "scenario_id": "thermal_long_degraded_1",
                        "csv_path": "mini.csv",
                        "video_path": "mini.mp4",
                        "frame_start": 0,
                        "frame_limit": 3,
                        "frame_stride": 1,
                        "health_segments": [{"health": "1", "length": 1}, {"health": "0", "length": 1}, {"health": "1", "length": 1}],
                        "transform_profile": "none",
                        "tags": ["thermal", "mini"],
                    },
                    {
                        "scenario_id": "rgb_long_degraded_control_1",
                        "csv_path": "mini.csv",
                        "video_path": "mini.mp4",
                        "frame_start": 0,
                        "frame_limit": 3,
                        "frame_stride": 1,
                        "health_segments": [{"health": "1", "length": 1}, {"health": "0", "length": 1}, {"health": "1", "length": 1}],
                        "transform_profile": "none",
                        "tags": ["rgb", "control", "mini"],
                    },
                ]
            }
            manifest_path = temp / "manifest.json"
            csv_path = temp / "mini.csv"
            video_path = temp / "mini.mp4"
            for item in manifest["scenarios"]:
                item["csv_path"] = str(csv_path)
                item["video_path"] = str(video_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            csv_path.write_text(
                "translation_x,translation_y,translation_z,frame_numbers\n0,0,1,f0\n0.1,0.0,1,f1\n0.2,0.0,1,f2\n",
                encoding="utf-8",
            )
            video_path.write_bytes(b"video")
            fake_frames = [DecodedFrame(None, None, 640, 512, 0, "thermal", i) for i in range(3)]
            self.assertEqual(load_task2_thermal_microlever_manifest(manifest_path)[0].scenario_id, "thermal_long_degraded_1")
            with patch(
                "src.evaluation.task2_thermal_microlever.iter_scenario_frames",
                side_effect=lambda scenario: iter(fake_frames),
            ):
                payload = evaluate_task2_thermal_microlever(
                    manifest_path=manifest_path,
                    output_dir=temp,
                    variant="baseline",
                )
            self.assertIn("final_recommendation", payload)
            self.assertTrue((temp / "task2_thermal_microlever_results.json").exists())
            self.assertTrue((temp / "task2_thermal_microlever_results.md").exists())
            self.assertTrue((temp / "task2_thermal_microlever_design.md").exists())
            self.assertTrue((temp / "task2_thermal_microlever_recommendation.md").exists())


if __name__ == "__main__":
    unittest.main()
