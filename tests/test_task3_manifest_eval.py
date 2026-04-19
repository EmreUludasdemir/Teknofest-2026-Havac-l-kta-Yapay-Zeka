from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.core.vision import is_cv2_available
from src.evaluation.task3_manifest_eval import evaluate_task3_manifest, load_task3_manifest


def _pattern_frame(width: int, height: int, *, embed: bool) -> tuple[bytes, bytes, bytes]:
    values = bytearray(width * height)
    ref_w = 64
    ref_h = 64
    ref = bytearray(ref_w * ref_h)
    for y in range(ref_h):
        for x in range(ref_w):
            if x == y or x + y == ref_w - 1 or (8 <= x < ref_w - 8 and 8 <= y < ref_h - 8 and (x // 4 + y // 4) % 2 == 0):
                ref[y * ref_w + x] = 255
    if embed:
        offset_x, offset_y = 32, 40
        for y in range(ref_h):
            for x in range(ref_w):
                target_x = x + offset_x
                target_y = y + offset_y
                values[target_y * width + target_x] = ref[y * ref_w + x]
    return bytes(ref), bytes(values), bytes(bytearray(width * height))


@unittest.skipUnless(is_cv2_available(), "Task3 manifest evaluator tests require cv2")
class Task3ManifestEvalTests(unittest.TestCase):
    def test_manifest_loader_and_evaluator_support_present_and_absent(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            refs_dir = temp_path / "refs"
            refs_dir.mkdir()
            ref_bytes, present_pixels, blank_pixels = _pattern_frame(160, 160, embed=True)
            (refs_dir / "ref-001.pgm").write_bytes(b"P5\n64 64\n255\n" + ref_bytes)

            present_video = temp_path / "present.avi"
            absent_video = temp_path / "absent.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            for video_path, pixels in ((present_video, present_pixels), (absent_video, blank_pixels)):
                writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (160, 160), isColor=True)
                self.assertTrue(writer.isOpened())
                frame = np.frombuffer(pixels, dtype=np.uint8).reshape(160, 160)
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                for _ in range(3):
                    writer.write(bgr)
                writer.release()

            manifest_path = temp_path / "task3_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "scenarios": [
                            {
                                "id": "present_targets_case",
                                "video": str(present_video),
                                "references_dir": str(refs_dir),
                                "reference_mode": "present_targets",
                                "frame_stride": 1,
                                "frame_limit": 2,
                                "expected_present_refs": ["ref-001"],
                                "tags": ["present"],
                            },
                            {
                                "id": "absent_targets_case",
                                "video": str(absent_video),
                                "references_dir": str(refs_dir),
                                "reference_mode": "synthetic_absent",
                                "frame_stride": 1,
                                "frame_limit": 2,
                                "expected_present_refs": [],
                                "tags": ["absent"],
                            },
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            loaded = load_task3_manifest(manifest_path)
            self.assertEqual([item["id"] for item in loaded["scenarios"]], ["present_targets_case", "absent_targets_case"])

            reports_dir = temp_path / "reports"
            comparison = evaluate_task3_manifest(
                runtime_settings=MvpRuntimeSettings(
                    task3_reference_dir=refs_dir,
                    task3_eval_reference_dir=refs_dir,
                    task3_orb_features=512,
                    task3_match_min_inliers=2,
                    task3_match_ratio_threshold=0.9,
                    task3_yoloe_weight_path=temp_path / "missing.pt",
                ),
                manifest_path=manifest_path,
                output_dir=reports_dir,
            )

            self.assertIn("orb_template", comparison["modes"])
            self.assertIn("yoloe_vp_lightglue", comparison["modes"])
            orb_results = comparison["modes"]["orb_template"]["results"]
            yoloe_results = comparison["modes"]["yoloe_vp_lightglue"]["results"]
            absent = next(item for item in orb_results if item["scenario_id"] == "absent_targets_case")
            yoloe_present = next(item for item in yoloe_results if item["scenario_id"] == "present_targets_case")
            self.assertEqual(absent["reference_mode"], "synthetic_absent")
            self.assertEqual(absent["false_positive_proxy_count"], absent["accepted_match_count"])
            self.assertEqual(yoloe_present["fallback_reason"], "missing_yoloe_weight")
            self.assertEqual(yoloe_present["effective_mode_counts"], {"orb_template": 2})
            markdown = (reports_dir / "task3_manifest_comparison.md").read_text(encoding="utf-8")
            self.assertIn("missing_yoloe_weight", markdown)
            self.assertIn("orb_template", markdown)
            self.assertTrue((reports_dir / "task3_manifest_comparison.md").exists())

    def test_debug_dump_is_off_by_default_and_scenario_filter_limits_output(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            refs_dir = temp_path / "refs"
            refs_dir.mkdir()
            ref_bytes, present_pixels, blank_pixels = _pattern_frame(160, 160, embed=True)
            (refs_dir / "ref-001.pgm").write_bytes(b"P5\n64 64\n255\n" + ref_bytes)

            present_video = temp_path / "present.avi"
            absent_video = temp_path / "absent.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            for video_path, pixels in ((present_video, present_pixels), (absent_video, blank_pixels)):
                writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (160, 160), isColor=True)
                self.assertTrue(writer.isOpened())
                frame = np.frombuffer(pixels, dtype=np.uint8).reshape(160, 160)
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                for _ in range(2):
                    writer.write(bgr)
                writer.release()

            manifest_path = temp_path / "task3_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "scenarios": [
                            {
                                "id": "present_targets_case",
                                "video": str(present_video),
                                "references_dir": str(refs_dir),
                                "reference_mode": "present_targets",
                                "frame_stride": 1,
                                "frame_limit": 2,
                            },
                            {
                                "id": "absent_targets_case",
                                "video": str(absent_video),
                                "references_dir": str(refs_dir),
                                "reference_mode": "synthetic_absent",
                                "frame_stride": 1,
                                "frame_limit": 2,
                            },
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            debug_dir = temp_path / "debug_rejects"
            reports_dir = temp_path / "reports"
            comparison = evaluate_task3_manifest(
                runtime_settings=MvpRuntimeSettings(
                    task3_reference_dir=refs_dir,
                    task3_eval_reference_dir=refs_dir,
                    task3_orb_features=512,
                    task3_match_min_inliers=2,
                    task3_match_ratio_threshold=0.9,
                    task3_yoloe_weight_path=temp_path / "missing.pt",
                    task3_debug_dump_dir=debug_dir,
                ),
                manifest_path=manifest_path,
                output_dir=reports_dir,
                scenario_ids=("present_targets_case",),
            )

            self.assertEqual(len(comparison["comparison_rows"]), 1)
            self.assertEqual(comparison["comparison_rows"][0]["scenario_id"], "present_targets_case")
            self.assertFalse(debug_dir.exists())


if __name__ == "__main__":
    unittest.main()
