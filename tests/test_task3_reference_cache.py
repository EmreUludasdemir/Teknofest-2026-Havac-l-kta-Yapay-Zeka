from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame, FrameEnvelope
from src.task3.matcher import Task3Matcher
from src.task3.reference_cache import ReferenceCache


def _candidate(object_id: str, score: float, source: str) -> CanonicalUndefinedObject:
    return CanonicalUndefinedObject(
        object_id=object_id,
        top_left_x=0.0,
        top_left_y=0.0,
        bottom_right_x=10.0,
        bottom_right_y=10.0,
        metadata={"match_score": score, "matcher_source": source},
    )


class Task3ReferenceCacheTests(unittest.TestCase):
    def test_v3_spec_auto_routing_matches_phase_b_reference_table(self) -> None:
        cache = ReferenceCache()
        loaded = cache.preload_from_directory(Path("data/references/2026_baseline"))

        self.assertEqual(loaded, 6)
        self.assertEqual(cache.get_candidate_suppression_mode(), "per_reference_top_1")
        expected = {
            "ref_01": ("rgb", "orb", ["rgb"]),
            "ref_02": ("rgb", "orb", ["rgb"]),
            "ref_03": ("rgb", "orb", ["rgb"]),
            "ref_04": ("thermal", "yoloe", ["thermal"]),
            "ref_05": ("rgb", "yoloe", ["rgb"]),
            "ref_06": ("rgb", "yoloe", ["rgb"]),
        }
        for reference_id, (modality, detector, modalities) in expected.items():
            item = cache.get(reference_id)
            self.assertIsNotNone(item, reference_id)
            self.assertEqual(item["reference_modality"], modality, reference_id)
            self.assertEqual(item["detector"], detector, reference_id)
            self.assertEqual(item["detector_modalities"], modalities, reference_id)

    def test_manual_override_replaces_auto_assignment_and_logs_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            shutil.copyfile("data/references/2026_baseline/ref_01.jpg", temp_path / "ref_01.jpg")
            (temp_path / "manifest.json").write_text(
                json.dumps({"spec_path": "spec.json"}),
                encoding="utf-8",
            )
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "references": {"ref_01": {"file": "ref_01.jpg"}},
                        "overrides": {
                            "ref_01": {
                                "detector": "yoloe",
                                "detector_modalities": ["rgb"],
                                "rationale": "manual override for test",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            cache = ReferenceCache()
            with self.assertLogs("src.task3.reference_cache", level="WARNING") as captured:
                cache.preload_from_directory(temp_path)

        self.assertEqual(cache.get_detector("ref_01"), "yoloe")
        self.assertIn("manual override", "\n".join(captured.output))
        self.assertEqual(len(cache.get_overrides_applied()), 1)

    def test_malformed_override_missing_detector_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            shutil.copyfile("data/references/2026_baseline/ref_01.jpg", temp_path / "ref_01.jpg")
            (temp_path / "manifest.json").write_text(json.dumps({"spec_path": "spec.json"}), encoding="utf-8")
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "references": {"ref_01": {"file": "ref_01.jpg"}},
                        "overrides": {"ref_01": {"rationale": "missing detector"}},
                    }
                ),
                encoding="utf-8",
            )

            cache = ReferenceCache()
            with self.assertRaises(ValueError):
                cache.preload_from_directory(temp_path)

    def test_dual_path_detector_both_routes_reference_through_both_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            gradient = np.tile(np.linspace(100, 150, 64, dtype=np.uint8), (64, 1))
            image[..., 0] = gradient
            image[..., 1] = np.clip(gradient + 8, 0, 255)
            image[..., 2] = np.clip(gradient + 12, 0, 255)
            Image.fromarray(image, mode="RGB").save(temp_path / "ref_ambiguous.png")
            (temp_path / "manifest.json").write_text(json.dumps({"spec_path": "spec.json"}), encoding="utf-8")
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "references": {"ref_ambiguous": {"file": "ref_ambiguous.png"}},
                        "overrides": {
                            "ref_ambiguous": {
                                "detector": "both",
                                "detector_modalities": ["rgb", "thermal"],
                                "rationale": "dual-path test override",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            cache = ReferenceCache()
            cache.preload_from_directory(temp_path)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=MvpRuntimeSettings(task3_mode="yoloe_vp_lightglue"))
            decoded = DecodedFrame(
                bgr=object(),
                gray=object(),
                width=16,
                height=16,
                channel_count=3,
                modality="rgb",
                frame_index=0,
            )
            frame = FrameEnvelope(
                frame_url="http://mock/frames/1/",
                image_url="/frame.jpg",
                video_name="rgb_reference_session",
                translation_x=0.0,
                translation_y=0.0,
                translation_z=0.0,
                health_status="1",
            )

            with patch.object(
                Task3Matcher,
                "_match_with_yoloe_vp_lightglue",
                return_value=[_candidate("ref_ambiguous", 0.51, "task3_yoloe_vp_lightglue")],
            ) as yoloe_mock, patch.object(
                Task3Matcher,
                "_match_with_real_orb_only",
                return_value=[_candidate("ref_ambiguous", 0.83, "task3_orb_bf_homography")],
            ) as orb_mock:
                matches = matcher.match(frame, b"", cache.list_ids(), decoded_frame=decoded, mode="yoloe_vp_lightglue")

        self.assertEqual([item.metadata["matcher_source"] for item in matches], ["task3_orb_bf_homography", "task3_yoloe_vp_lightglue"])
        self.assertEqual(yoloe_mock.call_args.args[-1], ["ref_ambiguous"])
        self.assertEqual(orb_mock.call_args.args[-1], ["ref_ambiguous"])
        self.assertEqual(matcher.last_run_info["yoloe_routed_refs"], ["ref_ambiguous"])
        self.assertEqual(matcher.last_run_info["orb_routed_refs"], ["ref_ambiguous"])
        self.assertIn("ref_ambiguous", matcher.last_run_info["auto_routing"])
        self.assertEqual(len(matcher.last_run_info["overrides_applied"]), 1)

    def test_v2_style_spec_still_loads_while_legacy_manual_fields_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            shutil.copyfile("data/references/2026_baseline/ref_05.jpg", temp_path / "ref_05.jpg")
            (temp_path / "manifest.json").write_text(json.dumps({"spec_path": "spec.json"}), encoding="utf-8")
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "version": "v2",
                        "references": {
                            "ref_05": {
                                "file": "ref_05.jpg",
                                "modality": "thermal",
                                "detector": "orb",
                                "detector_modalities": ["thermal"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            cache = ReferenceCache()
            loaded = cache.preload_from_directory(temp_path)

        self.assertEqual(loaded, 1)
        self.assertEqual(cache.get_detector("ref_05"), "yoloe")
        self.assertEqual(cache.get("ref_05")["reference_modality"], "rgb")

    def test_spec_can_opt_in_to_per_reference_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            shutil.copyfile("data/references/2026_baseline/ref_05.jpg", temp_path / "ref_05.jpg")
            (temp_path / "manifest.json").write_text(json.dumps({"spec_path": "spec.json"}), encoding="utf-8")
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "per_reference_suppression": True,
                        "references": {
                            "ref_05": {
                                "file": "ref_05.jpg",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            cache = ReferenceCache()
            loaded = cache.preload_from_directory(temp_path)

        self.assertEqual(loaded, 1)
        self.assertEqual(cache.get_candidate_suppression_mode(), "per_reference_top_1")

    def test_spec_can_explicitly_opt_out_of_per_reference_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            shutil.copyfile("data/references/2026_baseline/ref_05.jpg", temp_path / "ref_05.jpg")
            (temp_path / "manifest.json").write_text(json.dumps({"spec_path": "spec.json"}), encoding="utf-8")
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "per_reference_suppression": False,
                        "references": {
                            "ref_05": {
                                "file": "ref_05.jpg",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            cache = ReferenceCache()
            loaded = cache.preload_from_directory(temp_path)

        self.assertEqual(loaded, 1)
        self.assertEqual(cache.get_candidate_suppression_mode(), "global_top_1")


if __name__ == "__main__":
    unittest.main()
