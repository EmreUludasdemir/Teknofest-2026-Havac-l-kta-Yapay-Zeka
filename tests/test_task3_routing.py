from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame, FrameEnvelope
from src.task3.matcher import Task3Matcher
from src.task3.reference_cache import ReferenceCache


def _pgm_bytes(width: int, height: int, fill: int = 127) -> bytes:
    return f"P5\n{width} {height}\n255\n".encode("ascii") + bytes([fill] * (width * height))


def _candidate(object_id: str, score: float, source: str) -> CanonicalUndefinedObject:
    return CanonicalUndefinedObject(
        object_id=object_id,
        top_left_x=0.0,
        top_left_y=0.0,
        bottom_right_x=10.0,
        bottom_right_y=10.0,
        metadata={"match_score": score, "matcher_source": source},
    )


class Task3RoutingTests(unittest.TestCase):
    def _frame(self) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url="http://mock/frames/1/",
            image_url="/frame.jpg",
            video_name="rgb_reference_session",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
        )

    def test_reference_spec_loading_respects_detector_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref_001.pgm").write_bytes(_pgm_bytes(8, 8))
            (temp_path / "manifest.json").write_text(
                json.dumps({"spec_path": "spec.json", "items": [{"reference_id": "ref_001", "output_file": "ref_001.pgm"}]}),
                encoding="utf-8",
            )
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "references": {
                            "ref_001": {
                                "file": "ref_001.pgm",
                                "modality": "rgb",
                                "detector": "orb",
                                "dimensions": [8, 8],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            cache = ReferenceCache()
            cache.preload_from_directory(temp_path)

            self.assertEqual(cache.get_detector("ref_001"), "orb")
            self.assertEqual(cache.get("ref_001")["reference_metadata"]["modality"], "rgb")

    def test_invalid_detector_value_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref_001.pgm").write_bytes(_pgm_bytes(8, 8))
            (temp_path / "manifest.json").write_text(
                json.dumps({"spec_path": "spec.json", "items": [{"reference_id": "ref_001", "output_file": "ref_001.pgm"}]}),
                encoding="utf-8",
            )
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "references": {
                            "ref_001": {
                                "file": "ref_001.pgm",
                                "modality": "rgb",
                                "detector": "xyz",
                                "dimensions": [8, 8],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            cache = ReferenceCache()
            with self.assertRaises(ValueError):
                cache.preload_from_directory(temp_path)

    def test_routing_dispatch_splits_references_between_yoloe_and_orb(self) -> None:
        cache = ReferenceCache()
        cache.put("ref_01", {"detector": "yoloe"})
        cache.put("ref_02", {"detector": "orb"})
        matcher = Task3Matcher(reference_cache=cache, runtime_settings=MvpRuntimeSettings(task3_mode="yoloe_vp_lightglue"))

        with patch.object(
            Task3Matcher,
            "_match_with_yoloe_vp_lightglue",
            return_value=[_candidate("ref_01", 0.61, "task3_yoloe_vp_lightglue")],
        ) as yoloe_mock, patch.object(
            Task3Matcher,
            "_match_with_real_orb_only",
            return_value=[_candidate("ref_02", 0.91, "task3_orb_bf_homography")],
        ) as orb_mock:
            matches = matcher.match(self._frame(), b"", ["ref_01", "ref_02"], decoded_frame=None, mode="yoloe_vp_lightglue")

        self.assertEqual([item.object_id for item in matches], ["ref_02", "ref_01"])
        yoloe_mock.assert_called_once()
        self.assertEqual(yoloe_mock.call_args.args[-1], ["ref_01"])
        orb_mock.assert_called_once()
        self.assertEqual(orb_mock.call_args.args[-1], ["ref_02"])
        self.assertEqual(matcher.last_run_info["yoloe_routed_refs"], ["ref_01"])
        self.assertEqual(matcher.last_run_info["orb_routed_refs"], ["ref_02"])
        self.assertEqual(matcher.last_run_info["yoloe_candidates_total"], 1)
        self.assertEqual(matcher.last_run_info["orb_candidates_total"], 1)
        self.assertEqual(matcher.last_run_info["orb_placeholder_candidates"], 0)
        self.assertEqual(matcher.last_run_info["effective_mode"], "per_reference_routing")

    def test_rgb_only_detector_is_excluded_from_thermal_frames(self) -> None:
        cache = ReferenceCache()
        cache.put("ref_01", {"detector": "yoloe"})
        cache.put("ref_02", {"detector": "orb", "detector_modalities": ["rgb"]})
        matcher = Task3Matcher(reference_cache=cache, runtime_settings=MvpRuntimeSettings(task3_mode="yoloe_vp_lightglue"))
        decoded = DecodedFrame(
            bgr=object(),
            gray=object(),
            width=16,
            height=16,
            channel_count=1,
            modality="thermal",
            frame_index=0,
        )

        with patch.object(
            Task3Matcher,
            "_match_with_yoloe_vp_lightglue",
            return_value=[_candidate("ref_01", 0.61, "task3_yoloe_vp_lightglue")],
        ) as yoloe_mock, patch.object(Task3Matcher, "_match_with_real_orb_only", return_value=[]) as orb_mock:
            matches = matcher.match(self._frame(), b"", ["ref_01", "ref_02"], decoded_frame=decoded, mode="yoloe_vp_lightglue")

        self.assertEqual([item.object_id for item in matches], ["ref_01"])
        yoloe_mock.assert_called_once()
        self.assertEqual(yoloe_mock.call_args.args[-1], ["ref_01"])
        orb_mock.assert_not_called()
        self.assertEqual(matcher.last_run_info["yoloe_routed_refs"], ["ref_01"])
        self.assertEqual(matcher.last_run_info["orb_routed_refs"], [])

    def test_missing_detector_modalities_is_backward_compatible(self) -> None:
        cache = ReferenceCache()
        cache.put("ref_01", {"detector": "yoloe"})
        cache.put("ref_02", {"detector": "orb"})

        filtered = cache.filter_reference_ids_by_detector_modality(["ref_01", "ref_02"], modality="thermal")

        self.assertEqual(filtered, ["ref_01", "ref_02"])

    def test_invalid_detector_modality_value_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref_001.pgm").write_bytes(_pgm_bytes(8, 8))
            (temp_path / "manifest.json").write_text(
                json.dumps({"spec_path": "spec.json", "items": [{"reference_id": "ref_001", "output_file": "ref_001.pgm"}]}),
                encoding="utf-8",
            )
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "references": {
                            "ref_001": {
                                "file": "ref_001.pgm",
                                "modality": "rgb",
                                "detector": "orb",
                                "detector_modalities": ["infrared"],
                                "dimensions": [8, 8],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            cache = ReferenceCache()
            with self.assertRaises(ValueError):
                cache.preload_from_directory(temp_path)

    def test_routed_orb_path_excludes_placeholder_candidates(self) -> None:
        cache = ReferenceCache()
        cache.put("ref_02", {"detector": "orb"})
        matcher = Task3Matcher(reference_cache=cache, runtime_settings=MvpRuntimeSettings(task3_mode="yoloe_vp_lightglue"))
        decoded = DecodedFrame(
            bgr=object(),
            gray=object(),
            width=16,
            height=16,
            channel_count=1,
            modality="rgb",
            frame_index=0,
        )

        with patch.object(Task3Matcher, "_match_with_descriptors", return_value=[]), patch.object(
            Task3Matcher,
            "_match_with_template",
            return_value=[_candidate("ref_02", 0.82, "task3_placeholder")],
        ):
            matches = matcher.match(self._frame(), b"", ["ref_02"], decoded_frame=decoded, mode="yoloe_vp_lightglue")

        self.assertEqual(matches, [])
        self.assertEqual(matcher.last_run_info["orb_routed_refs"], ["ref_02"])
        self.assertEqual(matcher.last_run_info["orb_candidates_total"], 0)
        self.assertEqual(matcher.last_run_info["orb_placeholder_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
