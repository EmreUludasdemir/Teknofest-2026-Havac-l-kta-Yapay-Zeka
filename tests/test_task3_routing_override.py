from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.task3.reference_cache import ReferenceCache
from src.task3.routing_policy import RoutingAssignment


def _write_image(path: Path) -> None:
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    gradient = np.tile(np.linspace(0, 255, 48, dtype=np.uint8), (48, 1))
    image[..., 0] = gradient
    image[..., 1] = gradient.T
    image[..., 2] = ((gradient.astype(np.uint16) + gradient.T.astype(np.uint16)) // 2).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


class Task3RoutingOverrideTests(unittest.TestCase):
    def test_nested_routing_override_forces_ref07_to_rgb_only_orb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            _write_image(temp_path / "ref_07.png")
            _write_image(temp_path / "ref_11.png")
            (temp_path / "manifest.json").write_text(json.dumps({"spec_path": "spec.json"}), encoding="utf-8")
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "references": {
                            "ref_07": {
                                "file": "ref_07.png",
                                "routing_override": {
                                    "modality": "rgb",
                                    "detector": "orb",
                                    "rationale": "D6 forensic: cross-modality ORB false-match in thermal video",
                                },
                            },
                            "ref_11": {
                                "file": "ref_11.png",
                                "modality": "thermal",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            def _assignment_for(path: Path) -> RoutingAssignment:
                if path.name == "ref_07.png":
                    return RoutingAssignment(
                        detector="both",
                        detector_modalities=["rgb", "thermal"],
                        confidence="low",
                        rationale="ambiguous auto-routing before manual containment",
                        signals={"modality": "unknown"},
                    )
                return RoutingAssignment(
                    detector="both",
                    detector_modalities=["rgb", "thermal"],
                    confidence="low",
                    rationale="thermal dual-path fallback remains unchanged",
                    signals={"modality": "thermal"},
                )

            cache = ReferenceCache()
            with patch("src.task3.reference_cache.assign_detector", side_effect=_assignment_for):
                loaded = cache.preload_from_directory(temp_path)

        self.assertEqual(loaded, 2)
        self.assertEqual(cache.get_detector("ref_07"), "orb")
        self.assertEqual(cache.get("ref_07")["detector_modalities"], ["rgb"])
        self.assertEqual(cache.get("ref_07")["reference_modality"], "rgb")
        self.assertEqual(cache.get_detector("ref_11"), "both")
        self.assertEqual(cache.get("ref_11")["detector_modalities"], ["rgb", "thermal"])
        diagnostics = cache.get_routing_diagnostics()
        self.assertEqual(diagnostics["ref_07"]["override"]["detector"], "orb")
        self.assertEqual(diagnostics["ref_11"]["override"], None)

    def test_override_free_reference_keeps_auto_routing_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            _write_image(temp_path / "ref_11.png")
            (temp_path / "manifest.json").write_text(json.dumps({"spec_path": "spec.json"}), encoding="utf-8")
            (temp_path / "spec.json").write_text(
                json.dumps({"references": {"ref_11": {"file": "ref_11.png", "modality": "thermal"}}}),
                encoding="utf-8",
            )

            assignment = RoutingAssignment(
                detector="both",
                detector_modalities=["rgb", "thermal"],
                confidence="low",
                rationale="thermal dual-path fallback remains unchanged",
                signals={"modality": "thermal"},
            )
            cache = ReferenceCache()
            with patch("src.task3.reference_cache.assign_detector", return_value=assignment):
                cache.preload_from_directory(temp_path)

        self.assertEqual(cache.get_detector("ref_11"), "both")
        self.assertEqual(cache.get("ref_11")["detector_modalities"], ["rgb", "thermal"])
        self.assertEqual(cache.get_overrides_applied(), [])

    def test_invalid_nested_routing_override_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            _write_image(temp_path / "ref_07.png")
            (temp_path / "manifest.json").write_text(json.dumps({"spec_path": "spec.json"}), encoding="utf-8")
            (temp_path / "spec.json").write_text(
                json.dumps(
                    {
                        "references": {
                            "ref_07": {
                                "file": "ref_07.png",
                                "routing_override": "orb",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            cache = ReferenceCache()
            with self.assertRaises(ValueError):
                cache.preload_from_directory(temp_path)


if __name__ == "__main__":
    unittest.main()
