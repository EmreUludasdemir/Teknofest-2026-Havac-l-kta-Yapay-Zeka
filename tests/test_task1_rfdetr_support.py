from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.core.frame_state import CanonicalDetection
from src.task1.experimental.rfdetr_support import (
    build_tile_windows,
    canonical_task1_class_from_uavdt_label,
    canonical_task1_class_from_visdrone_category,
    discover_local_task1_artifacts,
    discover_task1_datasets,
    offset_detection_to_frame,
)


class Task1RfDetrSupportTests(unittest.TestCase):
    def test_artifact_discovery_finds_baseline_and_rfdetr_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "yolo26n.pt").write_bytes(b"baseline")
            (root / "yolo11n.onnx").write_bytes(b"control")
            (root / "nested").mkdir()
            (root / "nested" / "rfdetr-base.onnx").write_bytes(b"experimental")
            (root / "nested" / "rtdetr-l.pt").write_bytes(b"transformer")
            probe = discover_local_task1_artifacts(root)
            self.assertEqual(len(probe.baseline_yolo26n_paths), 1)
            self.assertEqual(len(probe.control_yolo11n_paths), 1)
            self.assertEqual(len(probe.rfdetr_candidate_paths), 1)
            self.assertEqual(len(probe.rtdetr_candidate_paths), 1)

    def test_dataset_discovery_finds_visdrone_and_uavdt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "VisDrone2019-DET").mkdir()
            (root / "UAVDT").mkdir()
            datasets = discover_task1_datasets((root,))
            self.assertEqual([item.name for item in datasets], ["UAVDT", "VisDrone2019-DET"])

    def test_mapping_helpers_keep_vehicle_human_scope_explicit(self) -> None:
        self.assertEqual(canonical_task1_class_from_visdrone_category(1), 1)
        self.assertEqual(canonical_task1_class_from_visdrone_category(4), 0)
        self.assertIsNone(canonical_task1_class_from_visdrone_category(11))
        self.assertEqual(canonical_task1_class_from_uavdt_label("car"), 0)
        self.assertEqual(canonical_task1_class_from_uavdt_label("person"), 1)
        self.assertIsNone(canonical_task1_class_from_uavdt_label("uap"))

    def test_tile_window_builder_and_offset_keep_frame_bounds(self) -> None:
        windows = build_tile_windows(1920, 1080, tile_size=960, overlap=0.25)
        self.assertGreaterEqual(len(windows), 4)
        detection = CanonicalDetection(
            class_id=0,
            top_left_x=10.0,
            top_left_y=20.0,
            bottom_right_x=50.0,
            bottom_right_y=80.0,
            metadata={"score": 0.9},
        )
        shifted = offset_detection_to_frame(
            detection,
            x_offset=900,
            y_offset=1000,
            frame_width=1920,
            frame_height=1080,
        )
        self.assertEqual(shifted.top_left_x, 910.0)
        self.assertEqual(shifted.top_left_y, 1020.0)
        self.assertLessEqual(shifted.bottom_right_y, 1079.0)


if __name__ == "__main__":
    unittest.main()
