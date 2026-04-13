from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from src.task1.experimental.data_pipeline import (
    build_local_yolo_dataset,
    inspect_zip_archive,
    parse_yolo_label_line,
    split_local_dataset_by_frame_index,
)


class Task1DataPipelineTests(unittest.TestCase):
    def test_parse_yolo_label_line_returns_none_for_invalid_rows(self) -> None:
        self.assertIsNone(parse_yolo_label_line("0 0.1 0.2"))
        parsed = parse_yolo_label_line("2 0.5 0.5 0.1 0.2")
        self.assertEqual(parsed["class_id"], 2)

    def test_inspect_zip_archive_detects_yolo_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "labels.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("labels/frame_000000.txt", "0 0.5 0.5 0.1 0.1\n")
                archive.writestr("labels/frame_000005.txt", "")
            inventory = inspect_zip_archive(archive_path)
            self.assertEqual(inventory.label_format, "yolo")
            self.assertEqual(inventory.label_count, 2)
            self.assertEqual(inventory.empty_label_files, 1)
            self.assertEqual(inventory.class_counts, {"0": 1})

    def test_split_local_dataset_by_frame_index_is_blocked_not_random(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir) / "frames"
            frames_dir.mkdir()
            for index in range(10):
                (frames_dir / f"frame_{index * 5:06d}.jpg").write_bytes(b"image")
            split_map = split_local_dataset_by_frame_index(frames_dir)
            self.assertEqual(len(split_map["train"]), 7)
            self.assertEqual(len(split_map["val"]), 1)
            self.assertEqual(len(split_map["test"]), 2)
            train_last = split_map["train"][-1].stem
            test_first = split_map["test"][0].stem
            self.assertLess(int(train_last[-6:]), int(test_first[-6:]))

    def test_build_local_yolo_dataset_collapses_uap_uai_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir) / "frames"
            labels_dir = Path(tmp_dir) / "labels"
            output_dir = Path(tmp_dir) / "yolo"
            frames_dir.mkdir()
            labels_dir.mkdir()
            for index in range(4):
                (frames_dir / f"frame_{index * 5:06d}.jpg").write_bytes(b"image")
            (labels_dir / "frame_000000.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (labels_dir / "frame_000005.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (labels_dir / "frame_000010.txt").write_text("2 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (labels_dir / "frame_000015.txt").write_text("3 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            summary = build_local_yolo_dataset(output_dir, frames_dir=frames_dir, labels_dir=labels_dir)
            self.assertEqual(summary["class_counts"], {"0": 1, "1": 1, "2": 2})
            emitted_path = next((output_dir / "labels").rglob("frame_000010.txt"))
            emitted = emitted_path.read_text(encoding="utf-8")
            self.assertTrue(emitted.startswith("2 "))


if __name__ == "__main__":
    unittest.main()
