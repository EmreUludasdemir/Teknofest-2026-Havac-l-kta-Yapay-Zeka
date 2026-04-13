from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.task1.experimental.human_focus import build_human_focus_dataset, collect_human_focus_audit


class Task1HumanFocusTests(unittest.TestCase):
    def test_collect_human_focus_audit_flags_local_only_test_humans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            for split in ("train", "val", "test"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            (root / "images" / "train" / "local_a.jpg").write_bytes(b"x")
            (root / "labels" / "train" / "local_a.txt").write_text("1 0.5 0.5 0.02 0.02\n", encoding="utf-8")
            (root / "images" / "train" / "public_b.jpg").write_bytes(b"x")
            (root / "labels" / "train" / "public_b.txt").write_text("1 0.5 0.5 0.01 0.01\n", encoding="utf-8")
            (root / "images" / "test" / "local_c.jpg").write_bytes(b"x")
            (root / "labels" / "test" / "local_c.txt").write_text("1 0.5 0.5 0.01 0.01\n", encoding="utf-8")
            audit = collect_human_focus_audit(root)
            self.assertIn("test_humans_are_local_only", audit["diagnosis"])

    def test_build_human_focus_dataset_repeats_local_human_files_without_touching_val_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir) / "base"
            out = Path(tmp_dir) / "out"
            yaml_path = Path(tmp_dir) / "variant.yaml"
            for split in ("train", "val", "test"):
                (base / "images" / split).mkdir(parents=True)
                (base / "labels" / split).mkdir(parents=True)
            (base / "images" / "train" / "local_frame_000000.jpg").write_bytes(b"x")
            (base / "labels" / "train" / "local_frame_000000.txt").write_text("1 0.5 0.5 0.02 0.02\n", encoding="utf-8")
            (base / "images" / "val" / "local_frame_000005.jpg").write_bytes(b"x")
            (base / "labels" / "val" / "local_frame_000005.txt").write_text("1 0.5 0.5 0.02 0.02\n", encoding="utf-8")
            (base / "images" / "test" / "local_frame_000010.jpg").write_bytes(b"x")
            (base / "labels" / "test" / "local_frame_000010.txt").write_text("1 0.5 0.5 0.02 0.02\n", encoding="utf-8")
            summary = build_human_focus_dataset(
                base,
                out,
                yaml_path,
                class_names=["kara_tasiti", "insan", "uap_uai_ozel"],
                local_human_repeat_factor=3,
            )
            self.assertEqual(summary["added_repeat_images"], 2)
            self.assertTrue((out / "labels" / "train" / "local_frame_000000__humanrep1.txt").exists())
            self.assertFalse((out / "labels" / "val" / "local_frame_000005__humanrep1.txt").exists())
            self.assertFalse((out / "labels" / "test" / "local_frame_000010__humanrep1.txt").exists())


if __name__ == "__main__":
    unittest.main()
