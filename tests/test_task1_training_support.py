from __future__ import annotations

import unittest

from src.task1.experimental.training_support import (
    extract_detection_metrics,
    get_training_experiment_defaults,
    render_experiment_results_markdown,
    summarize_learning_dynamics,
)


class _FakeBox:
    def __init__(self) -> None:
        self.ap_class_index = [0, 1]
        self.all_ap = [
            [0.81, 0.72],
            [0.42, 0.21],
        ]
        self.maps = [0.55, 0.18, 0.31]
        self.p = [0.91, 0.52]
        self.r = [0.77, 0.40]


class _FakeResults:
    def __init__(self) -> None:
        self.results_dict = {
            "metrics/precision(B)": 0.7,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": 0.5,
            "metrics/mAP50-95(B)": 0.3,
            "fitness": 0.3,
        }
        self.box = _FakeBox()


class Task1TrainingSupportTests(unittest.TestCase):
    def test_full_experiment_registry_resolves_defaults(self) -> None:
        defaults = get_training_experiment_defaults("combined_yolo11s_full")
        self.assertEqual(defaults["model"], "yolo11s.pt")
        self.assertEqual(defaults["epochs"], 60)
        self.assertEqual(defaults["batch"], 8)

    def test_extract_detection_metrics_marks_missing_ground_truth_class(self) -> None:
        payload = extract_detection_metrics(
            _FakeResults(),
            class_names=["kara_tasiti", "insan", "uap_uai_ozel"],
            split_support={"class_counts": {"0": 10, "1": 4}},
        )
        self.assertEqual(payload["per_class"]["kara_tasiti"]["ap50"], 0.81)
        self.assertEqual(payload["per_class"]["insan"]["ap50_95"], 0.18)
        self.assertEqual(payload["per_class"]["uap_uai_ozel"]["status"], "no_ground_truth_in_split")
        self.assertIsNone(payload["per_class"]["uap_uai_ozel"]["ap50"])

    def test_learning_dynamics_flags_underfit_when_tail_keeps_improving(self) -> None:
        history = []
        for epoch in range(1, 13):
            history.append(
                {
                    "epoch": float(epoch),
                    "train/box_loss": 2.0 - epoch * 0.05,
                    "train/cls_loss": 1.5 - epoch * 0.04,
                    "train/dfl_loss": 1.0 - epoch * 0.03,
                    "metrics/mAP50-95(B)": 0.05 * epoch,
                }
            )
        summary = summarize_learning_dynamics(history)
        self.assertEqual(summary["status"], "still_improving")
        self.assertIn("underfit_or_budget_left", summary["signals"])

    def test_markdown_renderer_includes_training_and_holdout_blocks(self) -> None:
        markdown = render_experiment_results_markdown(
            {
                "decision": "EXPERIMENTAL ONLY",
                "experiments": {
                    "combined_yolo11m_full": {
                        "status": "blocked",
                        "reason": "blocked_by_vram_or_runtime",
                        "model": "yolo11m.pt",
                        "yaml_path": "data/task1_combined.yaml",
                        "best_weights_path": None,
                        "effective_batch": None,
                        "training_metrics": None,
                        "holdout_metrics": None,
                        "learning_dynamics": None,
                    }
                },
            }
        )
        self.assertIn("training metrics", markdown)
        self.assertIn("holdout metrics", markdown)
        self.assertIn("blocked_by_vram_or_runtime", markdown)


if __name__ == "__main__":
    unittest.main()
