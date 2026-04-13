from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.task1_rfdetr_comparison import (
    Task1GroundTruth,
    Task1Prediction,
    compute_task1_label_aware_metrics,
    run_task1_rfdetr_comparison,
)
from src.task1.experimental.rfdetr_support import ArtifactProbe, PythonRuntimeProbe


class Task1RfDetrComparisonTests(unittest.TestCase):
    def test_label_aware_metrics_gate_vehicle_motion_and_landing_status(self) -> None:
        ground_truth = [
            Task1GroundTruth("frame-1", 0, 0.0, 0.0, 10.0, 10.0, motion_status=1),
            Task1GroundTruth("frame-1", 2, 20.0, 20.0, 40.0, 40.0, landing_status=1),
        ]
        predictions = [
            Task1Prediction("frame-1", 0, 0.95, 0.0, 0.0, 10.0, 10.0, motion_status=1),
            Task1Prediction("frame-1", 2, 0.90, 20.0, 20.0, 40.0, 40.0, landing_status=0),
        ]
        metrics = compute_task1_label_aware_metrics(ground_truth, predictions)
        self.assertEqual(metrics["overall_mAP"], 0.5)
        self.assertEqual(metrics["motion_status_sensitive_performance"], 1.0)
        self.assertEqual(metrics["landing_status_sensitive_performance"], 0.0)

    def test_proxy_only_run_writes_reports_for_yolo_first_variants(self) -> None:
        runtime_probe = PythonRuntimeProbe(
            selected_path=r"C:\Python312\python.exe",
            version="3.12.10",
            is_repo_compatible=True,
            checked=[{"path": r"C:\Python312\python.exe", "status": "ok"}],
            issues=[],
        )
        proxy_rows = {
            "baseline_yolo26n": {
                "sample_count": 4,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 30.0,
                "zero_detection_rate": 0.25,
                "duplicate_ratio": 0.05,
                "small_box_count": 2,
                "aggregate_class_histogram": {"0": 4, "1": 1},
                "used_fallback": False,
                "tile_probe": None,
            },
            "control_yolo11n": {
                "sample_count": 4,
                "latency_p50_ms": 18.0,
                "latency_p95_ms": 28.0,
                "zero_detection_rate": 0.5,
                "duplicate_ratio": 0.02,
                "small_box_count": 1,
                "aggregate_class_histogram": {"0": 3},
                "used_fallback": False,
                "tile_probe": None,
            },
            "tiled_yolo26n_probe": {
                "sample_count": 4,
                "latency_p50_ms": 32.0,
                "latency_p95_ms": 50.0,
                "zero_detection_rate": 0.0,
                "duplicate_ratio": 0.10,
                "small_box_count": 5,
                "aggregate_class_histogram": {"0": 4, "1": 2},
                "used_fallback": False,
                "tile_probe": {"tile_count_total": 16, "raw_detection_count": 12, "merged_detection_count": 8},
            },
        }

        def fake_proxy(_settings, *, spec, **_kwargs):
            return proxy_rows[spec.name]

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.evaluation.task1_rfdetr_comparison.discover_python_runtime", return_value=runtime_probe), patch(
                "src.evaluation.task1_rfdetr_comparison.discover_local_task1_artifacts",
                return_value=ArtifactProbe(
                    baseline_yolo26n_paths=[r"C:\models\yolo26n.pt"],
                    control_yolo11n_paths=[r"C:\models\yolo11n.pt"],
                    rtdetr_candidate_paths=[],
                    rfdetr_candidate_paths=[],
                ),
            ), patch(
                "src.evaluation.task1_rfdetr_comparison.discover_task1_datasets",
                return_value=[],
            ), patch(
                "src.evaluation.task1_rfdetr_comparison.discover_task1_validation_video",
                return_value=Path("validation.mp4"),
            ), patch(
                "src.evaluation.task1_rfdetr_comparison.sample_task1_validation_frames",
                return_value=[object(), object(), object(), object()],
            ), patch(
                "src.evaluation.task1_rfdetr_comparison._run_variant_profile",
                return_value={"available": True, "warm_p50_latency_ms": 12.5},
            ), patch(
                "src.evaluation.task1_rfdetr_comparison._run_variant_proxy_replay",
                side_effect=fake_proxy,
            ):
                payload = run_task1_rfdetr_comparison(output_dir=tmp_dir, sample_count=4)
            self.assertEqual(payload["decision"], "EXPERIMENTAL ONLY")
            self.assertEqual(payload["proxy_summary"]["highest_small_box_variant"], "tiled_yolo26n_probe")
            self.assertEqual(payload["label_aware_comparison"]["status"], "blocked_by_labels")
            report_path = Path(tmp_dir) / "task1_rfdetr_vs_yolo26n.json"
            self.assertTrue(report_path.exists())
            on_disk = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["decision"], "EXPERIMENTAL ONLY")
            self.assertIn("tiled_yolo26n_probe", on_disk["variants"])

    def test_variant_registry_can_mix_proxy_and_transformer_gate(self) -> None:
        runtime_probe = PythonRuntimeProbe(
            selected_path=r"C:\Python312\python.exe",
            version="3.12.10",
            is_repo_compatible=True,
            checked=[{"path": r"C:\Python312\python.exe", "status": "ok"}],
            issues=[],
        )

        def fake_proxy(_settings, *, spec, **_kwargs):
            return {
                "sample_count": 2,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "zero_detection_rate": 0.0,
                "duplicate_ratio": 0.0,
                "small_box_count": 0,
                "aggregate_class_histogram": {"0": 1},
                "used_fallback": False,
                "tile_probe": None,
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.evaluation.task1_rfdetr_comparison.discover_python_runtime", return_value=runtime_probe), patch(
                "src.evaluation.task1_rfdetr_comparison.discover_local_task1_artifacts",
                return_value=ArtifactProbe(
                    baseline_yolo26n_paths=[r"C:\models\yolo26n.pt"],
                    control_yolo11n_paths=[],
                    rtdetr_candidate_paths=[],
                    rfdetr_candidate_paths=[],
                ),
            ), patch(
                "src.evaluation.task1_rfdetr_comparison.discover_task1_datasets",
                return_value=[],
            ), patch(
                "src.evaluation.task1_rfdetr_comparison.discover_task1_validation_video",
                return_value=Path("validation.mp4"),
            ), patch(
                "src.evaluation.task1_rfdetr_comparison.sample_task1_validation_frames",
                return_value=[object(), object()],
            ), patch(
                "src.evaluation.task1_rfdetr_comparison._run_variant_profile",
                return_value={"available": True, "warm_p50_latency_ms": 12.5},
            ), patch(
                "src.evaluation.task1_rfdetr_comparison._run_variant_proxy_replay",
                side_effect=fake_proxy,
            ):
                payload = run_task1_rfdetr_comparison(
                    output_dir=tmp_dir,
                    sample_count=2,
                    variants=["baseline_yolo26n", "rfdetr_base"],
                )
            self.assertEqual(payload["variants"]["baseline_yolo26n"]["status"], "proxy_only")
            self.assertEqual(payload["variants"]["rfdetr_base"]["status"], "blocked_missing_artifact")
            self.assertIn("no_local_rfdetr_base_artifact_found", payload["variants"]["rfdetr_base"]["issues"])

    def test_candidate_probe_variant_accepts_explicit_candidate_path(self) -> None:
        runtime_probe = PythonRuntimeProbe(
            selected_path=r"C:\Python312\python.exe",
            version="3.12.10",
            is_repo_compatible=True,
            checked=[{"path": r"C:\Python312\python.exe", "status": "ok"}],
            issues=[],
        )

        def fake_proxy(_settings, *, spec, **_kwargs):
            return {
                "sample_count": 1,
                "latency_p50_ms": 10.0,
                "latency_p95_ms": 11.0,
                "zero_detection_rate": 0.0,
                "duplicate_ratio": 0.0,
                "small_box_count": 0,
                "aggregate_class_histogram": {"0": 1, "2": 1},
                "used_fallback": False,
                "tile_probe": None,
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.evaluation.task1_rfdetr_comparison.discover_python_runtime", return_value=runtime_probe), patch(
                "src.evaluation.task1_rfdetr_comparison.discover_local_task1_artifacts",
                return_value=ArtifactProbe(),
            ), patch(
                "src.evaluation.task1_rfdetr_comparison.discover_task1_datasets",
                return_value=[],
            ), patch(
                "src.evaluation.task1_rfdetr_comparison.discover_task1_validation_video",
                return_value=Path("validation.mp4"),
            ), patch(
                "src.evaluation.task1_rfdetr_comparison.sample_task1_validation_frames",
                return_value=[object()],
            ), patch(
                "src.evaluation.task1_rfdetr_comparison._run_variant_profile",
                return_value={"available": True, "warm_p50_latency_ms": 9.0},
            ), patch(
                "src.evaluation.task1_rfdetr_comparison._run_variant_proxy_replay",
                side_effect=fake_proxy,
            ):
                payload = run_task1_rfdetr_comparison(
                    output_dir=tmp_dir,
                    sample_count=1,
                    variants=["candidate_probe"],
                    candidate_name="task1_candidate",
                    candidate_path=r"C:\models\candidate.pt",
                )
            self.assertEqual(payload["variants"]["candidate_probe"]["status"], "proxy_only")
            self.assertEqual(payload["variants"]["candidate_probe"]["artifact_path"], r"C:\models\candidate.pt")


if __name__ == "__main__":
    unittest.main()
