from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.export_readiness import evaluate_export_readiness


class ExportReadinessTests(unittest.TestCase):
    def test_missing_reports_return_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = evaluate_export_readiness(reports_dir=tmp_dir, tests_ok=True)
            self.assertFalse(payload["ready"])
            self.assertTrue((Path(tmp_dir) / "export_readiness.md").exists())

    def test_complete_mock_reports_can_return_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            (base / "profiling").mkdir(parents=True, exist_ok=True)
            (base / "task2_long_sequence_before_summary.json").write_text(
                json.dumps({"aggregate": {"health0_drift_accumulation": 50.0, "fallback_rate": 0.1, "recovery_error_after_health_returns_to_1": 0.2}}),
                encoding="utf-8",
            )
            (base / "task2_long_sequence_after_summary.json").write_text(
                json.dumps({"aggregate": {"health0_drift_accumulation": 35.0, "fallback_rate": 0.05, "recovery_error_after_health_returns_to_1": 0.1}}),
                encoding="utf-8",
            )
            (base / "task3_baseline_orb_summary.json").write_text(
                json.dumps({"aggregate": {"false_positive_proxy_count": 20, "no_match_suppression_rate": 0.4, "accepted_match_count": 50}}),
                encoding="utf-8",
            )
            (base / "task3_baseline_learned_summary.json").write_text(
                json.dumps({"aggregate": {"false_positive_proxy_count": 10, "no_match_suppression_rate": 0.45, "accepted_match_count": 48}}),
                encoding="utf-8",
            )
            (base / "profiling" / "task1_real_profile_cpu.json").write_text(
                json.dumps({"results": [], "recommendation": {"model_fallback_candidate": "yolo26n"}}),
                encoding="utf-8",
            )
            (base / "profiling" / "task1_real_profile_gpu.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "available": True,
                                "peak_vram_mb": 4200.0,
                                "diagnostics": {"last_output": {"runtime_device": "cuda:0"}},
                            }
                        ],
                        "recommendation": {"model_fallback_candidate": "yolo26n"},
                    }
                ),
                encoding="utf-8",
            )
            payload = evaluate_export_readiness(reports_dir=tmp_dir, tests_ok=True)
            self.assertTrue(payload["ready"])


if __name__ == "__main__":
    unittest.main()
