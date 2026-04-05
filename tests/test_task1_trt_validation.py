from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.evaluation.task1_trt_validation import compare_task1_native_onnx_trt_runs


class Task1TrtValidationTests(unittest.TestCase):
    def test_triple_comparison_accepts_matching_runs_with_safe_vram(self) -> None:
        native = {
            "frame_results": [{"frame_index": 1, "bbox_count": 1, "class_histogram": {"0": 1}, "latency_ms": 20.0}],
            "aggregate_class_histogram": {"0": 1},
            "latency_p50_ms": 20.0,
        }
        onnx = {
            "frame_results": [{"frame_index": 1, "bbox_count": 1, "class_histogram": {"0": 1}, "latency_ms": 18.0}],
            "aggregate_class_histogram": {"0": 1},
            "latency_p50_ms": 18.0,
        }
        trt = {
            "frame_results": [{"frame_index": 1, "bbox_count": 1, "class_histogram": {"0": 1}, "latency_ms": 17.0}],
            "aggregate_class_histogram": {"0": 1},
            "latency_p50_ms": 17.0,
            "peak_vram_mb": 512.0,
        }
        comparison = compare_task1_native_onnx_trt_runs(
            native_run=native,
            onnx_run=onnx,
            trt_run=trt,
            runtime_settings=MvpRuntimeSettings(task1_trt_max_vram_mb=6500.0),
        )
        self.assertTrue(comparison["accepted"])
        self.assertTrue(comparison["trt_latency_ok"])
        self.assertTrue(comparison["trt_vram_ok"])

    def test_triple_comparison_rejects_high_vram(self) -> None:
        native = {
            "frame_results": [{"frame_index": 1, "bbox_count": 1, "class_histogram": {"0": 1}, "latency_ms": 20.0}],
            "aggregate_class_histogram": {"0": 1},
            "latency_p50_ms": 20.0,
        }
        onnx = {
            "frame_results": [{"frame_index": 1, "bbox_count": 1, "class_histogram": {"0": 1}, "latency_ms": 18.0}],
            "aggregate_class_histogram": {"0": 1},
            "latency_p50_ms": 18.0,
        }
        trt = {
            "frame_results": [{"frame_index": 1, "bbox_count": 1, "class_histogram": {"0": 1}, "latency_ms": 17.0}],
            "aggregate_class_histogram": {"0": 1},
            "latency_p50_ms": 17.0,
            "peak_vram_mb": 9000.0,
        }
        comparison = compare_task1_native_onnx_trt_runs(
            native_run=native,
            onnx_run=onnx,
            trt_run=trt,
            runtime_settings=MvpRuntimeSettings(task1_trt_max_vram_mb=6500.0),
        )
        self.assertFalse(comparison["accepted"])
        self.assertFalse(comparison["trt_vram_ok"])


if __name__ == "__main__":
    unittest.main()
