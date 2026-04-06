from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import MvpRuntimeSettings
from src.task3.experimental import weight_staging


class Task3WeightStagingTests(unittest.TestCase):
    def test_empty_search_roots_report_weight_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = weight_staging.evaluate_weight_staging(
                MvpRuntimeSettings(),
                search_roots=[Path(tmp_dir)],
            )
        self.assertEqual(payload["status"], weight_staging.STAGING_WEIGHT_MISSING)
        self.assertEqual(payload["chosen_weight"], None)

    def test_runtime_incompatible_when_yoloe_missing(self) -> None:
        with patch.object(weight_staging, "YOLOE", None):
            payload = weight_staging.evaluate_weight_staging(
                MvpRuntimeSettings(),
                search_roots=[],
            )
        self.assertEqual(payload["status"], weight_staging.STAGING_RUNTIME_INCOMPATIBLE)


if __name__ == "__main__":
    unittest.main()
