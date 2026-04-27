from __future__ import annotations

import unittest

from src.core.frame_state import CanonicalUndefinedObject
from src.task3.no_match_logic import (
    SUPPRESSION_MODE_GLOBAL_TOP_1,
    SUPPRESSION_MODE_PER_REFERENCE_TOP_1,
    filter_no_match_candidates,
)


def _candidate(reference_id: str, score: float) -> CanonicalUndefinedObject:
    return CanonicalUndefinedObject(
        object_id=reference_id,
        top_left_x=0.0,
        top_left_y=0.0,
        bottom_right_x=10.0,
        bottom_right_y=10.0,
        metadata={"match_score": round(float(score), 6)},
    )


class Task3PerReferenceSuppressionTests(unittest.TestCase):
    def test_global_mode_keeps_existing_global_top1_behavior(self) -> None:
        matches = [
            _candidate("ref_02", 0.91),
            _candidate("ref_05", 0.592158),
        ]

        filtered = filter_no_match_candidates(
            matches,
            min_score=0.70,
            suppression_mode=SUPPRESSION_MODE_GLOBAL_TOP_1,
        )

        self.assertEqual([item.object_id for item in filtered], ["ref_02"])

    def test_per_reference_mode_keeps_best_candidate_per_reference(self) -> None:
        matches = [
            _candidate("ref_02", 0.91),
            _candidate("ref_05", 0.592158),
        ]

        filtered = filter_no_match_candidates(
            matches,
            min_score=0.4520,
            mode="yoloe_vp_lightglue",
            yoloe_min_score=0.4520,
            modality="rgb",
            yoloe_thermal_min_score=0.50,
            suppression_mode=SUPPRESSION_MODE_PER_REFERENCE_TOP_1,
        )

        self.assertEqual([item.object_id for item in filtered], ["ref_02", "ref_05"])

    def test_per_reference_mode_preserves_top1_within_same_reference(self) -> None:
        matches = [
            _candidate("ref_05", 0.88),
            _candidate("ref_05", 0.70),
            _candidate("ref_06", 0.52),
        ]

        filtered = filter_no_match_candidates(
            matches,
            min_score=0.4520,
            mode="yoloe_vp_lightglue",
            yoloe_min_score=0.4520,
            modality="rgb",
            yoloe_thermal_min_score=0.50,
            suppression_mode=SUPPRESSION_MODE_PER_REFERENCE_TOP_1,
        )

        self.assertEqual([item.object_id for item in filtered], ["ref_05", "ref_06"])
        self.assertEqual(float(filtered[0].metadata["match_score"]), 0.88)


if __name__ == "__main__":
    unittest.main()
