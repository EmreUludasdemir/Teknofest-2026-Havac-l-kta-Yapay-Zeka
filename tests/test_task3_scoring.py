from __future__ import annotations

import unittest

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject
from src.task3.no_match_logic import (
    ORB_SCORE_CONFIDENCE_WEIGHT,
    ORB_SCORE_MATCHES_WEIGHT,
    compute_mode_candidate_score,
    filter_no_match_candidates,
    normalize_yoloe_match_count,
)


def _candidate(score: float) -> CanonicalUndefinedObject:
    return CanonicalUndefinedObject(
        object_id="ref_001",
        top_left_x=0.0,
        top_left_y=0.0,
        bottom_right_x=10.0,
        bottom_right_y=10.0,
        metadata={"match_score": round(float(score), 6)},
    )


class Task3ScoringTests(unittest.TestCase):
    def test_orb_formula_regression_guard(self) -> None:
        score = compute_mode_candidate_score(
            confidence=0.5,
            normalized_matches=0.3,
            inlier_ratio=0.4,
            mode="orb_template",
            yoloe_confidence_weight=0.20,
            yoloe_matches_weight=0.55,
            yoloe_inlier_weight=0.25,
        )
        expected = (0.5 * ORB_SCORE_CONFIDENCE_WEIGHT) + (0.3 * ORB_SCORE_MATCHES_WEIGHT)
        self.assertAlmostEqual(score, expected, places=6)
        self.assertAlmostEqual(score, 0.44, places=6)

    def test_yoloe_formula_uses_three_term_weights(self) -> None:
        settings = MvpRuntimeSettings()
        score = compute_mode_candidate_score(
            confidence=0.15,
            normalized_matches=normalize_yoloe_match_count(
                match_count=20,
                normalization_scale=settings.task3_yoloe_match_normalization_scale,
            ),
            inlier_ratio=0.4,
            mode="yoloe_vp_lightglue",
            yoloe_confidence_weight=settings.task3_yoloe_score_confidence_weight,
            yoloe_matches_weight=settings.task3_yoloe_score_matches_weight,
            yoloe_inlier_weight=settings.task3_yoloe_score_inlier_weight,
        )
        expected = (
            (0.15 * settings.task3_yoloe_score_confidence_weight)
            + (0.4 * settings.task3_yoloe_score_matches_weight)
            + (0.4 * settings.task3_yoloe_score_inlier_weight)
        )
        self.assertAlmostEqual(score, expected, places=6)
        self.assertAlmostEqual(score, 0.3375, places=6)

    def test_yoloe_normalization_retains_match_count_variance(self) -> None:
        settings = MvpRuntimeSettings()
        low_score = compute_mode_candidate_score(
            confidence=0.2,
            normalized_matches=normalize_yoloe_match_count(
                match_count=20,
                normalization_scale=settings.task3_yoloe_match_normalization_scale,
            ),
            inlier_ratio=0.3,
            mode="yoloe_vp_lightglue",
            yoloe_confidence_weight=settings.task3_yoloe_score_confidence_weight,
            yoloe_matches_weight=settings.task3_yoloe_score_matches_weight,
            yoloe_inlier_weight=settings.task3_yoloe_score_inlier_weight,
        )
        high_score = compute_mode_candidate_score(
            confidence=0.2,
            normalized_matches=normalize_yoloe_match_count(
                match_count=400,
                normalization_scale=settings.task3_yoloe_match_normalization_scale,
            ),
            inlier_ratio=0.3,
            mode="yoloe_vp_lightglue",
            yoloe_confidence_weight=settings.task3_yoloe_score_confidence_weight,
            yoloe_matches_weight=settings.task3_yoloe_score_matches_weight,
            yoloe_inlier_weight=settings.task3_yoloe_score_inlier_weight,
        )

        self.assertLess(low_score, high_score)
        self.assertAlmostEqual(low_score, 0.336, places=6)
        self.assertAlmostEqual(high_score, 0.702, places=6)

    def test_yoloe_normalization_caps_at_one(self) -> None:
        settings = MvpRuntimeSettings()
        normalized = normalize_yoloe_match_count(
            match_count=500,
            normalization_scale=settings.task3_yoloe_match_normalization_scale,
        )
        self.assertLessEqual(normalized, 1.0)
        self.assertAlmostEqual(normalized, 1.0, places=6)

    def test_threshold_dispatch_uses_mode_specific_cutoff(self) -> None:
        settings = MvpRuntimeSettings()
        shared_score = compute_mode_candidate_score(
            confidence=0.15,
            normalized_matches=1.0,
            inlier_ratio=0.4,
            mode="yoloe_vp_lightglue",
            yoloe_confidence_weight=settings.task3_yoloe_score_confidence_weight,
            yoloe_matches_weight=settings.task3_yoloe_score_matches_weight,
            yoloe_inlier_weight=settings.task3_yoloe_score_inlier_weight,
        )
        orb_score = compute_mode_candidate_score(
            confidence=0.15,
            normalized_matches=1.0,
            inlier_ratio=0.4,
            mode="orb_template",
            yoloe_confidence_weight=settings.task3_yoloe_score_confidence_weight,
            yoloe_matches_weight=settings.task3_yoloe_score_matches_weight,
            yoloe_inlier_weight=settings.task3_yoloe_score_inlier_weight,
        )

        yoloe_filtered = filter_no_match_candidates(
            [_candidate(shared_score)],
            min_score=settings.task3_min_score,
            mode="yoloe_vp_lightglue",
            yoloe_min_score=settings.task3_yoloe_min_score,
            modality="rgb",
            yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
            ambiguity_margin=settings.task3_ambiguity_margin,
        )
        orb_filtered = filter_no_match_candidates(
            [_candidate(orb_score)],
            min_score=settings.task3_min_score,
            mode="orb_template",
            yoloe_min_score=settings.task3_yoloe_min_score,
            modality="rgb",
            yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
            ambiguity_margin=settings.task3_ambiguity_margin,
        )

        self.assertGreaterEqual(shared_score, settings.task3_yoloe_min_score)
        self.assertLess(orb_score, settings.task3_min_score)
        self.assertEqual(len(yoloe_filtered), 1)
        self.assertEqual(orb_filtered, [])

    def test_thermal_modality_uses_higher_yoloe_cutoff(self) -> None:
        # 2026-04-21: thermal threshold raised to 0.50 > RGB threshold 0.4520.
        # Thermal is now more restrictive: a score that passes RGB is rejected by thermal.
        settings = MvpRuntimeSettings()
        borderline_score = 0.47  # passes RGB (>=0.4520) but fails thermal (<0.50)

        thermal_filtered = filter_no_match_candidates(
            [_candidate(borderline_score)],
            min_score=settings.task3_min_score,
            mode="yoloe_vp_lightglue",
            yoloe_min_score=settings.task3_yoloe_min_score,
            modality="thermal",
            yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
            ambiguity_margin=settings.task3_ambiguity_margin,
        )
        rgb_filtered = filter_no_match_candidates(
            [_candidate(borderline_score)],
            min_score=settings.task3_min_score,
            mode="yoloe_vp_lightglue",
            yoloe_min_score=settings.task3_yoloe_min_score,
            modality="rgb",
            yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
            ambiguity_margin=settings.task3_ambiguity_margin,
        )

        self.assertGreaterEqual(borderline_score, settings.task3_yoloe_min_score)
        self.assertLess(borderline_score, settings.task3_yoloe_thermal_min_score)
        self.assertEqual(thermal_filtered, [])
        self.assertEqual(len(rgb_filtered), 1)

    def test_yoloe_inlier_ratio_monotonicity(self) -> None:
        settings = MvpRuntimeSettings()
        low_inlier_score = compute_mode_candidate_score(
            confidence=0.15,
            normalized_matches=0.4,
            inlier_ratio=0.1,
            mode="yoloe_vp_lightglue",
            yoloe_confidence_weight=settings.task3_yoloe_score_confidence_weight,
            yoloe_matches_weight=settings.task3_yoloe_score_matches_weight,
            yoloe_inlier_weight=settings.task3_yoloe_score_inlier_weight,
        )
        high_inlier_score = compute_mode_candidate_score(
            confidence=0.15,
            normalized_matches=0.4,
            inlier_ratio=0.5,
            mode="yoloe_vp_lightglue",
            yoloe_confidence_weight=settings.task3_yoloe_score_confidence_weight,
            yoloe_matches_weight=settings.task3_yoloe_score_matches_weight,
            yoloe_inlier_weight=settings.task3_yoloe_score_inlier_weight,
        )

        self.assertGreater(high_inlier_score, low_inlier_score)
        self.assertAlmostEqual(low_inlier_score, 0.2955, places=6)
        self.assertAlmostEqual(high_inlier_score, 0.3515, places=6)

    def test_settings_sanity_keeps_yoloe_weight_sum_at_one(self) -> None:
        settings = MvpRuntimeSettings()
        weight_sum = (
            settings.task3_yoloe_score_confidence_weight
            + settings.task3_yoloe_score_matches_weight
            + settings.task3_yoloe_score_inlier_weight
        )
        self.assertAlmostEqual(weight_sum, 1.0, places=9)

    def test_default_yoloe_scoring_constants_are_calibrated_values(self) -> None:
        settings = MvpRuntimeSettings()
        self.assertAlmostEqual(settings.task3_yoloe_min_score, 0.4520, places=6)
        self.assertAlmostEqual(settings.task3_yoloe_thermal_min_score, 0.50, places=6)
        self.assertAlmostEqual(settings.task3_yoloe_score_confidence_weight, 0.25, places=6)
        self.assertAlmostEqual(settings.task3_yoloe_score_matches_weight, 0.61, places=6)
        self.assertAlmostEqual(settings.task3_yoloe_score_inlier_weight, 0.14, places=6)
        self.assertEqual(settings.task3_yoloe_match_normalization_scale, 50)
        self.assertAlmostEqual(settings.task3_yoloe_homography_ransac_reproj_threshold, 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
