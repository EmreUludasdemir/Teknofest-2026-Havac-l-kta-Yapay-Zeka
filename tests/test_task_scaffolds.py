from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalTranslation, CanonicalUndefinedObject, DecodedFrame, FrameEnvelope
from src.task1.detector import Task1Detector
from src.task1.landing_logic import assign_landing_status
from src.task1.motion_logic import assign_motion_status
from src.task1.postprocess import deduplicate_detections
from src.task1.tracker import Task1Tracker
from src.task2.calibration import build_calibration_state
from src.task2.drift_control import apply_drift_guard
from src.task2.estimator import Task2Estimator
from src.task2.health_logic import should_use_reference_translation
from src.task2.masking import mask_dynamic_objects
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches


class TaskScaffoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = FrameEnvelope(
            frame_url="http://mock/frames/1/",
            image_url="/frame.jpg",
            video_name="session",
            translation_x=1.0,
            translation_y=2.0,
            translation_z=3.0,
            health_status="1",
        )
        self.image_bytes = b"placeholder-image"
        self.decoded = DecodedFrame(
            bgr=None,
            gray=None,
            width=640,
            height=512,
            channel_count=0,
            modality="thermal",
            frame_index=1,
        )

    def test_task1_scaffolds_are_import_safe(self) -> None:
        detector = Task1Detector(runtime_settings=MvpRuntimeSettings())
        tracker = Task1Tracker()
        detections = detector.detect(self.frame, self.image_bytes)
        self.assertIsInstance(detections, list)
        self.assertEqual(tracker.update(self.frame, detections), detections)
        self.assertEqual(assign_motion_status(detections, self.frame), detections)
        self.assertEqual(assign_landing_status(detections, self.frame), detections)
        self.assertEqual(deduplicate_detections(detections), detections)

    def test_task2_scaffolds_return_expected_types(self) -> None:
        estimator = Task2Estimator(runtime_settings=MvpRuntimeSettings())
        translation, _ = estimator.estimate(self.frame, self.decoded)
        self.assertIsInstance(translation, CanonicalTranslation)
        self.assertTrue(should_use_reference_translation(self.frame))
        self.assertEqual(mask_dynamic_objects(self.decoded, [])[0], None)
        self.assertIsInstance(build_calibration_state([self.frame]), dict)
        guarded, _ = apply_drift_guard(translation, None, None, confidence=0.9)
        self.assertEqual(guarded.translation_x, translation.translation_x)
        self.assertEqual(guarded.translation_y, translation.translation_y)
        self.assertEqual(guarded.translation_z, translation.translation_z)

    def test_task3_scaffolds_return_expected_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            (temp_path / "ref-001.jpg").write_bytes(b"ref-data")
            cache = ReferenceCache()
            cache.preload_from_directory(temp_path)
            matcher = Task3Matcher(reference_cache=cache, runtime_settings=MvpRuntimeSettings(task3_reference_dir=temp_path))
            matches = matcher.match(self.frame, self.image_bytes, ["ref-001"])
        self.assertIsInstance(matches, list)
        self.assertEqual(verify_matches(self.frame, matches), matches)
        self.assertEqual(filter_no_match_candidates(matches), matches)
        cache.put("ref-001", {"source": "test"})
        self.assertEqual(cache.get("ref-001"), {"source": "test"})
        self.assertTrue(all(isinstance(item, CanonicalUndefinedObject) for item in matches))


if __name__ == "__main__":
    unittest.main()
