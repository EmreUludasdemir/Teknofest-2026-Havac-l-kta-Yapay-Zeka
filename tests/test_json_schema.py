from __future__ import annotations

import unittest

from src.config.settings import OfficialRepoSettings
from src.core.frame_state import CanonicalDetection, CanonicalUndefinedObject, FrameEnvelope
from src.data.validators import SchemaValidator
from src.server.json_builder import build_protocol_placeholder_result
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter


class JsonSchemaTests(unittest.TestCase):
    def test_wire_includes_task3_and_motion_status_per_2026_spec(self) -> None:
        """2026 spec requires motion_status and detected_undefined_objects in wire."""
        frame = FrameEnvelope(
            frame_url="http://mock/frames/1/",
            image_url="/mock/frame_000000.jpg",
            video_name="mock_session",
            translation_x=1.0,
            translation_y=2.0,
            translation_z=3.0,
            health_status="1",
        )
        result = build_protocol_placeholder_result(frame)

        # Vehicle with motion_status
        result.detected_objects.append(
            CanonicalDetection(
                class_id=0,  # Vehicle
                landing_status=-1,
                motion_status=1,  # Moving
                top_left_x=10.0,
                top_left_y=10.0,
                bottom_right_x=20.0,
                bottom_right_y=20.0,
            )
        )
        # UAP area with landing_status
        result.detected_objects.append(
            CanonicalDetection(
                class_id=2,  # UAP
                landing_status=1,
                motion_status=None,
                top_left_x=30.0,
                top_left_y=30.0,
                bottom_right_x=50.0,
                bottom_right_y=50.0,
            )
        )
        # Task 3 undefined object
        result.detected_undefined_objects.append(
            CanonicalUndefinedObject(
                object_id="ref-001",
                top_left_x=1.0,
                top_left_y=2.0,
                bottom_right_x=3.0,
                bottom_right_y=4.0,
            )
        )

        validator = SchemaValidator()
        canonical_payload = result.to_canonical_dict()
        validator.validate_canonical_result(canonical_payload)

        adapter = OfficialRepoBatchAdapter(
            OfficialRepoSettings(
                base_url="http://mock/",
                username="team",
                password="password",
            )
        )
        wire_payload = adapter.build_wire_prediction(result)

        # 2026 spec: detected_undefined_objects MUST be in wire
        self.assertIn("detected_undefined_objects", wire_payload)
        self.assertEqual(len(wire_payload["detected_undefined_objects"]), 1)
        self.assertEqual(wire_payload["detected_undefined_objects"][0]["object_id"], "ref-001")

        # 2026 spec: motion_status MUST be in wire for vehicles (class_id=0)
        vehicle_obj = wire_payload["detected_objects"][0]
        self.assertIn("motion_status", vehicle_obj)
        self.assertEqual(vehicle_obj["motion_status"], "1")

        # UAP should NOT have motion_status (it's None)
        uap_obj = wire_payload["detected_objects"][1]
        self.assertNotIn("motion_status", uap_obj)

        validator.validate_official_repo_prediction(wire_payload)


if __name__ == "__main__":
    unittest.main()
