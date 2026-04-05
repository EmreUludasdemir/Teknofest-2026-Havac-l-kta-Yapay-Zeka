from __future__ import annotations

import unittest

from src.config.settings import OfficialRepoSettings
from src.core.frame_state import CanonicalDetection, CanonicalUndefinedObject, FrameEnvelope
from src.data.validators import SchemaValidator
from src.server.json_builder import build_protocol_placeholder_result
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter


class JsonSchemaTests(unittest.TestCase):
    def test_canonical_model_keeps_task3_but_official_wire_omits_it(self) -> None:
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
        result.detected_objects.append(
            CanonicalDetection(
                class_id=2,
                landing_status=1,
                motion_status=-1,
                top_left_x=10.0,
                top_left_y=10.0,
                bottom_right_x=20.0,
                bottom_right_y=20.0,
            )
        )
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
        self.assertNotIn("detected_undefined_objects", wire_payload)
        self.assertNotIn("motion_status", wire_payload["detected_objects"][0])
        validator.validate_official_repo_prediction(wire_payload)


if __name__ == "__main__":
    unittest.main()
