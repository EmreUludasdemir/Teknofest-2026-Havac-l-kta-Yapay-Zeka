from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    """Schema dogrulama hatasi."""


class SchemaValidator:
    def validate_canonical_result(self, payload: dict[str, Any]) -> None:
        self._require_keys(payload, ("frame", "detected_objects", "detected_translations", "detected_undefined_objects", "diagnostics"))
        if not isinstance(payload["frame"], str):
            raise SchemaValidationError("Canonical frame alani string olmali.")
        self._ensure_list(payload["detected_objects"], "detected_objects")
        self._ensure_list(payload["detected_translations"], "detected_translations")
        self._ensure_list(payload["detected_undefined_objects"], "detected_undefined_objects")
        if not isinstance(payload["diagnostics"], dict):
            raise SchemaValidationError("Canonical diagnostics dict olmali.")

        for item in payload["detected_objects"]:
            self._require_keys(
                item,
                ("class_id", "landing_status", "motion_status", "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y", "metadata"),
            )
            if not isinstance(item["class_id"], int):
                raise SchemaValidationError("Canonical class_id int olmali.")
            self._ensure_number(item["top_left_x"], "top_left_x")
            self._ensure_number(item["top_left_y"], "top_left_y")
            self._ensure_number(item["bottom_right_x"], "bottom_right_x")
            self._ensure_number(item["bottom_right_y"], "bottom_right_y")

        for item in payload["detected_translations"]:
            self._require_keys(item, ("translation_x", "translation_y", "translation_z", "source"))
            self._ensure_number(item["translation_x"], "translation_x")
            self._ensure_number(item["translation_y"], "translation_y")
            self._ensure_number(item["translation_z"], "translation_z")
            if not isinstance(item["source"], str):
                raise SchemaValidationError("Canonical source string olmali.")

        for item in payload["detected_undefined_objects"]:
            self._require_keys(item, ("object_id", "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y", "metadata"))
            if not isinstance(item["object_id"], str):
                raise SchemaValidationError("Canonical object_id string olmali.")
            self._ensure_number(item["top_left_x"], "undefined.top_left_x")
            self._ensure_number(item["top_left_y"], "undefined.top_left_y")
            self._ensure_number(item["bottom_right_x"], "undefined.bottom_right_x")
            self._ensure_number(item["bottom_right_y"], "undefined.bottom_right_y")

    def validate_official_repo_prediction(self, payload: dict[str, Any]) -> None:
        """Validate official 2025 batch wire format."""
        self._require_keys(payload, ("frame", "detected_objects", "detected_translations"))
        unexpected = set(payload.keys()) - {"frame", "detected_objects", "detected_translations"}
        if unexpected:
            raise SchemaValidationError(f"Official batch wire beklenmeyen alanlar iceriyor: {sorted(unexpected)}")
        if not isinstance(payload["frame"], str):
            raise SchemaValidationError("Official frame alani string olmali.")

        self._ensure_list(payload["detected_objects"], "detected_objects")
        self._ensure_list(payload["detected_translations"], "detected_translations")

        for item in payload["detected_objects"]:
            self._require_keys(item, ("cls", "landing_status", "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"))
            if not isinstance(item["cls"], str) or "/classes/" not in item["cls"]:
                raise SchemaValidationError("Official cls alani classes resource URL olmali.")
            for key in ("landing_status", "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"):
                if not isinstance(item[key], str):
                    raise SchemaValidationError(f"Official {key} string olmali.")
            if "motion_status" in item:
                raise SchemaValidationError("Official batch wire 2025 motion_status icermemeli.")

        for item in payload["detected_translations"]:
            self._require_keys(item, ("translation_x", "translation_y", "translation_z"))
            for key in ("translation_x", "translation_y", "translation_z"):
                if not isinstance(item[key], str):
                    raise SchemaValidationError(f"Official {key} string olmali.")

    def validate_sequential_prediction(
        self,
        payload: dict[str, Any],
        *,
        profile: str = "teknofest_2026",
    ) -> None:
        """Validate sequential wire format per 2026 spec."""
        self._require_keys(payload, ("session_id", "frame_id", "frame", "detected_objects", "detected_translations", "detected_undefined_objects"))
        if not isinstance(payload["session_id"], str) or not payload["session_id"]:
            raise SchemaValidationError("Sequential session_id string olmali.")
        if not isinstance(payload["frame_id"], (int, str)):
            raise SchemaValidationError("Sequential frame_id int veya string olmali.")
        
        # Validate common fields via official repo validator
        self.validate_official_repo_prediction(
            {
                "frame": payload["frame"],
                "detected_objects": payload["detected_objects"],
                "detected_translations": payload["detected_translations"],
            }
        )
        self._ensure_list(payload["detected_undefined_objects"], "detected_undefined_objects")
        for item in payload["detected_undefined_objects"]:
            self._require_keys(item, ("object_id", "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"))
            if not isinstance(item["object_id"], str):
                raise SchemaValidationError("Sequential undefined object_id string olmali.")
            for key in ("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"):
                if not isinstance(item[key], str):
                    raise SchemaValidationError(f"Sequential undefined {key} string olmali.")

    @staticmethod
    def _require_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> None:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise SchemaValidationError(f"Eksik alanlar: {missing}")

    @staticmethod
    def _ensure_list(value: Any, field_name: str) -> None:
        if not isinstance(value, list):
            raise SchemaValidationError(f"{field_name} list olmali.")

    @staticmethod
    def _ensure_number(value: Any, field_name: str) -> None:
        if not isinstance(value, (int, float)):
            raise SchemaValidationError(f"{field_name} sayisal olmali.")
