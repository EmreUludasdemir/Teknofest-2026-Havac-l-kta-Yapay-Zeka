from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from src.core.vision import is_cv2_available
from src.task3.routing_policy import assign_detector

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2, np
else:  # pragma: no cover - cv2 yoksa
    cv2 = None
    np = None

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ReferenceCache:
    """Referans nesne metadata ve descriptor cache iskeleti."""

    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    auto_routing_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    overrides_applied: list[dict[str, Any]] = field(default_factory=list)
    per_reference_suppression: bool = False
    IMAGE_EXTENSIONS: ClassVar[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pgm")
    VALID_DETECTORS: ClassVar[set[str]] = {"yoloe", "orb", "both"}
    VALID_MODALITIES: ClassVar[set[str]] = {"rgb", "thermal", "unknown"}
    VALID_OVERRIDE_FIELDS: ClassVar[set[str]] = {"detector", "detector_modalities", "modality", "rationale"}

    def put(self, reference_id: str, payload: dict[str, Any]) -> None:
        self.items[reference_id] = payload

    def get(self, reference_id: str) -> dict[str, Any] | None:
        return self.items.get(reference_id)

    def list_ids(self) -> list[str]:
        return sorted(self.items.keys())

    def get_auto_routing_summary(self) -> dict[str, dict[str, Any]]:
        return {reference_id: dict(payload) for reference_id, payload in self.auto_routing_summary.items()}

    def get_routing_diagnostics(self) -> dict[str, dict[str, Any]]:
        diagnostics: dict[str, dict[str, Any]] = {}
        for reference_id in self.list_ids():
            item = self.get(reference_id) or {}
            metadata = dict(item.get("reference_metadata") or {})
            diagnostics[reference_id] = {
                "modality": str(item.get("reference_modality") or metadata.get("modality") or "unknown"),
                "detector": str(item.get("detector") or metadata.get("detector") or "yoloe"),
                "detector_modalities": list(item.get("detector_modalities") or metadata.get("detector_modalities") or []),
                "confidence": str(metadata.get("routing_confidence") or "low"),
                "rationale": str(metadata.get("routing_rationale") or ""),
                "signals": dict(metadata.get("routing_signals") or {}),
                "override": dict(metadata.get("override") or {}) if metadata.get("override") else None,
            }
        return diagnostics

    def get_overrides_applied(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.overrides_applied]

    def get_candidate_suppression_mode(self) -> str:
        return "per_reference_top_1" if self.per_reference_suppression else "global_top_1"

    def set_learned_embedding(self, reference_id: str, backbone_name: str, embedding: Any) -> None:
        if reference_id not in self.items:
            return
        self.items[reference_id].setdefault("learned_embeddings", {})
        self.items[reference_id]["learned_embeddings"][backbone_name] = embedding

    def get_learned_embedding(self, reference_id: str, backbone_name: str) -> Any | None:
        item = self.get(reference_id)
        if not item:
            return None
        return item.get("learned_embeddings", {}).get(backbone_name)

    def get_detector(self, reference_id: str, *, default: str = "yoloe") -> str:
        item = self.get(reference_id) or {}
        detector = str(item.get("detector") or default)
        return detector

    def get_detector_modalities(self, reference_id: str) -> tuple[str, ...] | None:
        item = self.get(reference_id) or {}
        raw_modalities = item.get("detector_modalities")
        if not raw_modalities:
            return None
        return tuple(str(modality) for modality in raw_modalities)

    def allows_detector_for_modality(self, reference_id: str, modality: str | None) -> bool:
        allowed_modalities = self.get_detector_modalities(reference_id)
        if not allowed_modalities or not modality or str(modality) == "unknown":
            return True
        return str(modality) in allowed_modalities

    def filter_reference_ids_by_detector_modality(
        self,
        reference_ids: list[str],
        *,
        modality: str | None,
    ) -> list[str]:
        return [reference_id for reference_id in reference_ids if self.allows_detector_for_modality(reference_id, modality)]

    def split_reference_ids_by_detector(
        self,
        reference_ids: list[str],
        *,
        default_detector: str = "yoloe",
    ) -> tuple[list[str], list[str]]:
        yoloe_ids: list[str] = []
        orb_ids: list[str] = []
        for reference_id in reference_ids:
            detector = self.get_detector(reference_id, default=default_detector)
            if detector == "both":
                yoloe_ids.append(reference_id)
                orb_ids.append(reference_id)
            elif detector == "orb":
                orb_ids.append(reference_id)
            else:
                yoloe_ids.append(reference_id)
        return yoloe_ids, orb_ids

    def preload_from_directory(self, path: str | Path, *, orb_features: int = 256) -> int:
        directory = Path(path)
        if not directory.exists() or not directory.is_dir():
            return 0

        self.items.clear()
        self.auto_routing_summary.clear()
        self.overrides_applied.clear()
        self.per_reference_suppression = False

        spec_payload = self._load_reference_spec(directory)
        reference_metadata = spec_payload["references"]
        overrides = spec_payload["overrides"]
        auto_routing_enabled = bool(spec_payload.get("auto_routing_enabled"))
        self.per_reference_suppression = bool(spec_payload.get("per_reference_suppression", False))
        file_to_reference_id = {
            str(metadata.get("file", "")).lower(): reference_id
            for reference_id, metadata in reference_metadata.items()
            if str(metadata.get("file", "")).strip()
        }
        loaded_count = 0
        orb = cv2.ORB_create(nfeatures=orb_features) if is_cv2_available() else None
        for candidate in sorted(directory.iterdir()):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue
            data = candidate.read_bytes()
            reference_id = file_to_reference_id.get(candidate.name.lower(), candidate.stem)
            metadata = dict(reference_metadata.get(reference_id, {}))
            try:
                if auto_routing_enabled:
                    assignment = assign_detector(candidate)
                    effective_detector = assignment.detector
                    effective_modalities = list(assignment.detector_modalities)
                    effective_modality = str(assignment.signals.get("modality", metadata.get("modality") or "unknown"))
                    effective_confidence = str(assignment.confidence)
                    effective_rationale = str(assignment.rationale)
                    routing_signals = dict(assignment.signals)
                else:
                    effective_detector = "yoloe"
                    effective_modalities = None
                    effective_modality = str(metadata.get("modality") or "unknown")
                    effective_confidence = "low"
                    effective_rationale = "specless preload; auto-routing skipped and detector defaulted to yoloe"
                    routing_signals = {
                        "modality": effective_modality,
                        "modality_method": "none",
                        "modality_confidence": "low",
                        "modality_reason": "manifest/spec unavailable",
                        "modality_exif_signals": [],
                        "modality_pixel_signals": {},
                    }
            except Exception as exc:
                effective_detector = "yoloe"
                effective_modalities = None
                effective_modality = "unknown"
                effective_confidence = "low"
                effective_rationale = f"auto-routing unavailable; defaulted to yoloe ({exc})"
                routing_signals = {
                    "modality": "unknown",
                    "modality_method": "none",
                    "modality_confidence": "low",
                    "modality_reason": str(exc),
                    "modality_exif_signals": [],
                    "modality_pixel_signals": {},
                }
                LOGGER.warning("reference %s auto-routing failed: %s", reference_id, exc)
            manual_override = overrides.get(reference_id)
            override_summary: dict[str, Any] | None = None
            if manual_override is not None:
                auto_assignment_summary = {
                    "detector": effective_detector,
                    "detector_modalities": list(effective_modalities) if effective_modalities is not None else None,
                    "modality": effective_modality,
                }
                effective_detector = str(manual_override["detector"])
                if "detector_modalities" in manual_override:
                    effective_modalities = list(manual_override["detector_modalities"])
                if manual_override.get("modality") is not None:
                    effective_modality = str(manual_override["modality"])
                effective_confidence = "medium" if effective_confidence == "high" else effective_confidence
                override_rationale = str(manual_override.get("rationale") or "manual override")
                effective_rationale = f"{override_rationale}"
                override_summary = {
                    "reference_id": reference_id,
                    "auto_detected": auto_assignment_summary,
                    "override": {
                        "detector": effective_detector,
                        "detector_modalities": list(effective_modalities) if effective_modalities is not None else None,
                        "modality": effective_modality,
                        "rationale": override_rationale,
                    },
                }
                self.overrides_applied.append(override_summary)
                LOGGER.warning(
                    "reference %s has manual override; auto-detected was %s, overridden to %s (%s)",
                    reference_id,
                    auto_assignment_summary,
                    override_summary["override"],
                    override_rationale,
                )

            auto_summary_payload = {
                "modality": effective_modality,
                "detector": effective_detector,
                "detector_modalities": list(effective_modalities) if effective_modalities is not None else None,
                "confidence": effective_confidence,
                "rationale": effective_rationale,
            }
            self.auto_routing_summary[reference_id] = auto_summary_payload
            gray = None
            bgr = None
            descriptors = None
            keypoints = None
            width = 0
            height = 0
            descriptor_mode = "metadata_only"
            if orb is not None:
                bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    height, width = gray.shape[:2]
                    keypoints, descriptors = orb.detectAndCompute(gray, None)
                    descriptor_mode = "orb"
            self.put(
                reference_id,
                {
                    "reference_id": reference_id,
                    "path": str(candidate),
                    "byte_size": len(data),
                    "sha1": hashlib.sha1(data).hexdigest(),
                    "loaded_from": "directory_preload",
                    "bgr": bgr,
                    "gray": gray,
                    "width": width,
                    "height": height,
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                    "descriptor_mode": descriptor_mode,
                    "keypoint_count": len(keypoints or []),
                    "learned_embeddings": {},
                    "detector": effective_detector,
                    "detector_modalities": list(effective_modalities) if effective_modalities is not None else None,
                    "reference_modality": effective_modality,
                    "reference_dimensions": metadata.get("dimensions"),
                    "reference_metadata": {
                        **metadata,
                        "modality": effective_modality,
                        "detector": effective_detector,
                        "detector_modalities": list(effective_modalities) if effective_modalities is not None else None,
                        "routing_confidence": effective_confidence,
                        "routing_rationale": effective_rationale,
                        "routing_signals": routing_signals,
                        "override": override_summary["override"] if override_summary else None,
                    },
                },
            )
            loaded_count += 1
        return loaded_count

    def _load_reference_spec(self, directory: Path) -> dict[str, dict[str, Any]]:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return {"references": {}, "overrides": {}, "auto_routing_enabled": False, "per_reference_suppression": False}

        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        spec_path_raw = str(manifest_payload.get("spec_path") or "").strip()
        if not spec_path_raw:
            return {"references": {}, "overrides": {}, "auto_routing_enabled": False, "per_reference_suppression": False}

        spec_path = Path(spec_path_raw)
        if not spec_path.is_absolute():
            candidate = directory / spec_path_raw
            spec_path = candidate if candidate.exists() else Path(spec_path_raw)
        if not spec_path.exists():
            return {"references": {}, "overrides": {}, "auto_routing_enabled": False, "per_reference_suppression": False}

        spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
        references = spec_payload.get("references", {})
        if not isinstance(references, dict):
            raise ValueError(f"Task3 reference spec malformed: {spec_path}")
        per_reference_suppression = spec_payload.get("per_reference_suppression", False)
        if not isinstance(per_reference_suppression, bool):
            raise ValueError(f"Task3 reference spec per_reference_suppression must be boolean: {spec_path}")
        overrides_payload = spec_payload.get("overrides", {})
        if overrides_payload is None:
            overrides_payload = {}
        if not isinstance(overrides_payload, dict):
            raise ValueError(f"Task3 reference spec overrides malformed: {spec_path}")

        metadata_by_reference: dict[str, dict[str, Any]] = {}
        nested_overrides: dict[str, dict[str, Any]] = {}
        for reference_id, payload in references.items():
            if not isinstance(payload, dict):
                raise ValueError(f"Task3 reference spec entry malformed for {reference_id}: {spec_path}")
            reference_id_text = str(reference_id)
            routing_override = payload.get("routing_override")
            if routing_override is not None:
                nested_overrides[reference_id_text] = self._normalize_override_payload(
                    routing_override,
                    reference_id_text,
                    spec_path=spec_path,
                )
            metadata_by_reference[reference_id_text] = {
                "modality": payload.get("modality"),
                "dimensions": payload.get("dimensions"),
                "source_exif": payload.get("source_exif"),
                "file": payload.get("file"),
            }

        normalized_overrides: dict[str, dict[str, Any]] = {}
        for reference_id, payload in overrides_payload.items():
            reference_id_text = str(reference_id)
            normalized_overrides[reference_id_text] = self._normalize_override_payload(
                payload,
                reference_id_text,
                spec_path=spec_path,
            )
        for reference_id, payload in nested_overrides.items():
            if reference_id in normalized_overrides:
                raise ValueError(
                    f"Task3 reference spec override duplicated for {reference_id}: "
                    f"use either top-level overrides or references.{reference_id}.routing_override"
                )
            normalized_overrides[reference_id] = payload

        return {
            "references": metadata_by_reference,
            "overrides": normalized_overrides,
            "auto_routing_enabled": True,
            "per_reference_suppression": per_reference_suppression,
        }

    def _normalize_override_payload(
        self,
        payload: Any,
        reference_id: str,
        *,
        spec_path: Path,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(f"Task3 reference spec override malformed for {reference_id}: {spec_path}")
        unexpected_keys = sorted(set(payload.keys()) - self.VALID_OVERRIDE_FIELDS)
        if unexpected_keys:
            raise ValueError(
                f"Task3 reference spec override contains unsupported fields for {reference_id}: {unexpected_keys}"
            )
        if "detector" not in payload:
            raise ValueError(f"Task3 reference spec override missing detector for {reference_id}: {spec_path}")
        detector = self._validate_detector(str(payload["detector"]), reference_id)
        modality = payload.get("modality")
        if modality is not None and str(modality) not in self.VALID_MODALITIES:
            raise ValueError(
                f"Task3 reference spec override modality must be within {sorted(self.VALID_MODALITIES)}, "
                f"got {modality!r} for {reference_id}"
            )
        detector_modalities = self._normalize_detector_modalities(payload.get("detector_modalities"), reference_id)
        if detector_modalities is None and modality is not None and str(modality) in {"rgb", "thermal"}:
            detector_modalities = [str(modality)]
        return {
            "detector": detector,
            "detector_modalities": detector_modalities,
            "modality": str(modality) if modality is not None else None,
            "rationale": payload.get("rationale"),
        }

    def _validate_detector(self, detector: str, reference_id: str) -> str:
        if detector not in self.VALID_DETECTORS:
            raise ValueError(
                f"Task3 reference spec detector must be one of {sorted(self.VALID_DETECTORS)}, "
                f"got {detector!r} for {reference_id}"
            )
        return detector

    def _normalize_detector_modalities(self, detector_modalities: Any, reference_id: str) -> list[str] | None:
        if detector_modalities is None:
            return None
        if not isinstance(detector_modalities, list) or not detector_modalities:
            raise ValueError(f"Task3 reference spec detector_modalities must be a non-empty list for {reference_id}")
        normalized_modalities: list[str] = []
        for modality in detector_modalities:
            modality_text = str(modality)
            if modality_text not in self.VALID_MODALITIES:
                raise ValueError(
                    f"Task3 reference spec detector_modalities must be within {sorted(self.VALID_MODALITIES)}, "
                    f"got {modality_text!r} for {reference_id}"
                )
            if modality_text not in normalized_modalities:
                normalized_modalities.append(modality_text)
        return normalized_modalities
