from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from src.core.vision import is_cv2_available

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2, np
else:  # pragma: no cover - cv2 yoksa
    cv2 = None
    np = None


@dataclass(slots=True)
class ReferenceCache:
    """Referans nesne metadata ve descriptor cache iskeleti."""

    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    IMAGE_EXTENSIONS: ClassVar[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pgm")
    VALID_DETECTORS: ClassVar[set[str]] = {"yoloe", "orb"}
    VALID_MODALITIES: ClassVar[set[str]] = {"rgb", "thermal", "unknown"}

    def put(self, reference_id: str, payload: dict[str, Any]) -> None:
        self.items[reference_id] = payload

    def get(self, reference_id: str) -> dict[str, Any] | None:
        return self.items.get(reference_id)

    def list_ids(self) -> list[str]:
        return sorted(self.items.keys())

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
        if not allowed_modalities or not modality:
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
            if detector == "orb":
                orb_ids.append(reference_id)
            else:
                yoloe_ids.append(reference_id)
        return yoloe_ids, orb_ids

    def preload_from_directory(self, path: str | Path, *, orb_features: int = 256) -> int:
        directory = Path(path)
        if not directory.exists() or not directory.is_dir():
            return 0

        reference_metadata = self._load_reference_metadata(directory)
        loaded_count = 0
        orb = cv2.ORB_create(nfeatures=orb_features) if is_cv2_available() else None
        for candidate in sorted(directory.iterdir()):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue
            data = candidate.read_bytes()
            reference_id = candidate.stem
            metadata = dict(reference_metadata.get(reference_id, {}))
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
                    "detector": metadata.get("detector", "yoloe"),
                    "detector_modalities": metadata.get("detector_modalities"),
                    "reference_modality": metadata.get("modality"),
                    "reference_dimensions": metadata.get("dimensions"),
                    "reference_metadata": metadata,
                },
            )
            loaded_count += 1
        return loaded_count

    def _load_reference_metadata(self, directory: Path) -> dict[str, dict[str, Any]]:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return {}

        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        spec_path_raw = str(manifest_payload.get("spec_path") or "").strip()
        if not spec_path_raw:
            return {}

        spec_path = Path(spec_path_raw)
        if not spec_path.is_absolute():
            candidate = directory / spec_path_raw
            spec_path = candidate if candidate.exists() else Path(spec_path_raw)
        if not spec_path.exists():
            return {}

        spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
        references = spec_payload.get("references", {})
        if not isinstance(references, dict):
            raise ValueError(f"Task3 reference spec malformed: {spec_path}")

        metadata_by_reference: dict[str, dict[str, Any]] = {}
        for reference_id, payload in references.items():
            if not isinstance(payload, dict):
                raise ValueError(f"Task3 reference spec entry malformed for {reference_id}: {spec_path}")
            detector = str(payload.get("detector", "yoloe"))
            if detector not in self.VALID_DETECTORS:
                raise ValueError(
                    f"Task3 reference spec detector must be one of {sorted(self.VALID_DETECTORS)}, "
                    f"got {detector!r} for {reference_id}"
                )
            detector_modalities = payload.get("detector_modalities")
            if detector_modalities is not None:
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
                detector_modalities = normalized_modalities
            metadata_by_reference[str(reference_id)] = {
                "detector": detector,
                "detector_modalities": detector_modalities,
                "modality": payload.get("modality"),
                "dimensions": payload.get("dimensions"),
                "source_exif": payload.get("source_exif"),
                "file": payload.get("file"),
            }
        return metadata_by_reference
