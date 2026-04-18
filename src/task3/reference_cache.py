from __future__ import annotations

import hashlib
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

    def preload_from_directory(self, path: str | Path, *, orb_features: int = 256) -> int:
        directory = Path(path)
        if not directory.exists() or not directory.is_dir():
            return 0

        loaded_count = 0
        orb = cv2.ORB_create(nfeatures=orb_features) if is_cv2_available() else None
        for candidate in sorted(directory.iterdir()):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue
            data = candidate.read_bytes()
            reference_id = candidate.stem
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
                },
            )
            loaded_count += 1
        return loaded_count
