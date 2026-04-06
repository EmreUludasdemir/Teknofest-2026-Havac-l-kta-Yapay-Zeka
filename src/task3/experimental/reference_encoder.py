from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.vision import is_cv2_available
from src.task3.reference_cache import ReferenceCache

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2
else:  # pragma: no cover - cv2 yoksa
    cv2 = None


@dataclass(slots=True)
class ExperimentalReferenceEncoder:
    """Referans nesneler icin yeniden kullanilabilir deneysel durum hazirlar."""

    runtime_settings: MvpRuntimeSettings

    def prepare(self, reference_cache: ReferenceCache) -> dict[str, dict[str, Any]]:
        prepared: dict[str, dict[str, Any]] = {}
        model_path = self.runtime_settings.task3_experimental_model_path
        yoloe_weight_exists = bool(model_path and Path(model_path).exists())
        for reference_id in reference_cache.list_ids():
            item = reference_cache.get(reference_id) or {}
            refer_bgr = item.get("bgr")
            if refer_bgr is None and item.get("gray") is not None and cv2 is not None:
                refer_bgr = cv2.cvtColor(item["gray"], cv2.COLOR_GRAY2BGR)
            width = max(int(item.get("width", 0)), 1)
            height = max(int(item.get("height", 0)), 1)
            modality = str(item.get("modality", "rgb"))
            capability = {
                "reference_id": reference_id,
                "visual_prompt": {
                    "bboxes": [[0.0, 0.0, float(width - 1), float(height - 1)]],
                    "cls": [0],
                },
                "refer_image": refer_bgr,
                "yoloe_ready": refer_bgr is not None and yoloe_weight_exists,
                "orb_ready": item.get("descriptors") is not None,
                "lightglue_ready": False,
                "cross_sensor_risky": modality == "thermal",
                "reference_modality": modality,
                "reference_path": str(item.get("path", "")),
            }
            item["experimental"] = capability
            prepared[reference_id] = capability
        return prepared
