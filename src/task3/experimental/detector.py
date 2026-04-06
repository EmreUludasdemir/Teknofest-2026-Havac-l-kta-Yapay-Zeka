from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame, FrameEnvelope
from src.task3.experimental.reference_encoder import ExperimentalReferenceEncoder
from src.task3.experimental.tiled import ExperimentalTiledInference
from src.task3.matcher import Task3Matcher
from src.task3.reference_cache import ReferenceCache

try:  # pragma: no cover - ortama bagli
    from ultralytics import YOLOE  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - ortama bagli
    YOLOE = None

try:  # pragma: no cover - ortama bagli
    import torch  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - ortama bagli
    torch = None


@dataclass(slots=True)
class ExperimentalPromptDetector:
    runtime_settings: MvpRuntimeSettings
    reference_cache: ReferenceCache
    baseline_matcher: Task3Matcher
    encoder: ExperimentalReferenceEncoder = field(init=False)
    tiled_inference: ExperimentalTiledInference = field(init=False)
    reference_states: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    integration_status: str = field(init=False, default="uninitialized")
    model: Any | None = field(init=False, default=None)
    model_error: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.encoder = ExperimentalReferenceEncoder(runtime_settings=self.runtime_settings)
        self.tiled_inference = ExperimentalTiledInference(
            enabled=self.runtime_settings.task3_experimental_tiled_inference,
        )

    def ensure_reference_states(self) -> dict[str, dict[str, Any]]:
        if not self.reference_states:
            self.reference_states = self.encoder.prepare(self.reference_cache)
        return self.reference_states

    def search(
        self,
        frame: FrameEnvelope,
        decoded_frame: DecodedFrame,
        reference_ids: list[str],
    ) -> tuple[list[CanonicalUndefinedObject], dict[str, Any]]:
        states = self.ensure_reference_states()
        if not reference_ids:
            return [], {"integration_status": self.integration_status, "searched_reference_ids": []}
        requested_mode = self.runtime_settings.task3_experimental_mode
        if requested_mode != "yoloe_prompted" and self.integration_status == "uninitialized":
            self.integration_status = "prompt_approx"
        true_yoloe = self._ensure_yoloe_model() if requested_mode == "yoloe_prompted" else False
        candidates: list[CanonicalUndefinedObject] = []
        for reference_id in reference_ids:
            reference_state = states.get(reference_id)
            if not reference_state:
                continue
            if true_yoloe:
                candidates.extend(self._search_with_yoloe(decoded_frame, reference_id, reference_state))
            else:
                candidates.extend(self._search_with_prompt_approx(decoded_frame, reference_id))
        diagnostics = {
            "requested_mode": requested_mode,
            "active_detector_mode": "true_yoloe" if true_yoloe else "prompt_approx",
            "integration_status": self.integration_status,
            "searched_reference_ids": list(reference_ids),
            "tiled_status": self.tiled_inference.status(),
            "model_error": self.model_error,
            "candidate_count": len(candidates),
        }
        return candidates, diagnostics

    def _ensure_yoloe_model(self) -> bool:
        if self.integration_status == "true_yoloe" and self.model is not None:
            return True
        if self.integration_status in {"yoloe_weight_missing", "prompt_approx", "yoloe_load_failed"}:
            return False
        if YOLOE is None:
            self.integration_status = "prompt_approx"
            self.model_error = "missing_dependency:ultralytics_yoloe"
            return False
        model_path = self.runtime_settings.task3_experimental_model_path
        if model_path is None or not Path(model_path).exists():
            self.integration_status = "yoloe_weight_missing"
            self.model_error = f"missing_yoloe_weights:{model_path}"
            return False
        try:
            self.model = YOLOE(str(model_path), task="detect")
            if torch is not None and bool(getattr(torch.cuda, "is_available", lambda: False)()):
                self.model.to("cuda:0")
            self.integration_status = "true_yoloe"
            return True
        except Exception as exc:
            self.integration_status = "yoloe_load_failed"
            self.model_error = str(exc)
            self.model = None
            return False

    def _search_with_prompt_approx(
        self,
        decoded_frame: DecodedFrame,
        reference_id: str,
    ) -> list[CanonicalUndefinedObject]:
        proposals = self.baseline_matcher.collect_proposals(decoded_frame, [reference_id])
        threshold = self.runtime_settings.task3_experimental_prompt_confidence
        selected: list[CanonicalUndefinedObject] = []
        for proposal in proposals:
            score = float(proposal.metadata.get("match_score", 0.0))
            if score < threshold:
                continue
            proposal.metadata.update(
                {
                    "matcher_source": "task3_prompt_approx",
                    "candidate_name": "prompt_approx",
                    "detection_score": round(score, 4),
                    "integration_status": "prompt_approx",
                }
            )
            selected.append(proposal)
        return selected[:2]

    def _search_with_yoloe(
        self,
        decoded_frame: DecodedFrame,
        reference_id: str,
        reference_state: dict[str, Any],
    ) -> list[CanonicalUndefinedObject]:
        if self.model is None or decoded_frame.bgr is None:
            return []
        device_name = "cuda:0" if torch is not None and bool(getattr(torch.cuda, "is_available", lambda: False)()) else "cpu"
        results = self.model.predict(
            source=decoded_frame.bgr,
            stream=False,
            verbose=False,
            imgsz=self.runtime_settings.task3_experimental_tile_size,
            conf=self.runtime_settings.task3_experimental_detector_confidence,
            visual_prompts=reference_state["visual_prompt"],
            refer_image=reference_state["refer_image"],
            device=device_name,
        )
        if not results:
            return []
        prediction = results[0]
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            return []
        xyxy_values = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else []
        conf_values = boxes.conf.tolist() if hasattr(boxes, "conf") and hasattr(boxes.conf, "tolist") else []
        candidates: list[CanonicalUndefinedObject] = []
        for index, coords in enumerate(xyxy_values[:2]):
            if len(coords) != 4:
                continue
            score = float(conf_values[index]) if index < len(conf_values) else 0.0
            if score < self.runtime_settings.task3_experimental_detector_confidence:
                continue
            candidates.append(
                CanonicalUndefinedObject(
                    object_id=reference_id,
                    top_left_x=float(coords[0]),
                    top_left_y=float(coords[1]),
                    bottom_right_x=float(coords[2]),
                    bottom_right_y=float(coords[3]),
                    metadata={
                        "matcher_source": "task3_yoloe_prompted",
                        "candidate_name": "yoloe_prompted",
                        "detection_score": round(score, 4),
                        "integration_status": "true_yoloe",
                    },
                )
            )
        return candidates
