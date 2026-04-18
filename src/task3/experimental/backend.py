from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame
from src.core.vision import is_cv2_available
from src.task3.reference_cache import ReferenceCache

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2, np
else:  # pragma: no cover - cv2 yoksa
    cv2 = None
    np = None


class Task3ExperimentalUnavailableError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class YoloeReferenceBank:
    reference_cache: ReferenceCache
    model: Any
    extractor: Any
    predictor_cls: Any
    device: str
    runtime_settings: MvpRuntimeSettings
    ref_names: list[str] | None = None
    sp_features: list[dict[str, Any]] | None = None

    def build(self, reference_ids: list[str]) -> None:
        import torch  # type: ignore[import-not-found]
        from lightglue.utils import numpy_image_to_torch

        ref_names: list[str] = []
        vpes = []
        sp_features: list[dict[str, Any]] = []
        for reference_id in reference_ids:
            item = self.reference_cache.get(reference_id)
            if not item:
                continue
            image = item.get("bgr")
            if image is None:
                path = item.get("path")
                if path:
                    image = cv2.imread(str(path))
            if image is None:
                continue

            ref_names.append(reference_id)
            height, width = image.shape[:2]
            prompts = {
                "bboxes": np.array([[0, 0, max(width - 1, 0), max(height - 1, 0)]], dtype=np.float32),
                "cls": np.array([0], dtype=np.int64),
            }
            vpe = self._extract_vpe(image, prompts)
            vpes.append(vpe)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = numpy_image_to_torch(rgb).to(self.device)
            if tensor.dim() == 3:
                tensor = tensor[None]
            features = self.extractor.extract(tensor)
            sp_features.append(features)

        if not ref_names or not vpes:
            raise Task3ExperimentalUnavailableError("no_reference_images")

        self.ref_names = ref_names
        self.sp_features = sp_features
        embeddings = torch.cat(vpes, dim=0)
        self.model.set_classes(self.ref_names, embeddings)

    def _extract_vpe(self, image: Any, prompts: dict[str, Any]) -> Any:
        predictor = self.predictor_cls(
            overrides={
                "task": self.model.model.task,
                "mode": "predict",
                "save": False,
                "verbose": False,
                "batch": 1,
                "device": self.device,
                "half": False,
                "imgsz": self.runtime_settings.task3_yoloe_imgsz,
            },
            _callbacks=self.model.callbacks,
        )
        predictor.set_prompts(prompts.copy())
        predictor.setup_model(model=self.model.model)
        vpe = predictor.get_vpe(image)
        while getattr(vpe, "dim", lambda: 0)() > 2:
            vpe = vpe.squeeze(0)
        if getattr(vpe, "dim", lambda: 0)() == 1:
            vpe = vpe.unsqueeze(0)
        return vpe


@dataclass(slots=True)
class LightGlueCropVerifier:
    extractor: Any
    matcher: Any
    device: str
    runtime_settings: MvpRuntimeSettings

    def _prepare_crop(self, crop_bgr: Any) -> Any | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        height, width = crop_bgr.shape[:2]
        if min(height, width) < self.runtime_settings.task3_min_crop_side:
            return None
        if max(height, width) < self.runtime_settings.task3_resize_crop_to:
            scale = self.runtime_settings.task3_resize_crop_to / max(height, width)
            new_height = int(round(height * scale))
            new_width = int(round(width * scale))
            crop_bgr = cv2.resize(crop_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        from lightglue.utils import numpy_image_to_torch

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = numpy_image_to_torch(rgb).to(self.device)
        if tensor.dim() == 3:
            tensor = tensor[None]
        return tensor

    def verify(self, crop_bgr: Any, ref_features: dict[str, Any]) -> tuple[bool, int, float]:
        from lightglue.utils import rbd

        crop_tensor = self._prepare_crop(crop_bgr)
        if crop_tensor is None:
            return False, 0, 0.0

        start = time.perf_counter()
        try:
            crop_features = self.extractor.extract(crop_tensor)
            result = self.matcher({"image0": ref_features, "image1": crop_features})
            result = rbd(result)
            matches = result.get("matches")
            match_count = int(matches.shape[0]) if matches is not None else 0
        except Exception:
            return False, 0, 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        passed = match_count >= self.runtime_settings.task3_lightglue_min_matches
        return passed, match_count, elapsed_ms


@dataclass(slots=True)
class YoloeVpLightGlueBackend:
    reference_cache: ReferenceCache
    runtime_settings: MvpRuntimeSettings
    model: Any | None = None
    extractor: Any | None = None
    matcher: Any | None = None
    reference_bank: YoloeReferenceBank | None = None
    verifier: LightGlueCropVerifier | None = None
    device: str | None = None

    def match(
        self,
        *,
        decoded_frame: DecodedFrame,
        reference_ids: list[str],
    ) -> tuple[list[CanonicalUndefinedObject], dict[str, Any]]:
        task3_info = {
            "requested_mode": "yoloe_vp_lightglue",
            "effective_mode": "yoloe_vp_lightglue",
            "fallback_reason": None,
            "references_loaded": len(reference_ids),
            "yoloe_inference_ms": 0.0,
            "lightglue_verify_ms_total": 0.0,
            "candidates_generated": 0,
            "candidates_accepted": 0,
            "candidates_rejected": 0,
        }
        self._ensure_ready(reference_ids)

        if decoded_frame.bgr is None:
            raise Task3ExperimentalUnavailableError("missing_bgr_frame")
        if self.reference_bank is None or self.reference_bank.ref_names is None or self.reference_bank.sp_features is None or self.verifier is None:
            raise Task3ExperimentalUnavailableError("reference_bank_not_ready")

        start = time.perf_counter()
        results = self.model.predict(
            decoded_frame.bgr,
            conf=self.runtime_settings.task3_yoloe_conf,
            iou=self.runtime_settings.task3_yoloe_iou,
            imgsz=self.runtime_settings.task3_yoloe_imgsz,
            max_det=max(self.runtime_settings.task3_yoloe_max_det_per_class * max(len(reference_ids), 1), 1),
            verbose=False,
            device=self.device,
        )
        task3_info["yoloe_inference_ms"] = round((time.perf_counter() - start) * 1000.0, 4)
        if not results:
            return [], task3_info

        prediction = results[0]
        boxes = getattr(prediction, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], task3_info

        width = decoded_frame.width
        height = decoded_frame.height
        raw_boxes = boxes.xyxy.cpu().numpy()
        raw_classes = boxes.cls.cpu().numpy().astype(int)
        raw_confidences = boxes.conf.cpu().numpy()
        allowed_ids = set(reference_ids)
        grouped: dict[int, list[int]] = {}
        for index, class_id in enumerate(raw_classes):
            if not 0 <= int(class_id) < len(self.reference_bank.ref_names):
                continue
            ref_name = self.reference_bank.ref_names[int(class_id)]
            if ref_name not in allowed_ids:
                continue
            grouped.setdefault(int(class_id), []).append(index)

        keep_indices: list[int] = []
        for class_id, indices in grouped.items():
            del class_id
            sorted_indices = sorted(indices, key=lambda idx: -float(raw_confidences[idx]))
            keep_indices.extend(sorted_indices[: self.runtime_settings.task3_yoloe_max_det_per_class])

        verify_frame = decoded_frame.frame_index % max(self.runtime_settings.task3_yoloe_verify_every_k, 1) == 0
        task3_info["candidates_generated"] = len(keep_indices)
        accepted: list[CanonicalUndefinedObject] = []
        rejected = 0
        verify_ms_total = 0.0
        for index in keep_indices:
            x1 = max(int(raw_boxes[index, 0]), 0)
            y1 = max(int(raw_boxes[index, 1]), 0)
            x2 = min(int(raw_boxes[index, 2]), width - 1)
            y2 = min(int(raw_boxes[index, 3]), height - 1)
            if x2 <= x1 or y2 <= y1:
                rejected += 1
                continue

            class_id = int(raw_classes[index])
            ref_name = self.reference_bank.ref_names[class_id]
            confidence = float(raw_confidences[index])
            match_count = 0
            verify_passed = True
            if verify_frame:
                crop = decoded_frame.bgr[y1:y2, x1:x2]
                verify_passed, match_count, verify_ms = self.verifier.verify(crop, self.reference_bank.sp_features[class_id])
                verify_ms_total += verify_ms

            if not verify_passed:
                rejected += 1
                continue

            normalized_matches = 1.0 if not verify_frame else min(match_count / max(self.runtime_settings.task3_lightglue_min_matches, 1), 1.0)
            score = (confidence * 0.70) + (normalized_matches * 0.30)
            accepted.append(
                CanonicalUndefinedObject(
                    object_id=ref_name,
                    top_left_x=float(x1),
                    top_left_y=float(y1),
                    bottom_right_x=float(x2),
                    bottom_right_y=float(y2),
                    metadata={
                        "matcher_source": "task3_yoloe_vp_lightglue",
                        "candidate_name": "yoloe_vp_lightglue",
                        "match_score": round(float(score), 4),
                        "task3_yoloe": {
                            "confidence": round(confidence, 4),
                            "match_count": int(match_count),
                            "verify_passed": bool(verify_passed),
                            "verify_skipped": bool(not verify_frame),
                        },
                    },
                )
            )

        task3_info["lightglue_verify_ms_total"] = round(verify_ms_total, 4)
        task3_info["candidates_accepted"] = len(accepted)
        task3_info["candidates_rejected"] = rejected
        return accepted, task3_info

    def _ensure_ready(self, reference_ids: list[str]) -> None:
        if not is_cv2_available():
            raise Task3ExperimentalUnavailableError("opencv_unavailable")
        weight_path = Path(self.runtime_settings.task3_yoloe_weight_path)
        if not weight_path.exists():
            raise Task3ExperimentalUnavailableError("missing_yoloe_weight")
        if importlib.util.find_spec("ultralytics") is None:
            raise Task3ExperimentalUnavailableError("missing_ultralytics")
        if importlib.util.find_spec("torch") is None:
            raise Task3ExperimentalUnavailableError("missing_torch")
        if importlib.util.find_spec("lightglue") is None:
            raise Task3ExperimentalUnavailableError("missing_lightglue")

        import torch  # type: ignore[import-not-found]
        from lightglue import LightGlue, SuperPoint  # type: ignore[import-not-found]
        from ultralytics import YOLOE  # type: ignore[import-not-found]
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor  # type: ignore[import-not-found]

        resolved_device = self._resolve_device(torch)
        if self.model is None:
            self.model = YOLOE(str(weight_path))
        if self.extractor is None:
            self.extractor = SuperPoint(max_num_keypoints=self.runtime_settings.task3_superpoint_max_kpts).eval().to(resolved_device)
        if self.matcher is None:
            self.matcher = LightGlue(features="superpoint").eval().to(resolved_device)
        if self.reference_bank is None:
            self.reference_bank = YoloeReferenceBank(
                reference_cache=self.reference_cache,
                model=self.model,
                extractor=self.extractor,
                predictor_cls=YOLOEVPSegPredictor,
                device=resolved_device,
                runtime_settings=self.runtime_settings,
            )
            self.reference_bank.build(self.reference_cache.list_ids() or reference_ids)
        if self.verifier is None:
            self.verifier = LightGlueCropVerifier(
                extractor=self.extractor,
                matcher=self.matcher,
                device=resolved_device,
                runtime_settings=self.runtime_settings,
            )
        self.device = resolved_device

    def _resolve_device(self, torch_module: Any) -> str:
        configured = self.runtime_settings.task3_yoloe_device
        if configured:
            if str(configured).startswith("cuda") and not bool(torch_module.cuda.is_available()):
                if self.runtime_settings.task3_yoloe_allow_cpu:
                    return "cpu"
                raise Task3ExperimentalUnavailableError("cuda_unavailable")
            return str(configured)
        if bool(torch_module.cuda.is_available()):
            return "cuda:0"
        if self.runtime_settings.task3_yoloe_allow_cpu:
            return "cpu"
        raise Task3ExperimentalUnavailableError("cuda_unavailable")
