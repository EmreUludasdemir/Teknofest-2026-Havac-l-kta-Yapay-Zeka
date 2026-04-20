from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame
from src.core.vision import is_cv2_available
from src.task3.no_match_logic import compute_mode_candidate_score, normalize_yoloe_match_count
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


FALLBACK_REASON_MISSING_WEIGHT = "missing_yoloe_weight"
FALLBACK_REASON_MISSING_LIGHTGLUE = "missing_lightglue"
FALLBACK_REASON_CUDA_REQUIRED = "cuda_required_but_unavailable"
FALLBACK_REASON_BACKEND_EXCEPTION = "backend_exception"


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
        embeddings = torch.cat(vpes, dim=1)
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
        if getattr(vpe, "dim", lambda: 0)() == 2:
            vpe = vpe.unsqueeze(0)
        elif getattr(vpe, "dim", lambda: 0)() != 3:
            raise Task3ExperimentalUnavailableError(FALLBACK_REASON_BACKEND_EXCEPTION)
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

    def verify(self, crop_bgr: Any, ref_features: dict[str, Any]) -> tuple[bool, int, float, dict[str, Any] | None]:
        from lightglue.utils import rbd

        crop_tensor = self._prepare_crop(crop_bgr)
        if crop_tensor is None:
            return False, 0, 0.0, None

        start = time.perf_counter()
        try:
            crop_features = self.extractor.extract(crop_tensor)
            result = self.matcher({"image0": ref_features, "image1": crop_features})
            result = rbd(result)
            matches = result.get("matches")
            match_count = int(matches.shape[0]) if matches is not None else 0
        except Exception:
            return False, 0, 0.0, None
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        passed = match_count >= self.runtime_settings.task3_lightglue_min_matches

        kpts_data: dict[str, Any] | None = None
        if match_count > 0 and matches is not None:
            try:
                kpts0 = ref_features.get("keypoints")
                kpts1 = crop_features.get("keypoints")
                if kpts0 is not None and kpts1 is not None:
                    # features may retain batch dim [1,N,2]; squeeze to [N,2]
                    k0 = kpts0.squeeze(0) if kpts0.dim() == 3 else kpts0
                    k1 = kpts1.squeeze(0) if kpts1.dim() == 3 else kpts1
                    m_kpts0 = k0[matches[:, 0]].detach().cpu().numpy()
                    m_kpts1 = k1[matches[:, 1]].detach().cpu().numpy()
                    kpts_data = {"m_kpts0": m_kpts0, "m_kpts1": m_kpts1, "match_count": match_count}
            except Exception:
                kpts_data = None

        return passed, match_count, elapsed_ms, kpts_data


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
        scenario_id: str | None = None,
    ) -> tuple[list[CanonicalUndefinedObject], dict[str, Any]]:
        task3_info = {
            "requested_mode": "yoloe_vp_lightglue",
            "effective_mode": "yoloe_vp_lightglue",
            "fallback_reason": None,
            "references_loaded": len(reference_ids),
            "yoloe_inference_ms": 0.0,
            "lightglue_verify_ms_total": 0.0,
            "homography_compute_ms_total": 0.0,
            "candidates_generated": 0,
            "candidates_accepted": 0,
            "candidates_rejected": 0,
            "candidates_rejected_by_gate": 0,
        }
        self._ensure_ready(reference_ids)

        if decoded_frame.bgr is None:
            raise Task3ExperimentalUnavailableError("missing_bgr_frame")
        if self.reference_bank is None or self.reference_bank.ref_names is None or self.reference_bank.sp_features is None or self.verifier is None:
            raise Task3ExperimentalUnavailableError("reference_bank_not_ready")

        start = time.perf_counter()
        try:
            results = self.model.predict(
                decoded_frame.bgr,
                conf=self.runtime_settings.task3_yoloe_conf,
                iou=self.runtime_settings.task3_yoloe_iou,
                imgsz=self.runtime_settings.task3_yoloe_imgsz,
                max_det=max(self.runtime_settings.task3_yoloe_max_det_per_class * max(len(reference_ids), 1), 1),
                verbose=False,
                device=self.device,
            )
        except Task3ExperimentalUnavailableError:
            raise
        except Exception as exc:
            raise Task3ExperimentalUnavailableError(FALLBACK_REASON_BACKEND_EXCEPTION) from exc
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
        homography_ms_total = 0.0
        for candidate_idx, index in enumerate(keep_indices):
            x1 = max(int(raw_boxes[index, 0]), 0)
            y1 = max(int(raw_boxes[index, 1]), 0)
            x2 = min(int(raw_boxes[index, 2]), width - 1)
            y2 = min(int(raw_boxes[index, 3]), height - 1)
            if x2 <= x1 or y2 <= y1:
                self._dump_rejected_candidate(
                    scenario_id=scenario_id,
                    frame_idx=decoded_frame.frame_index,
                    candidate_idx=candidate_idx,
                    object_id=self.reference_bank.ref_names[int(raw_classes[index])],
                    match_count=0,
                    yoloe_confidence=float(raw_confidences[index]),
                    bbox=[x1, y1, x2, y2],
                    crop_bgr=None,
                )
                rejected += 1
                continue

            class_id = int(raw_classes[index])
            ref_name = self.reference_bank.ref_names[class_id]
            confidence = float(raw_confidences[index])
            match_count = 0
            verify_passed = True
            kpts_data: dict[str, Any] | None = None
            if verify_frame:
                crop = decoded_frame.bgr[y1:y2, x1:x2]
                verify_passed, match_count, verify_ms, kpts_data = self.verifier.verify(crop, self.reference_bank.sp_features[class_id])
                verify_ms_total += verify_ms
            else:
                crop = decoded_frame.bgr[y1:y2, x1:x2]

            inlier_count, inlier_ratio, homography_found, homography_compute_ms = _compute_homography_consistency(
                kpts_data,
                reproj_threshold=self.runtime_settings.task3_yoloe_homography_ransac_reproj_threshold,
            )
            homography_ms_total += homography_compute_ms

            if not verify_passed:
                self._dump_rejected_candidate(
                    scenario_id=scenario_id,
                    frame_idx=decoded_frame.frame_index,
                    candidate_idx=candidate_idx,
                    object_id=ref_name,
                    match_count=match_count,
                    yoloe_confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    crop_bgr=crop,
                )
                rejected += 1
                continue

            normalized_matches = (
                1.0
                if not verify_frame
                else normalize_yoloe_match_count(
                    match_count=match_count,
                    normalization_scale=self.runtime_settings.task3_yoloe_match_normalization_scale,
                )
            )
            score = compute_mode_candidate_score(
                confidence=confidence,
                normalized_matches=normalized_matches,
                inlier_ratio=inlier_ratio,
                mode="yoloe_vp_lightglue",
                yoloe_confidence_weight=self.runtime_settings.task3_yoloe_score_confidence_weight,
                yoloe_matches_weight=self.runtime_settings.task3_yoloe_score_matches_weight,
                yoloe_inlier_weight=self.runtime_settings.task3_yoloe_score_inlier_weight,
            )
            self._dump_post_gate_candidate(
                scenario_id=scenario_id,
                frame_idx=decoded_frame.frame_index,
                candidate_idx=candidate_idx,
                object_id=ref_name,
                match_count=match_count,
                normalized_matches=normalized_matches,
                yoloe_confidence=confidence,
                score=score,
                bbox=[x1, y1, x2, y2],
                crop_bgr=crop,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                homography_found=homography_found,
                homography_compute_ms=homography_compute_ms,
                kpts_data=kpts_data,
            )
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
                            "normalized_matches": round(float(normalized_matches), 4),
                            "verify_passed": bool(verify_passed),
                            "verify_skipped": bool(not verify_frame),
                            "inlier_count": int(inlier_count),
                            "inlier_ratio": round(float(inlier_ratio), 4),
                            "homography_found": bool(homography_found),
                        },
                    },
                )
            )

        task3_info["lightglue_verify_ms_total"] = round(verify_ms_total, 4)
        task3_info["homography_compute_ms_total"] = round(homography_ms_total, 4)
        task3_info["candidates_accepted"] = len(accepted)
        task3_info["candidates_rejected"] = rejected
        task3_info["candidates_rejected_by_gate"] = rejected
        return accepted, task3_info

    def _dump_rejected_candidate(
        self,
        *,
        scenario_id: str | None,
        frame_idx: int,
        candidate_idx: int,
        object_id: str,
        match_count: int,
        yoloe_confidence: float,
        bbox: list[int],
        crop_bgr: Any | None,
    ) -> None:
        if not self.runtime_settings.task3_debug_dump_rejects:
            return
        resolved_scenario_id = _sanitize_component(scenario_id or "unknown_scenario")
        scenario_dir = Path(self.runtime_settings.task3_debug_dump_dir) / resolved_scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        reject_record = {
            "frame_idx": int(frame_idx),
            "candidate_idx": int(candidate_idx),
            "object_id": object_id,
            "match_count": int(match_count),
            "yoloe_confidence": round(float(yoloe_confidence), 6),
            "bbox": [int(value) for value in bbox],
            "crop_hw": [0, 0] if crop_bgr is None else [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])],
        }
        with (scenario_dir / "rejects.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(reject_record, ensure_ascii=True) + "\n")

        reference = self.reference_cache.get(object_id) or {}
        reference_bgr = reference.get("bgr")
        if reference_bgr is None and reference.get("path"):
            reference_bgr = cv2.imread(str(reference["path"]))
        if reference_bgr is None:
            return
        crop_panel = crop_bgr if crop_bgr is not None and getattr(crop_bgr, "size", 0) else np.zeros((max(reference_bgr.shape[0], 32), 32, 3), dtype=reference_bgr.dtype)
        composite = self._build_reject_composite(reference_bgr, crop_panel, object_id=object_id, match_count=match_count)
        filename = f"{int(frame_idx):06d}_{int(candidate_idx):02d}_{_sanitize_component(object_id)}_m{int(match_count)}.png"
        cv2.imwrite(str(scenario_dir / filename), composite)

    def _dump_post_gate_candidate(
        self,
        *,
        scenario_id: str | None,
        frame_idx: int,
        candidate_idx: int,
        object_id: str,
        match_count: int,
        normalized_matches: float,
        yoloe_confidence: float,
        score: float,
        bbox: list[int],
        crop_bgr: Any | None,
        inlier_count: int,
        inlier_ratio: float,
        homography_found: bool,
        homography_compute_ms: float,
        kpts_data: dict[str, Any] | None = None,
    ) -> None:
        if not self.runtime_settings.task3_debug_dump_rejects:
            return
        resolved_scenario_id = _sanitize_component(scenario_id or "unknown_scenario")
        scenario_dir = Path(self.runtime_settings.task3_debug_dump_dir) / resolved_scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "candidate_idx": int(candidate_idx),
            "object_id": object_id,
            "match_count": int(match_count),
            "normalized_matches": round(float(normalized_matches), 6),
            "yoloe_confidence": round(float(yoloe_confidence), 6),
            "score": round(float(score), 6),
            "inlier_count": int(inlier_count),
            "inlier_ratio": round(float(inlier_ratio), 6),
            "homography_found": bool(homography_found),
            "homography_compute_ms": round(float(homography_compute_ms), 6),
            "bbox": [int(value) for value in bbox],
            "crop_hw": [0, 0] if crop_bgr is None else [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])],
        }
        if self.runtime_settings.task3_debug_export_keypoints and kpts_data is not None:
            m0 = kpts_data.get("m_kpts0")
            m1 = kpts_data.get("m_kpts1")
            if m0 is not None and m1 is not None:
                payload["m_kpts0"] = m0.tolist()
                payload["m_kpts1"] = m1.tolist()
        with (scenario_dir / "post_gate.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _build_reject_composite(self, reference_bgr: Any, crop_bgr: Any, *, object_id: str, match_count: int) -> Any:
        ref_panel = _ensure_bgr(reference_bgr)
        crop_panel = _ensure_bgr(crop_bgr)
        target_height = max(ref_panel.shape[0], crop_panel.shape[0])
        ref_panel = _pad_to_height(ref_panel, target_height)
        crop_panel = _pad_to_height(crop_panel, target_height)
        composite = cv2.hconcat([ref_panel, crop_panel])
        overlay = f"{object_id} m={int(match_count)}"
        cv2.putText(
            composite,
            overlay,
            (8, min(28, composite.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return composite

    def _ensure_ready(self, reference_ids: list[str]) -> None:
        if not is_cv2_available():
            raise Task3ExperimentalUnavailableError("opencv_unavailable")
        weight_path = Path(self.runtime_settings.task3_yoloe_weight_path)
        if not weight_path.exists():
            raise Task3ExperimentalUnavailableError(FALLBACK_REASON_MISSING_WEIGHT)
        if importlib.util.find_spec("lightglue") is None:
            raise Task3ExperimentalUnavailableError(FALLBACK_REASON_MISSING_LIGHTGLUE)
        if importlib.util.find_spec("ultralytics") is None:
            raise Task3ExperimentalUnavailableError("missing_ultralytics")
        if importlib.util.find_spec("torch") is None:
            raise Task3ExperimentalUnavailableError("missing_torch")
        try:
            import torch  # type: ignore[import-not-found]
            from lightglue import LightGlue, SuperPoint  # type: ignore[import-not-found]
            from ultralytics import YOLOE  # type: ignore[import-not-found]
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor  # type: ignore[import-not-found]
        except Exception as exc:
            raise Task3ExperimentalUnavailableError(FALLBACK_REASON_BACKEND_EXCEPTION) from exc

        resolved_device = self._resolve_device(torch)
        try:
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
        except Task3ExperimentalUnavailableError:
            raise
        except Exception as exc:
            raise Task3ExperimentalUnavailableError(FALLBACK_REASON_BACKEND_EXCEPTION) from exc

    def _resolve_device(self, torch_module: Any) -> str:
        configured = self.runtime_settings.task3_yoloe_device
        if configured:
            if str(configured).startswith("cuda") and not bool(torch_module.cuda.is_available()):
                if self.runtime_settings.task3_yoloe_allow_cpu:
                    return "cpu"
                raise Task3ExperimentalUnavailableError(FALLBACK_REASON_CUDA_REQUIRED)
            return str(configured)
        if bool(torch_module.cuda.is_available()):
            return "cuda:0"
        if self.runtime_settings.task3_yoloe_allow_cpu:
            return "cpu"
        raise Task3ExperimentalUnavailableError(FALLBACK_REASON_CUDA_REQUIRED)


def _compute_homography_consistency(
    kpts_data: dict[str, Any] | None,
    *,
    reproj_threshold: float = 5.0,
) -> tuple[int, float, bool, float]:
    start = time.perf_counter()
    if kpts_data is None:
        return 0, 0.0, False, 0.0
    m_kpts0 = kpts_data.get("m_kpts0")
    m_kpts1 = kpts_data.get("m_kpts1")
    if isinstance(m_kpts0, list):
        import numpy as _np
        m_kpts0 = _np.array(m_kpts0, dtype="float32")
        m_kpts1 = _np.array(m_kpts1, dtype="float32")
    total = int(kpts_data.get("match_count", 0)) or (len(m_kpts0) if m_kpts0 is not None else 0)
    if m_kpts0 is None or m_kpts1 is None or total < 4:
        return 0, 0.0, False, round((time.perf_counter() - start) * 1000.0, 6)
    try:
        src_pts = m_kpts0.reshape(-1, 1, 2).astype("float32")
        dst_pts = m_kpts1.reshape(-1, 1, 2).astype("float32")
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold)
        inlier_count = int(mask.ravel().sum()) if mask is not None else 0
        inlier_ratio = inlier_count / max(total, 1)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 6)
        return inlier_count, inlier_ratio, homography is not None and mask is not None, elapsed_ms
    except Exception:
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 6)
        return 0, 0.0, False, elapsed_ms


def _sanitize_component(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return sanitized or "item"


def _ensure_bgr(image: Any) -> Any:
    if image is None:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _pad_to_height(image: Any, target_height: int) -> Any:
    if image.shape[0] >= target_height:
        return image
    bottom = target_height - image.shape[0]
    return cv2.copyMakeBorder(image, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
