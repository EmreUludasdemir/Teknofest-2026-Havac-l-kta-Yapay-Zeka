from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame, FrameEnvelope
from src.core.utils import extract_frame_index
from src.core.vision import is_cv2_available
from src.task3.reference_cache import ReferenceCache

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2, np
else:  # pragma: no cover - cv2 yoksa
    cv2 = None
    np = None

try:  # pragma: no branch - ortama bagli
    import timm  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - bagimli
    timm = None
    torch = None

KNOWN_FALLBACK_REASONS = {
    "missing_yoloe_weight",
    "missing_lightglue",
    "cuda_required_but_unavailable",
    "backend_exception",
    "missing_decoded_frame",
    "missing_bgr_frame",
    "reference_bank_not_ready",
}


@dataclass(slots=True)
class LearnedDescriptorEmbedder:
    backbone_name: str
    input_size: int = 224
    pretrained: bool = True
    model: Any | None = None
    device: Any | None = None

    def available(self) -> bool:
        return timm is not None and torch is not None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.available():
            raise RuntimeError("timm_or_torch_unavailable")
        self.device = torch.device("cpu")
        try:
            self.model = timm.create_model(
                self.backbone_name,
                pretrained=self.pretrained,
                num_classes=0,
                global_pool="avg",
            )
        except Exception:
            if self.pretrained:
                self.model = timm.create_model(
                    self.backbone_name,
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                )
            else:
                raise
        self.model.eval()
        self.model.to(self.device)

    def encode(self, image: Any) -> Any:
        self.load()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        embedding = embedding.detach().cpu().numpy().reshape(-1)
        norm = float(np.linalg.norm(embedding)) if np is not None else 0.0
        if norm > 0.0:
            embedding = embedding / norm
        return embedding


@dataclass(slots=True)
class Task3Matcher:
    """Descriptor tabanli baseline; learned ve YOLOE yollari opsiyoneldir."""

    reference_cache: ReferenceCache
    runtime_settings: MvpRuntimeSettings
    learned_embedder: LearnedDescriptorEmbedder | None = field(default=None)
    experimental_backend: Any | None = field(default=None, init=False, repr=False)
    last_run_info: dict[str, Any] = field(default_factory=dict, init=False)

    def match(
        self,
        frame: FrameEnvelope,
        image_bytes: bytes,
        reference_ids: list[str] | None = None,
        *,
        decoded_frame: DecodedFrame | None = None,
        mode: str | None = None,
    ) -> list[CanonicalUndefinedObject]:
        requested_mode = mode or self.runtime_settings.task3_mode
        available_ids = reference_ids or self.reference_cache.list_ids()
        self.last_run_info = self._empty_task3_info(requested_mode=requested_mode, references_loaded=len(available_ids))

        if not available_ids:
            self.last_run_info["effective_mode"] = "none"
            return []

        if requested_mode == "yoloe_vp_lightglue":
            try:
                matches = self._match_with_yoloe_vp_lightglue(decoded_frame, available_ids, scenario_id=frame.video_name)
                self._finalize_info(matches, effective_mode="yoloe_vp_lightglue")
                return matches
            except Exception as exc:
                reason = str(exc)
                self.last_run_info["fallback_reason"] = reason if reason in KNOWN_FALLBACK_REASONS else "backend_exception"

        if is_cv2_available() and decoded_frame is not None and decoded_frame.gray is not None:
            if requested_mode == "learned_descriptor":
                learned_matches = self._match_with_learned_descriptor(decoded_frame, available_ids)
                if learned_matches:
                    self._finalize_info(learned_matches, effective_mode="learned_descriptor")
                    return learned_matches
            matched = self._match_with_descriptors(decoded_frame, available_ids)
            if matched:
                self._finalize_info(matched, effective_mode="orb_template")
                return matched
            matched = self._match_with_template(decoded_frame, available_ids)
            if matched:
                self._finalize_info(matched, effective_mode="orb_template")
                return matched

        placeholder = self._placeholder_match(frame, available_ids)
        self._finalize_info(placeholder, effective_mode="placeholder")
        return placeholder

    def _empty_task3_info(self, *, requested_mode: str, references_loaded: int) -> dict[str, Any]:
        return {
            "requested_mode": requested_mode,
            "effective_mode": requested_mode if requested_mode != "yoloe_vp_lightglue" else "orb_template",
            "fallback_reason": None,
            "references_loaded": references_loaded,
            "yoloe_inference_ms": 0.0,
            "lightglue_verify_ms_total": 0.0,
            "candidates_generated": 0,
            "candidates_accepted": 0,
            "candidates_rejected": 0,
        }

    def _finalize_info(self, matches: list[CanonicalUndefinedObject], *, effective_mode: str) -> None:
        self.last_run_info["effective_mode"] = effective_mode
        if not self.last_run_info.get("candidates_generated"):
            self.last_run_info["candidates_generated"] = len(matches)
        if not self.last_run_info.get("candidates_accepted"):
            self.last_run_info["candidates_accepted"] = len(matches)

    def _match_with_yoloe_vp_lightglue(
        self,
        decoded_frame: DecodedFrame | None,
        reference_ids: list[str],
        *,
        scenario_id: str | None = None,
    ) -> list[CanonicalUndefinedObject]:
        if decoded_frame is None or decoded_frame.bgr is None:
            raise RuntimeError("missing_decoded_frame")

        try:
            from src.task3.experimental.backend import (
                Task3ExperimentalUnavailableError,
                YoloeVpLightGlueBackend,
            )
        except Exception:
            raise RuntimeError("backend_exception")

        if self.experimental_backend is None:
            self.experimental_backend = YoloeVpLightGlueBackend(
                reference_cache=self.reference_cache,
                runtime_settings=self.runtime_settings,
            )
        try:
            matches, task3_info = self.experimental_backend.match(
                decoded_frame=decoded_frame,
                reference_ids=reference_ids,
                scenario_id=scenario_id,
            )
        except Task3ExperimentalUnavailableError as exc:
            raise RuntimeError(exc.reason) from exc
        self.last_run_info.update(task3_info)
        return matches

    def _collect_proposals(
        self,
        decoded_frame: DecodedFrame,
        reference_ids: list[str],
    ) -> list[CanonicalUndefinedObject]:
        proposals: list[CanonicalUndefinedObject] = []
        proposals.extend(self._match_with_descriptors(decoded_frame, reference_ids))
        proposals.extend(
            self._match_with_template(
                decoded_frame,
                reference_ids,
                min_score=self.runtime_settings.task3_template_proposal_threshold,
                top_k=3,
            )
        )
        deduped: dict[tuple[str, int, int, int, int], CanonicalUndefinedObject] = {}
        for item in proposals:
            key = (
                item.object_id,
                int(item.top_left_x // 4),
                int(item.top_left_y // 4),
                int(item.bottom_right_x // 4),
                int(item.bottom_right_y // 4),
            )
            current = deduped.get(key)
            if current is None or float(item.metadata.get("match_score", 0.0)) > float(current.metadata.get("match_score", 0.0)):
                deduped[key] = item
        return sorted(deduped.values(), key=lambda item: float(item.metadata.get("match_score", 0.0)), reverse=True)

    def _match_with_learned_descriptor(
        self,
        decoded_frame: DecodedFrame,
        reference_ids: list[str],
    ) -> list[CanonicalUndefinedObject]:
        if not is_cv2_available():
            return []
        embedder = self.learned_embedder or LearnedDescriptorEmbedder(
            backbone_name=self.runtime_settings.task3_learned_backbone,
            input_size=self.runtime_settings.task3_learned_input_size,
            pretrained=self.runtime_settings.task3_learned_pretrained,
        )
        try:
            embedder.load()
        except Exception:
            return []

        frame_bgr = decoded_frame.bgr
        if frame_bgr is None and decoded_frame.gray is not None:
            frame_bgr = cv2.cvtColor(decoded_frame.gray, cv2.COLOR_GRAY2BGR)
        if frame_bgr is None:
            return []

        proposals = self._collect_proposals(decoded_frame, reference_ids)
        if not proposals:
            return []

        reranked: list[CanonicalUndefinedObject] = []
        for proposal in proposals:
            reference = self.reference_cache.get(proposal.object_id)
            if not reference:
                continue
            reference_embedding = self.reference_cache.get_learned_embedding(proposal.object_id, embedder.backbone_name)
            if reference_embedding is None:
                reference_bgr = reference.get("bgr")
                if reference_bgr is None and reference.get("gray") is not None:
                    reference_bgr = cv2.cvtColor(reference["gray"], cv2.COLOR_GRAY2BGR)
                if reference_bgr is None:
                    continue
                reference_embedding = embedder.encode(reference_bgr)
                self.reference_cache.set_learned_embedding(proposal.object_id, embedder.backbone_name, reference_embedding)

            crop = self._crop_bbox(frame_bgr, proposal)
            if crop is None:
                continue
            crop_embedding = embedder.encode(crop)
            similarity = self._cosine_similarity(reference_embedding, crop_embedding)
            corroboration = self._compute_crop_corroboration(reference, crop)
            score = (similarity * 0.85) + (max(corroboration, 0.0) * 0.15)
            proposal.metadata.update(
                {
                    "matcher_source": "task3_learned_descriptor",
                    "candidate_name": "learned_resnet18_descriptor",
                    "embedding_backbone": embedder.backbone_name,
                    "similarity": round(float(similarity), 4),
                    "corroboration": round(float(corroboration), 4),
                    "match_score": round(float(score), 4),
                    "inlier_count": self.runtime_settings.task3_match_min_inliers,
                    "inlier_ratio": round(float(max(similarity, 0.0)), 4),
                    "match_count": max(int(proposal.metadata.get("match_count", 1)), 1),
                }
            )
            reranked.append(proposal)

        reranked = [item for item in reranked if float(item.metadata.get("similarity", 0.0)) >= self.runtime_settings.task3_learned_min_similarity]
        reranked.sort(key=lambda item: float(item.metadata.get("match_score", 0.0)), reverse=True)
        if len(reranked) >= 2:
            gap = float(reranked[0].metadata.get("match_score", 0.0)) - float(reranked[1].metadata.get("match_score", 0.0))
            if gap < self.runtime_settings.task3_learned_ambiguity_gap:
                return []
        return reranked[:1]

    def _match_with_descriptors(
        self,
        decoded_frame: DecodedFrame,
        reference_ids: list[str],
    ) -> list[CanonicalUndefinedObject]:
        orb = cv2.ORB_create(nfeatures=self.runtime_settings.task3_orb_features)
        frame_keypoints, frame_descriptors = orb.detectAndCompute(decoded_frame.gray, None)
        if frame_descriptors is None or not len(frame_keypoints):
            return []

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        candidates: list[CanonicalUndefinedObject] = []
        for reference_id in reference_ids:
            reference = self.reference_cache.get(reference_id)
            if not reference:
                continue
            ref_descriptors = reference.get("descriptors")
            ref_keypoints = reference.get("keypoints")
            if ref_descriptors is None or not ref_keypoints:
                continue

            raw_matches = matcher.knnMatch(ref_descriptors, frame_descriptors, k=2)
            good_matches = []
            for pair in raw_matches:
                if len(pair) != 2:
                    continue
                first, second = pair
                if first.distance < self.runtime_settings.task3_match_ratio_threshold * second.distance:
                    good_matches.append(first)

            if len(good_matches) < self.runtime_settings.task3_match_min_inliers:
                continue

            ref_points = np.float32([ref_keypoints[item.queryIdx].pt for item in good_matches]).reshape(-1, 1, 2)
            frame_points = np.float32([frame_keypoints[item.trainIdx].pt for item in good_matches]).reshape(-1, 1, 2)
            homography, inlier_mask = cv2.findHomography(ref_points, frame_points, cv2.RANSAC, 5.0)
            if homography is None or inlier_mask is None:
                frame_xy = frame_points.reshape(-1, 2)
                top_left_x = float(frame_xy[:, 0].min())
                top_left_y = float(frame_xy[:, 1].min())
                bottom_right_x = float(frame_xy[:, 0].max())
                bottom_right_y = float(frame_xy[:, 1].max())
                inlier_ratio = min(len(good_matches) / max(self.runtime_settings.task3_match_min_inliers, 1), 1.0)
                candidates.append(
                    CanonicalUndefinedObject(
                        object_id=reference_id,
                        top_left_x=top_left_x,
                        top_left_y=top_left_y,
                        bottom_right_x=bottom_right_x,
                        bottom_right_y=bottom_right_y,
                        metadata={
                            "match_score": round(float(min(0.95, 0.60 + 0.05 * len(good_matches))), 4),
                            "matcher_source": "task3_orb_bf_bbox",
                            "candidate_name": "orb_bf_homography",
                            "inlier_count": len(good_matches),
                            "inlier_ratio": round(float(inlier_ratio), 4),
                            "match_count": len(good_matches),
                        },
                    )
                )
                continue

            inlier_count = int(inlier_mask.ravel().sum())
            if inlier_count < self.runtime_settings.task3_match_min_inliers:
                continue

            ref_width = max(int(reference.get("width", 0)), 1)
            ref_height = max(int(reference.get("height", 0)), 1)
            corners = np.float32(
                [[0, 0], [ref_width - 1, 0], [ref_width - 1, ref_height - 1], [0, ref_height - 1]]
            ).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
            top_left_x = float(projected[:, 0].min())
            top_left_y = float(projected[:, 1].min())
            bottom_right_x = float(projected[:, 0].max())
            bottom_right_y = float(projected[:, 1].max())
            inlier_ratio = inlier_count / max(len(good_matches), 1)
            score = min(0.99, 0.55 + (0.45 * inlier_ratio))
            candidates.append(
                CanonicalUndefinedObject(
                    object_id=reference_id,
                    top_left_x=top_left_x,
                    top_left_y=top_left_y,
                    bottom_right_x=bottom_right_x,
                    bottom_right_y=bottom_right_y,
                    metadata={
                        "match_score": round(float(score), 4),
                        "matcher_source": "task3_orb_bf_homography",
                        "candidate_name": "orb_bf_homography",
                        "inlier_count": inlier_count,
                        "inlier_ratio": round(float(inlier_ratio), 4),
                        "match_count": len(good_matches),
                    },
                )
            )
        return sorted(candidates, key=lambda item: float(item.metadata.get("match_score", 0.0)), reverse=True)

    def _match_with_template(
        self,
        decoded_frame: DecodedFrame,
        reference_ids: list[str],
        *,
        min_score: float = 0.72,
        top_k: int = 1,
    ) -> list[CanonicalUndefinedObject]:
        candidates: list[CanonicalUndefinedObject] = []
        for reference_id in reference_ids:
            reference = self.reference_cache.get(reference_id)
            if not reference:
                continue
            ref_gray = reference.get("gray")
            ref_width = int(reference.get("width", 0))
            ref_height = int(reference.get("height", 0))
            if ref_gray is None or ref_width <= 0 or ref_height <= 0:
                continue
            if decoded_frame.width < ref_width or decoded_frame.height < ref_height:
                continue

            response = cv2.matchTemplate(decoded_frame.gray, ref_gray, cv2.TM_CCOEFF_NORMED)
            flat = response.reshape(-1)
            if flat.size == 0:
                continue
            top_count = min(max(top_k, 1), flat.size)
            top_indices = np.argpartition(flat, -top_count)[-top_count:]
            top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
            taken_locations: list[tuple[int, int]] = []
            for flat_index in top_indices:
                max_value = float(flat[flat_index])
                if max_value < min_score:
                    continue
                y, x = divmod(int(flat_index), response.shape[1])
                if any(abs(x - existing[0]) < max(ref_width // 2, 1) and abs(y - existing[1]) < max(ref_height // 2, 1) for existing in taken_locations):
                    continue
                taken_locations.append((x, y))
                top_left_x = float(x)
                top_left_y = float(y)
                candidates.append(
                    CanonicalUndefinedObject(
                        object_id=reference_id,
                        top_left_x=top_left_x,
                        top_left_y=top_left_y,
                        bottom_right_x=top_left_x + ref_width,
                        bottom_right_y=top_left_y + ref_height,
                        metadata={
                            "match_score": round(max_value, 4),
                            "matcher_source": "task3_template_ncc",
                            "candidate_name": "template_ncc_verifier",
                            "inlier_count": self.runtime_settings.task3_match_min_inliers,
                            "inlier_ratio": round(max_value, 4),
                            "match_count": 1,
                        },
                    )
                )
        return sorted(candidates, key=lambda item: float(item.metadata.get("match_score", 0.0)), reverse=True)

    def _placeholder_match(self, frame: FrameEnvelope, available_ids: list[str]) -> list[CanonicalUndefinedObject]:
        frame_index = extract_frame_index(frame.frame_url)
        primary_id = sorted(available_ids)[frame_index % len(available_ids)]
        base_x = 60.0 + float((frame_index * 13) % 140)
        base_y = 70.0 + float((frame_index * 9) % 120)
        primary_score = 0.82 if frame_index % 5 != 0 else 0.74
        matches = [
            CanonicalUndefinedObject(
                object_id=primary_id,
                top_left_x=base_x,
                top_left_y=base_y,
                bottom_right_x=base_x + 52.0,
                bottom_right_y=base_y + 52.0,
                metadata={
                    "match_score": primary_score,
                    "matcher_source": "task3_placeholder",
                    "candidate_name": "placeholder_matcher",
                },
            )
        ]

        if len(available_ids) > 1 and frame_index % 5 == 0:
            secondary_id = sorted(available_ids)[(frame_index + 1) % len(available_ids)]
            matches.append(
                CanonicalUndefinedObject(
                    object_id=secondary_id,
                    top_left_x=base_x + 6.0,
                    top_left_y=base_y + 4.0,
                    bottom_right_x=base_x + 56.0,
                    bottom_right_y=base_y + 56.0,
                    metadata={
                        "match_score": primary_score - 0.02,
                        "matcher_source": "task3_placeholder_secondary",
                        "candidate_name": "placeholder_matcher",
                    },
                )
            )

        return matches

    def _crop_bbox(self, frame_bgr: Any, proposal: CanonicalUndefinedObject) -> Any | None:
        x1 = max(int(math.floor(proposal.top_left_x)), 0)
        y1 = max(int(math.floor(proposal.top_left_y)), 0)
        x2 = min(int(math.ceil(proposal.bottom_right_x)), frame_bgr.shape[1])
        y2 = min(int(math.ceil(proposal.bottom_right_y)), frame_bgr.shape[0])
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _compute_crop_corroboration(self, reference: dict[str, Any], crop_bgr: Any) -> float:
        ref_gray = reference.get("gray")
        if ref_gray is None:
            return 0.0
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(crop_gray, (ref_gray.shape[1], ref_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        ref_vector = ref_gray.astype("float32").reshape(-1)
        crop_vector = resized.astype("float32").reshape(-1)
        ref_std = float(ref_vector.std())
        crop_std = float(crop_vector.std())
        if ref_std <= 1e-6 or crop_std <= 1e-6:
            return 0.0
        corr = float(np.corrcoef(ref_vector, crop_vector)[0, 1])
        if math.isnan(corr):
            return 0.0
        return max(min(corr, 1.0), -1.0)

    def _cosine_similarity(self, first: Any, second: Any) -> float:
        denom = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(first, second) / denom)
