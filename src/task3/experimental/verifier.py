from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, DecodedFrame, FrameEnvelope
from src.core.utils import bbox_area
from src.core.vision import is_cv2_available
from src.task3.reference_cache import ReferenceCache

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2, np
else:  # pragma: no cover - cv2 yoksa
    cv2 = None
    np = None


@dataclass(slots=True)
class ExperimentalVerifier:
    runtime_settings: MvpRuntimeSettings
    reference_cache: ReferenceCache

    def effective_mode(self) -> tuple[str, bool]:
        requested = self.runtime_settings.task3_experimental_verifier
        if requested == "lightglue":
            try:
                __import__("lightglue")
                return "lightglue", True
            except Exception:
                return "orb_homography", False
        if requested == "off":
            return "off", True
        return "orb_homography", True

    def verify(
        self,
        frame: FrameEnvelope,
        decoded_frame: DecodedFrame,
        candidates: list[CanonicalUndefinedObject],
    ) -> tuple[list[CanonicalUndefinedObject], dict[str, Any]]:
        mode, exact = self.effective_mode()
        verified: list[CanonicalUndefinedObject] = []
        cross_sensor_used = 0
        for candidate in candidates:
            reference = self.reference_cache.get(candidate.object_id)
            if not reference:
                continue
            cross_sensor = bool(
                self.runtime_settings.task3_experimental_enable_cross_sensor_guard
                and (
                    str(reference.get("modality", "rgb")) != str(decoded_frame.modality)
                    or str(decoded_frame.modality) == "thermal"
                )
            )
            if cross_sensor:
                cross_sensor_used += 1
            if mode == "off":
                verified_candidate = self._verify_with_shape_guard(frame, decoded_frame, candidate, cross_sensor)
            elif mode == "lightglue" and exact:
                verified_candidate = self._verify_with_shape_guard(frame, decoded_frame, candidate, cross_sensor)
                if verified_candidate is not None:
                    verified_candidate.metadata["verification_mode"] = "lightglue_placeholder"
            else:
                verified_candidate = self._verify_with_orb(
                    frame,
                    decoded_frame,
                    candidate,
                    reference,
                    cross_sensor,
                )
            if verified_candidate is not None:
                verified.append(verified_candidate)
        diagnostics = {
            "requested_verifier_mode": self.runtime_settings.task3_experimental_verifier,
            "active_verifier_mode": mode,
            "verifier_exact": exact,
            "cross_sensor_guard_used": cross_sensor_used,
            "verified_candidate_count": len(verified),
        }
        return verified, diagnostics

    def _verify_with_shape_guard(
        self,
        frame: FrameEnvelope,
        decoded_frame: DecodedFrame,
        candidate: CanonicalUndefinedObject,
        cross_sensor: bool,
    ) -> CanonicalUndefinedObject | None:
        width = float(decoded_frame.width if decoded_frame.width else frame.metadata.get("image_width", 640))
        height = float(decoded_frame.height if decoded_frame.height else frame.metadata.get("image_height", 512))
        frame_area = max(width * height, 1.0)
        box = (
            float(candidate.top_left_x),
            float(candidate.top_left_y),
            float(candidate.bottom_right_x),
            float(candidate.bottom_right_y),
        )
        area_ratio = bbox_area(box) / frame_area
        sane = box[2] > box[0] and box[3] > box[1] and 0.0002 <= area_ratio <= 0.6
        if not sane:
            return None
        verifier_score = max(float(candidate.metadata.get("tracking_score", 0.0)), float(candidate.metadata.get("detection_score", candidate.metadata.get("match_score", 0.0))))
        if cross_sensor:
            verifier_score = max(verifier_score, 0.62)
        candidate.metadata["verification_mode"] = "shape_guard"
        candidate.metadata["verifier_score"] = round(verifier_score, 4)
        candidate.metadata["cross_sensor_guard"] = cross_sensor
        return candidate

    def _verify_with_orb(
        self,
        frame: FrameEnvelope,
        decoded_frame: DecodedFrame,
        candidate: CanonicalUndefinedObject,
        reference: dict[str, Any],
        cross_sensor: bool,
    ) -> CanonicalUndefinedObject | None:
        if not is_cv2_available() or decoded_frame.gray is None:
            return self._verify_with_shape_guard(frame, decoded_frame, candidate, cross_sensor)
        crop = self._crop_gray(decoded_frame.gray, candidate)
        ref_gray = reference.get("gray")
        if crop is None or ref_gray is None:
            return self._verify_with_shape_guard(frame, decoded_frame, candidate, cross_sensor)

        orb = cv2.ORB_create(nfeatures=self.runtime_settings.task3_orb_features)
        ref_kp, ref_desc = orb.detectAndCompute(ref_gray, None)
        crop_kp, crop_desc = orb.detectAndCompute(crop, None)
        if ref_desc is None or crop_desc is None or not ref_kp or not crop_kp:
            return self._verify_with_shape_guard(frame, decoded_frame, candidate, cross_sensor)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(ref_desc, crop_desc, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < self.runtime_settings.task3_match_ratio_threshold * second.distance:
                good.append(first)

        match_count = len(good)
        if cross_sensor:
            sane_candidate = self._verify_with_shape_guard(frame, decoded_frame, candidate, cross_sensor)
            if sane_candidate is None:
                return None
            crop_resized = cv2.resize(crop, (ref_gray.shape[1], ref_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            corroboration = self._corrcoef(ref_gray, crop_resized)
            if corroboration < -0.05:
                return None
            sane_candidate.metadata.update(
                {
                    "verification_mode": "cross_sensor_guard",
                    "verifier_score": round(max(float(candidate.metadata.get("detection_score", 0.0)), 0.6 + (max(corroboration, 0.0) * 0.2)), 4),
                    "verify_match_count": match_count,
                    "verify_corroboration": round(corroboration, 4),
                    "cross_sensor_guard": True,
                }
            )
            return sane_candidate

        if match_count < self.runtime_settings.task3_experimental_verify_min_matches:
            return None
        verifier_score = min(0.99, 0.45 + (0.05 * match_count))
        candidate.metadata.update(
            {
                "verification_mode": "orb_homography",
                "verifier_score": round(verifier_score, 4),
                "verify_match_count": match_count,
                "cross_sensor_guard": False,
            }
        )
        return candidate

    def _crop_gray(
        self,
        gray: Any,
        candidate: CanonicalUndefinedObject,
    ) -> Any | None:
        x1 = max(int(candidate.top_left_x), 0)
        y1 = max(int(candidate.top_left_y), 0)
        x2 = min(int(candidate.bottom_right_x), gray.shape[1])
        y2 = min(int(candidate.bottom_right_y), gray.shape[0])
        if x2 <= x1 or y2 <= y1:
            return None
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _corrcoef(self, first: Any, second: Any) -> float:
        if np is None:
            return 0.0
        first_vector = first.astype("float32").reshape(-1)
        second_vector = second.astype("float32").reshape(-1)
        if float(first_vector.std()) <= 1e-6 or float(second_vector.std()) <= 1e-6:
            return 0.0
        corr = float(np.corrcoef(first_vector, second_vector)[0, 1])
        if corr != corr:
            return 0.0
        return corr
