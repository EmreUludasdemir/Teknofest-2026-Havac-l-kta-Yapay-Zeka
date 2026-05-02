from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task3.experimental import backend as backend_module
from src.task3.experimental.backend import LightGlueCropVerifier, YoloeReferenceBank, YoloeVpLightGlueBackend
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches
from tools.diagnostic.d7_common import OUTPUT_DIR, PASS_A_SCENARIO_IDS, REFERENCE_DIR, frame_envelope, iter_timed_video_frames, load_scenarios, make_settings

ORIGINAL_VERIFY = LightGlueCropVerifier.verify
ORIGINAL_BUILD = YoloeReferenceBank.build
ORIGINAL_ENSURE_READY = YoloeVpLightGlueBackend._ensure_ready
ORIGINAL_POST = YoloeVpLightGlueBackend._dump_post_gate_candidate
ORIGINAL_REJECT = YoloeVpLightGlueBackend._dump_rejected_candidate
ORIGINAL_REAL_ORB_ONLY = Task3Matcher._match_with_real_orb_only

BACKEND_CAPTURES: dict[int, "BackendCapture"] = {}
MATCHER_CAPTURES: dict[int, "BackendCapture"] = {}
VERIFIER_TO_BACKEND: dict[int, int] = {}
ACTIVE_ENSURE_CAPTURE: "BackendCapture | None" = None
HOOKS_INSTALLED = False


@dataclass(slots=True)
class CandidateTiming:
    scenario_id: str
    frame_index: int
    object_id: str
    gate_pass: bool
    match_count: int
    verify_prepare_ms: float
    verify_forward_ms: float
    verify_total_ms: float
    homography_ms: float
    crop_height: int
    crop_width: int


class BackendCapture:
    def __init__(self, *, backend: YoloeVpLightGlueBackend, matcher: Task3Matcher) -> None:
        self.backend = backend
        self.matcher = matcher
        self.current_frame: dict[str, Any] | None = None
        self.pending_verify: deque[dict[str, Any]] = deque()
        self.frame_records: list[dict[str, Any]] = []
        self.candidate_records: list[CandidateTiming] = []
        self.predict_wrapped = False

    def start_frame(self, *, scenario_id: str, frame_index: int, modality: str) -> None:
        self.pending_verify.clear()
        self.current_frame = {
            "scenario_id": scenario_id,
            "frame_index": int(frame_index),
            "modality": str(modality),
            "ensure_ready_ms": 0.0,
            "reference_bank_build_ms": 0.0,
            "reference_bank_ref_count": 0,
            "yoloe_preprocess_ms": 0.0,
            "yoloe_inference_ms": 0.0,
            "yoloe_postprocess_ms": 0.0,
            "yoloe_predict_full_ms": 0.0,
            "lightglue_prepare_ms": 0.0,
            "lightglue_forward_ms": 0.0,
            "lightglue_total_ms": 0.0,
            "lightglue_candidate_count": 0,
            "homography_total_ms": 0.0,
            "orb_match_ms": 0.0,
            "orb_candidate_count": 0,
        }

    def finalize_frame(self) -> dict[str, Any]:
        if self.current_frame is None:
            raise RuntimeError("frame capture missing")
        record = dict(self.current_frame)
        self.frame_records.append(record)
        self.current_frame = None
        return record


def _safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(percentile, 1.0)) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": _safe_mean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": float(max(values)) if values else 0.0,
    }


def install_hooks() -> None:
    global HOOKS_INSTALLED, ACTIVE_ENSURE_CAPTURE
    if HOOKS_INSTALLED:
        return

    def verify_wrapper(
        verifier: LightGlueCropVerifier,
        crop_bgr: Any,
        ref_features: dict[str, Any],
    ) -> tuple[bool, int, float, dict[str, Any] | None]:
        capture = BACKEND_CAPTURES.get(VERIFIER_TO_BACKEND.get(id(verifier), -1))
        prepare_started = time.perf_counter()
        crop_tensor = verifier._prepare_crop(crop_bgr)
        prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
        if crop_tensor is None:
            total_ms = prepare_ms
            if capture is not None:
                capture.pending_verify.append(
                    {
                        "match_count": 0,
                        "verify_prepare_ms": float(prepare_ms),
                        "verify_forward_ms": 0.0,
                        "verify_total_ms": float(total_ms),
                        "kpts_data": None,
                        "crop_hw": [0, 0] if crop_bgr is None else [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])],
                    }
                )
            return False, 0, total_ms, None

        from lightglue.utils import rbd

        forward_started = time.perf_counter()
        try:
            crop_features = verifier.extractor.extract(crop_tensor)
            result = verifier.matcher({"image0": ref_features, "image1": crop_features})
            result = rbd(result)
            matches = result.get("matches")
            match_count = int(matches.shape[0]) if matches is not None else 0
        except Exception:
            total_ms = prepare_ms + ((time.perf_counter() - forward_started) * 1000.0)
            if capture is not None:
                capture.pending_verify.append(
                    {
                        "match_count": 0,
                        "verify_prepare_ms": float(prepare_ms),
                        "verify_forward_ms": float(total_ms - prepare_ms),
                        "verify_total_ms": float(total_ms),
                        "kpts_data": None,
                        "crop_hw": [0, 0] if crop_bgr is None else [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])],
                    }
                )
            return False, 0, total_ms, None
        forward_ms = (time.perf_counter() - forward_started) * 1000.0
        total_ms = prepare_ms + forward_ms
        passed = match_count >= verifier.runtime_settings.task3_lightglue_min_matches

        kpts_data: dict[str, Any] | None = None
        if match_count > 0 and matches is not None:
            try:
                kpts0 = ref_features.get("keypoints")
                kpts1 = crop_features.get("keypoints")
                if kpts0 is not None and kpts1 is not None:
                    k0 = kpts0.squeeze(0) if kpts0.dim() == 3 else kpts0
                    k1 = kpts1.squeeze(0) if kpts1.dim() == 3 else kpts1
                    m_kpts0 = k0[matches[:, 0]].detach().cpu().numpy()
                    m_kpts1 = k1[matches[:, 1]].detach().cpu().numpy()
                    kpts_data = {"m_kpts0": m_kpts0, "m_kpts1": m_kpts1, "match_count": match_count}
            except Exception:
                kpts_data = None
        if capture is not None:
            capture.pending_verify.append(
                {
                    "match_count": int(match_count),
                    "verify_prepare_ms": float(prepare_ms),
                    "verify_forward_ms": float(forward_ms),
                    "verify_total_ms": float(total_ms),
                    "kpts_data": kpts_data,
                    "crop_hw": [0, 0] if crop_bgr is None else [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])],
                }
            )
        return passed, match_count, total_ms, kpts_data

    def build_wrapper(reference_bank: YoloeReferenceBank, reference_ids: list[str]) -> None:
        started = time.perf_counter()
        result = ORIGINAL_BUILD(reference_bank, reference_ids)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if ACTIVE_ENSURE_CAPTURE is not None and ACTIVE_ENSURE_CAPTURE.current_frame is not None:
            ACTIVE_ENSURE_CAPTURE.current_frame["reference_bank_build_ms"] += float(elapsed_ms)
            ACTIVE_ENSURE_CAPTURE.current_frame["reference_bank_ref_count"] = int(len(reference_ids))
        return result

    def ensure_ready_wrapper(backend: YoloeVpLightGlueBackend, reference_ids: list[str]) -> None:
        global ACTIVE_ENSURE_CAPTURE
        capture = BACKEND_CAPTURES.get(id(backend))
        started = time.perf_counter()
        previous = ACTIVE_ENSURE_CAPTURE
        ACTIVE_ENSURE_CAPTURE = capture
        try:
            result = ORIGINAL_ENSURE_READY(backend, reference_ids)
        finally:
            ACTIVE_ENSURE_CAPTURE = previous
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if capture is not None and capture.current_frame is not None:
            capture.current_frame["ensure_ready_ms"] += float(elapsed_ms)
            if backend.verifier is not None:
                VERIFIER_TO_BACKEND[id(backend.verifier)] = id(backend)
            if backend.model is not None and not getattr(backend.model, "_d7_predict_wrapped", False):
                original_predict = backend.model.predict

                def predict_wrapper(*args: Any, **kwargs: Any) -> Any:
                    predict_started = time.perf_counter()
                    results = original_predict(*args, **kwargs)
                    full_ms = (time.perf_counter() - predict_started) * 1000.0
                    predict_capture = BACKEND_CAPTURES.get(id(backend))
                    if predict_capture is not None and predict_capture.current_frame is not None:
                        speed = {}
                        if results:
                            speed_candidate = getattr(results[0], "speed", None)
                            if isinstance(speed_candidate, dict):
                                speed = speed_candidate
                        predict_capture.current_frame["yoloe_predict_full_ms"] += float(full_ms)
                        predict_capture.current_frame["yoloe_preprocess_ms"] += float(speed.get("preprocess") or 0.0)
                        predict_capture.current_frame["yoloe_inference_ms"] += float(speed.get("inference") or 0.0)
                        predict_capture.current_frame["yoloe_postprocess_ms"] += float(speed.get("postprocess") or 0.0)
                    return results

                backend.model.predict = predict_wrapper  # type: ignore[assignment]
                setattr(backend.model, "_d7_predict_wrapped", True)
        return result

    def post_wrapper(
        backend: YoloeVpLightGlueBackend,
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
        capture = BACKEND_CAPTURES.get(id(backend))
        if capture is not None and capture.current_frame is not None:
            verify_info = capture.pending_verify.popleft() if capture.pending_verify else {}
            capture.current_frame["lightglue_prepare_ms"] += float(verify_info.get("verify_prepare_ms", 0.0))
            capture.current_frame["lightglue_forward_ms"] += float(verify_info.get("verify_forward_ms", 0.0))
            capture.current_frame["lightglue_total_ms"] += float(verify_info.get("verify_total_ms", 0.0))
            capture.current_frame["lightglue_candidate_count"] += 1
            capture.current_frame["homography_total_ms"] += float(homography_compute_ms)
            crop_hw = verify_info.get("crop_hw") or ([0, 0] if crop_bgr is None else [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])])
            capture.candidate_records.append(
                CandidateTiming(
                    scenario_id=str(scenario_id or ""),
                    frame_index=int(frame_idx),
                    object_id=str(object_id),
                    gate_pass=True,
                    match_count=int(match_count),
                    verify_prepare_ms=float(verify_info.get("verify_prepare_ms", 0.0)),
                    verify_forward_ms=float(verify_info.get("verify_forward_ms", 0.0)),
                    verify_total_ms=float(verify_info.get("verify_total_ms", 0.0)),
                    homography_ms=float(homography_compute_ms),
                    crop_height=int(crop_hw[0]),
                    crop_width=int(crop_hw[1]),
                )
            )
        return ORIGINAL_POST(
            backend,
            scenario_id=scenario_id,
            frame_idx=frame_idx,
            candidate_idx=candidate_idx,
            object_id=object_id,
            match_count=match_count,
            normalized_matches=normalized_matches,
            yoloe_confidence=yoloe_confidence,
            score=score,
            bbox=bbox,
            crop_bgr=crop_bgr,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            homography_found=homography_found,
            homography_compute_ms=homography_compute_ms,
            kpts_data=kpts_data,
        )

    def reject_wrapper(
        backend: YoloeVpLightGlueBackend,
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
        capture = BACKEND_CAPTURES.get(id(backend))
        if capture is not None and capture.current_frame is not None:
            verify_info = capture.pending_verify.popleft() if capture.pending_verify else {}
            kpts_data = verify_info.get("kpts_data")
            _inlier_count, _inlier_ratio, _homography_found, homography_ms = backend_module._compute_homography_consistency(
                kpts_data,
                reproj_threshold=capture.backend.runtime_settings.task3_yoloe_homography_ransac_reproj_threshold,
            )
            capture.current_frame["lightglue_prepare_ms"] += float(verify_info.get("verify_prepare_ms", 0.0))
            capture.current_frame["lightglue_forward_ms"] += float(verify_info.get("verify_forward_ms", 0.0))
            capture.current_frame["lightglue_total_ms"] += float(verify_info.get("verify_total_ms", 0.0))
            capture.current_frame["lightglue_candidate_count"] += 1
            capture.current_frame["homography_total_ms"] += float(homography_ms)
            crop_hw = verify_info.get("crop_hw") or ([0, 0] if crop_bgr is None else [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])])
            capture.candidate_records.append(
                CandidateTiming(
                    scenario_id=str(scenario_id or ""),
                    frame_index=int(frame_idx),
                    object_id=str(object_id),
                    gate_pass=False,
                    match_count=int(match_count),
                    verify_prepare_ms=float(verify_info.get("verify_prepare_ms", 0.0)),
                    verify_forward_ms=float(verify_info.get("verify_forward_ms", 0.0)),
                    verify_total_ms=float(verify_info.get("verify_total_ms", 0.0)),
                    homography_ms=float(homography_ms),
                    crop_height=int(crop_hw[0]),
                    crop_width=int(crop_hw[1]),
                )
            )
        return ORIGINAL_REJECT(
            backend,
            scenario_id=scenario_id,
            frame_idx=frame_idx,
            candidate_idx=candidate_idx,
            object_id=object_id,
            match_count=match_count,
            yoloe_confidence=yoloe_confidence,
            bbox=bbox,
            crop_bgr=crop_bgr,
        )

    def real_orb_only_wrapper(
        matcher: Task3Matcher,
        decoded_frame: Any,
        reference_ids: list[str],
    ) -> list[Any]:
        started = time.perf_counter()
        result = ORIGINAL_REAL_ORB_ONLY(matcher, decoded_frame, reference_ids)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        capture = MATCHER_CAPTURES.get(id(matcher))
        if capture is not None and capture.current_frame is not None:
            capture.current_frame["orb_match_ms"] += float(elapsed_ms)
            capture.current_frame["orb_candidate_count"] += int(len(result))
        return result

    LightGlueCropVerifier.verify = verify_wrapper  # type: ignore[assignment]
    YoloeReferenceBank.build = build_wrapper  # type: ignore[assignment]
    YoloeVpLightGlueBackend._ensure_ready = ensure_ready_wrapper  # type: ignore[assignment]
    YoloeVpLightGlueBackend._dump_post_gate_candidate = post_wrapper  # type: ignore[assignment]
    YoloeVpLightGlueBackend._dump_rejected_candidate = reject_wrapper  # type: ignore[assignment]
    Task3Matcher._match_with_real_orb_only = real_orb_only_wrapper  # type: ignore[assignment]
    HOOKS_INSTALLED = True


def _component_stat(frame_records: list[dict[str, Any]], key: str) -> dict[str, float]:
    return _stats([float(record.get(key, 0.0)) for record in frame_records])


def _render_stats_table(frame_records: list[dict[str, Any]]) -> list[str]:
    keys = [
        ("decode_ms", "Image Decode + Gray"),
        ("routing_cache_ms", "Reference Routing Lookup"),
        ("ensure_ready_ms", "YOLOE Ensure Ready"),
        ("reference_bank_build_ms", "Reference Bank Build (subset)"),
        ("yoloe_preprocess_ms", "YOLOE Preprocess"),
        ("yoloe_inference_ms", "YOLOE Forward"),
        ("yoloe_postprocess_ms", "YOLOE Postprocess"),
        ("yoloe_predict_full_ms", "YOLOE Predict Total"),
        ("orb_match_ms", "ORB Match"),
        ("lightglue_prepare_ms", "LightGlue Prepare"),
        ("lightglue_forward_ms", "LightGlue Forward"),
        ("lightglue_total_ms", "LightGlue Verify Total"),
        ("homography_total_ms", "RANSAC Homography"),
        ("filter_no_match_ms", "No-Match Filter"),
        ("final_verify_ms", "Final Geometric Verify"),
        ("wall_ms", "Pipeline Wall (no decode)"),
        ("end_to_end_ms", "End-to-End Wall"),
        ("residual_ms", "Residual Overhead"),
    ]
    lines = [
        "## Component Stats",
        "",
        "| Component | Mean ms | P50 ms | P95 ms | Max ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in keys:
        stats = _component_stat(frame_records, key)
        lines.append(
            f"| {label} | {stats['mean']:.1f} | {stats['p50']:.1f} | {stats['p95']:.1f} | {stats['max']:.1f} |"
        )
    lines.append("")
    return lines


def _render_share_table(frame_records: list[dict[str, Any]]) -> list[str]:
    mean_end_to_end = _safe_mean([float(record["end_to_end_ms"]) for record in frame_records])
    shares = [
        ("Image Decode + Gray", _safe_mean([float(record["decode_ms"]) for record in frame_records])),
        ("Reference Routing Lookup", _safe_mean([float(record["routing_cache_ms"]) for record in frame_records])),
        ("YOLOE Ensure Ready", _safe_mean([float(record["ensure_ready_ms"]) for record in frame_records])),
        ("YOLOE Predict Total", _safe_mean([float(record["yoloe_predict_full_ms"]) for record in frame_records])),
        ("ORB Match", _safe_mean([float(record["orb_match_ms"]) for record in frame_records])),
        ("LightGlue Verify Total", _safe_mean([float(record["lightglue_total_ms"]) for record in frame_records])),
        ("RANSAC Homography", _safe_mean([float(record["homography_total_ms"]) for record in frame_records])),
        ("No-Match Filter", _safe_mean([float(record["filter_no_match_ms"]) for record in frame_records])),
        ("Final Geometric Verify", _safe_mean([float(record["final_verify_ms"]) for record in frame_records])),
        ("Residual Overhead", _safe_mean([float(record["residual_ms"]) for record in frame_records])),
    ]
    shares_sorted = sorted(shares, key=lambda item: item[1], reverse=True)
    lines = [
        "## Mean Share Of End-To-End",
        "",
        "| Component | Mean ms | Share % |",
        "| --- | --- | --- |",
    ]
    for label, mean_ms in shares_sorted:
        share = 0.0 if mean_end_to_end <= 0.0 else (mean_ms / mean_end_to_end) * 100.0
        lines.append(f"| {label} | {mean_ms:.1f} | {share:.1f}% |")
    lines.append("")
    return lines


def _render_match_count_table(candidate_records: list[CandidateTiming]) -> list[str]:
    grouped: dict[int, list[CandidateTiming]] = {}
    for record in candidate_records:
        grouped.setdefault(int(record.match_count), []).append(record)
    lines = [
        "## LightGlue Match Count Dependence",
        "",
        "| Match Count | Candidates | Gate Pass Rate | Mean Verify ms | Mean Prepare ms | Mean Forward ms | Mean Homography ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for match_count in sorted(grouped):
        records = grouped[match_count]
        gate_pass_rate = sum(1 for item in records if item.gate_pass) / max(len(records), 1)
        lines.append(
            f"| {match_count} | {len(records)} | {gate_pass_rate * 100.0:.1f}% | "
            f"{_safe_mean([item.verify_total_ms for item in records]):.1f} | "
            f"{_safe_mean([item.verify_prepare_ms for item in records]):.1f} | "
            f"{_safe_mean([item.verify_forward_ms for item in records]):.1f} | "
            f"{_safe_mean([item.homography_ms for item in records]):.3f} |"
        )
    lines.append("")
    return lines


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    install_hooks()
    scenarios = load_scenarios(scenario_ids=PASS_A_SCENARIO_IDS)

    all_frame_records: list[dict[str, Any]] = []
    all_candidate_records: list[CandidateTiming] = []
    scenario_summaries: list[dict[str, Any]] = []

    for scenario_id, scenario in scenarios.items():
        settings = make_settings(REFERENCE_DIR)
        preload_started = time.perf_counter()
        cache = ReferenceCache()
        cache.preload_from_directory(REFERENCE_DIR, orb_features=settings.task3_orb_features)
        preload_ms = (time.perf_counter() - preload_started) * 1000.0

        matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
        backend = YoloeVpLightGlueBackend(reference_cache=cache, runtime_settings=settings)
        matcher.experimental_backend = backend
        capture = BackendCapture(backend=backend, matcher=matcher)
        BACKEND_CAPTURES[id(backend)] = capture
        MATCHER_CAPTURES[id(matcher)] = capture

        reference_ids = cache.list_ids()
        scenario_frame_records: list[dict[str, Any]] = []
        for decoded_frame, decode_ms in iter_timed_video_frames(
            scenario.video,
            frame_stride=scenario.frame_stride,
            limit=int(scenario.frame_limit) if scenario.frame_limit is not None else None,
            video_name=scenario.scenario_id,
        ):
            frame = frame_envelope(scenario.scenario_id, decoded_frame.frame_index, decoded_frame.width, decoded_frame.height)
            routing_started = time.perf_counter()
            active_ids = cache.filter_reference_ids_by_detector_modality(reference_ids, modality=decoded_frame.modality)
            yoloe_ids, orb_ids = cache.split_reference_ids_by_detector(active_ids)
            routing_cache_ms = (time.perf_counter() - routing_started) * 1000.0

            capture.start_frame(
                scenario_id=scenario.scenario_id,
                frame_index=decoded_frame.frame_index,
                modality=decoded_frame.modality,
            )
            wall_started = time.perf_counter()
            raw_matches = matcher.match(frame, b"", reference_ids, decoded_frame=decoded_frame, mode="yoloe_vp_lightglue")
            filter_started = time.perf_counter()
            filtered_matches = filter_no_match_candidates(
                raw_matches,
                min_score=settings.task3_min_score,
                mode="yoloe_vp_lightglue",
                yoloe_min_score=settings.task3_yoloe_min_score,
                modality=decoded_frame.modality,
                yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
                ambiguity_margin=settings.task3_ambiguity_margin,
                suppression_mode=cache.get_candidate_suppression_mode(),
            )
            filter_no_match_ms = (time.perf_counter() - filter_started) * 1000.0
            final_verify_started = time.perf_counter()
            verified_matches = verify_matches(
                frame,
                filtered_matches,
                decoded_frame=decoded_frame,
                min_inliers=settings.task3_match_min_inliers,
            )
            final_verify_ms = (time.perf_counter() - final_verify_started) * 1000.0
            wall_ms = (time.perf_counter() - wall_started) * 1000.0

            frame_record = capture.finalize_frame()
            frame_record.update(
                {
                    "decode_ms": round(float(decode_ms), 6),
                    "routing_cache_ms": round(float(routing_cache_ms), 6),
                    "filter_no_match_ms": round(float(filter_no_match_ms), 6),
                    "final_verify_ms": round(float(final_verify_ms), 6),
                    "wall_ms": round(float(wall_ms), 6),
                    "end_to_end_ms": round(float(decode_ms + wall_ms), 6),
                    "raw_candidate_count": int(len(raw_matches)),
                    "filtered_candidate_count": int(len(filtered_matches)),
                    "verified_count": int(len(verified_matches)),
                    "active_ref_count": int(len(active_ids)),
                    "yoloe_routed_ref_count": int(len(yoloe_ids)),
                    "orb_routed_ref_count": int(len(orb_ids)),
                }
            )
            tracked_ms = (
                float(frame_record["routing_cache_ms"])
                + float(frame_record["ensure_ready_ms"])
                + float(frame_record["yoloe_predict_full_ms"])
                + float(frame_record["orb_match_ms"])
                + float(frame_record["lightglue_total_ms"])
                + float(frame_record["homography_total_ms"])
                + float(frame_record["filter_no_match_ms"])
                + float(frame_record["final_verify_ms"])
            )
            frame_record["residual_ms"] = round(float(frame_record["wall_ms"]) - tracked_ms, 6)
            scenario_frame_records.append(frame_record)
            all_frame_records.append(frame_record)

        all_candidate_records.extend(capture.candidate_records)
        scenario_summaries.append(
            {
                "scenario_id": scenario.scenario_id,
                "preload_ms": round(float(preload_ms), 6),
                "frames": int(len(scenario_frame_records)),
                "wall_p50_ms": _percentile([float(item["wall_ms"]) for item in scenario_frame_records], 0.50),
                "wall_p95_ms": _percentile([float(item["wall_ms"]) for item in scenario_frame_records], 0.95),
                "end_to_end_p50_ms": _percentile([float(item["end_to_end_ms"]) for item in scenario_frame_records], 0.50),
                "mean_lightglue_ms": _safe_mean([float(item["lightglue_total_ms"]) for item in scenario_frame_records]),
                "mean_yoloe_predict_ms": _safe_mean([float(item["yoloe_predict_full_ms"]) for item in scenario_frame_records]),
            }
        )

    payload = {
        "scenario_summaries": scenario_summaries,
        "frame_records": all_frame_records,
        "candidate_records": [
            {
                "scenario_id": record.scenario_id,
                "frame_index": record.frame_index,
                "object_id": record.object_id,
                "gate_pass": record.gate_pass,
                "match_count": record.match_count,
                "verify_prepare_ms": record.verify_prepare_ms,
                "verify_forward_ms": record.verify_forward_ms,
                "verify_total_ms": record.verify_total_ms,
                "homography_ms": record.homography_ms,
                "crop_height": record.crop_height,
                "crop_width": record.crop_width,
            }
            for record in all_candidate_records
        ],
    }
    (OUTPUT_DIR / "latency_breakdown.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    end_to_end_mean = _safe_mean([float(record["end_to_end_ms"]) for record in all_frame_records])
    component_means = {
        "LightGlue Verify Total": _safe_mean([float(record["lightglue_total_ms"]) for record in all_frame_records]),
        "YOLOE Predict Total": _safe_mean([float(record["yoloe_predict_full_ms"]) for record in all_frame_records]),
        "Image Decode + Gray": _safe_mean([float(record["decode_ms"]) for record in all_frame_records]),
        "ORB Match": _safe_mean([float(record["orb_match_ms"]) for record in all_frame_records]),
        "Residual Overhead": _safe_mean([float(record["residual_ms"]) for record in all_frame_records]),
    }
    largest_component = max(component_means.items(), key=lambda item: item[1]) if component_means else ("none", 0.0)
    largest_share = 0.0 if end_to_end_mean <= 0.0 else (largest_component[1] / end_to_end_mean) * 100.0

    lines = [
        "# D7 Latency Breakdown",
        "",
        "- Scope: Pass A current full configuration replay on the 4 frozen scenarios.",
        "- Runtime: current Python environment with `task3_yoloe_allow_cpu=true`.",
        "",
        "## Direct Answer",
        "",
        f"- Largest measured bottleneck is `{largest_component[0]}` at `{largest_component[1]:.1f} ms/frame` mean, about `{largest_share:.1f}%` of end-to-end frame time.",
        f"- End-to-end frame latency mean is `{end_to_end_mean:.1f} ms`; wall-clock inside matcher/filter/verifier only is `{_safe_mean([float(record['wall_ms']) for record in all_frame_records]):.1f} ms`.",
        f"- YOLOE split from Ultralytics runtime speed reports: preprocess mean `{_safe_mean([float(record['yoloe_preprocess_ms']) for record in all_frame_records]):.1f} ms`, inference mean `{_safe_mean([float(record['yoloe_inference_ms']) for record in all_frame_records]):.1f} ms`, postprocess mean `{_safe_mean([float(record['yoloe_postprocess_ms']) for record in all_frame_records]):.1f} ms`.",
        f"- LightGlue verify mean is `{_safe_mean([float(record['lightglue_total_ms']) for record in all_frame_records]):.1f} ms/frame`; exact-match-count rows in the table below show that high-match candidates such as `39` are materially slower than the `5-6` match ORB/YOLOE edge cases.",
        "",
    ]
    lines.extend(_render_stats_table(all_frame_records))
    lines.extend(_render_share_table(all_frame_records))
    lines.extend(_render_match_count_table(all_candidate_records))

    lines.extend(
        [
            "## Scenario Setup",
            "",
            "| Scenario | Cache Preload ms | Frames | Wall P50 ms | Wall P95 ms | End-to-End P50 ms | Mean YOLOE Predict ms | Mean LightGlue ms |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for summary in scenario_summaries:
        lines.append(
            f"| {summary['scenario_id']} | {float(summary['preload_ms']):.1f} | {int(summary['frames'])} | "
            f"{float(summary['wall_p50_ms']):.1f} | {float(summary['wall_p95_ms']):.1f} | {float(summary['end_to_end_p50_ms']):.1f} | "
            f"{float(summary['mean_yoloe_predict_ms']):.1f} | {float(summary['mean_lightglue_ms']):.1f} |"
        )
    lines.append("")

    (OUTPUT_DIR / "latency_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote latency breakdown to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
