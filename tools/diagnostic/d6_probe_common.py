from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, FrameEnvelope
from src.core.video_io import iter_video_frames
from src.task3.experimental.backend import (
    LightGlueCropVerifier,
    YoloeVpLightGlueBackend,
    _compute_homography_consistency,
)
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import (
    compute_mode_candidate_score,
    filter_no_match_candidates,
    normalize_yoloe_match_count,
    resolve_min_score_for_mode,
)
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches

REFERENCE_DIR = PROJECT_ROOT / "data" / "references" / "2026_baseline"
MANIFEST_PATH = PROJECT_ROOT / "data" / "task3_eval_manifest.json"
TEMP_ROOT = PROJECT_ROOT / "_logs" / "d6_temp"

REFERENCE_METADATA = {
    "ref_01": {"file": "Referans_Nesne_01.JPG", "dimensions": [3019, 2145], "modality": "rgb", "source_exif": "default"},
    "ref_02": {"file": "Referans_Nesne_02.JPG", "dimensions": [3388, 2095], "modality": "rgb", "source_exif": "default"},
    "ref_03": {"file": "Referans_Nesne_03.JPG", "dimensions": [1716, 1516], "modality": "rgb", "source_exif": "default"},
    "ref_04": {"file": "Referans_Nesne_04.JPG", "dimensions": [459, 428], "modality": "thermal", "source_exif": "whitehot"},
    "ref_05": {"file": "Referans_Nesne_05.jpg", "dimensions": [791, 688], "modality": "rgb", "source_exif": None},
    "ref_06": {"file": "Referans_Nesne_06.jpg", "dimensions": [857, 450], "modality": "rgb", "source_exif": None},
    "ref_07": {"file": "Referans_Nesne_07.png", "dimensions": [398, 210], "modality": "unknown", "source_exif": None},
    "ref_08": {"file": "Referans_Nesne_08.png", "dimensions": [544, 357], "modality": "rgb", "source_exif": None},
    "ref_09": {"file": "Referans_Nesne_09.png", "dimensions": [157, 198], "modality": "rgb", "source_exif": None},
    "ref_10": {"file": "Referans_Nesne_10.png", "dimensions": [161, 272], "modality": "rgb", "source_exif": None},
    "ref_11": {"file": "Referans_Nesne_11.png", "dimensions": [978, 576], "modality": "thermal", "source_exif": None},
    "ref_12": {"file": "Referans_Nesne_12.png", "dimensions": [402, 189], "modality": "thermal", "source_exif": None},
}

VERIFY_CAPTURE_REGISTRY: dict[int, "YoloeFrameCapture"] = {}
BACKEND_CAPTURE_REGISTRY: dict[int, "YoloeFrameCapture"] = {}
GLOBAL_HOOKS_INSTALLED = False
ORIGINAL_VERIFY = LightGlueCropVerifier.verify
ORIGINAL_REJECT = YoloeVpLightGlueBackend._dump_rejected_candidate
ORIGINAL_POST = YoloeVpLightGlueBackend._dump_post_gate_candidate


@dataclass(slots=True)
class YoloeProbeRecord:
    scenario: str
    frame_idx: int
    candidate_idx: int
    object_id: str
    gate_pass: bool
    yoloe_confidence: float
    match_count: int
    normalized_matches: float
    inlier_count: int
    inlier_ratio: float
    homography_found: bool
    final_score: float
    score_det_term: float
    score_match_term: float
    score_inlier_term: float
    bbox: list[int] | None = None
    crop_hw: list[int] | None = None


class YoloeFrameCapture:
    def __init__(self, backend: YoloeVpLightGlueBackend, settings: MvpRuntimeSettings) -> None:
        self.backend = backend
        self.settings = settings
        self.records: list[YoloeProbeRecord] = []
        self._pending_verify: deque[dict[str, Any]] = deque()
        self._installed = False

    def ensure_installed(self) -> None:
        if self._installed:
            return
        if self.backend.verifier is None:
            raise RuntimeError("backend verifier not ready")
        install_global_hooks()
        VERIFY_CAPTURE_REGISTRY[id(self.backend.verifier)] = self
        BACKEND_CAPTURE_REGISTRY[id(self.backend)] = self
        self._installed = True

    def start_frame(self) -> None:
        self.records = []
        self._pending_verify.clear()


def install_global_hooks() -> None:
    global GLOBAL_HOOKS_INSTALLED
    if GLOBAL_HOOKS_INSTALLED:
        return

    def verify_wrapper(
        verifier: LightGlueCropVerifier,
        crop_bgr: Any,
        ref_features: dict[str, Any],
    ) -> tuple[bool, int, float, dict[str, Any] | None]:
        passed, match_count, verify_ms, kpts_data = ORIGINAL_VERIFY(verifier, crop_bgr, ref_features)
        capture = VERIFY_CAPTURE_REGISTRY.get(id(verifier))
        if capture is not None:
            capture._pending_verify.append(
                {
                    "passed": bool(passed),
                    "match_count": int(match_count),
                    "verify_ms": float(verify_ms),
                    "kpts_data": kpts_data,
                }
            )
        return passed, match_count, verify_ms, kpts_data

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
        capture = BACKEND_CAPTURE_REGISTRY.get(id(backend))
        if capture is not None:
            verify_info = capture._pending_verify.popleft() if capture._pending_verify else None
            raw_match_count = int(match_count)
            kpts_data = verify_info.get("kpts_data") if verify_info is not None else None
            inlier_count, inlier_ratio, homography_found, _homography_ms = _compute_homography_consistency(
                kpts_data,
                reproj_threshold=capture.settings.task3_yoloe_homography_ransac_reproj_threshold,
            )
            normalized_matches = normalize_yoloe_match_count(
                match_count=raw_match_count,
                normalization_scale=capture.settings.task3_yoloe_match_normalization_scale,
            )
            det_term = float(yoloe_confidence) * float(capture.settings.task3_yoloe_score_confidence_weight)
            match_term = normalized_matches * float(capture.settings.task3_yoloe_score_matches_weight)
            inlier_term = float(inlier_ratio) * float(capture.settings.task3_yoloe_score_inlier_weight)
            final_score = compute_mode_candidate_score(
                confidence=float(yoloe_confidence),
                normalized_matches=normalized_matches,
                inlier_ratio=float(inlier_ratio),
                mode="yoloe_vp_lightglue",
                yoloe_confidence_weight=capture.settings.task3_yoloe_score_confidence_weight,
                yoloe_matches_weight=capture.settings.task3_yoloe_score_matches_weight,
                yoloe_inlier_weight=capture.settings.task3_yoloe_score_inlier_weight,
            )
            capture.records.append(
                YoloeProbeRecord(
                    scenario=str(scenario_id or ""),
                    frame_idx=int(frame_idx),
                    candidate_idx=int(candidate_idx),
                    object_id=str(object_id),
                    gate_pass=False,
                    yoloe_confidence=float(yoloe_confidence),
                    match_count=raw_match_count,
                    normalized_matches=float(normalized_matches),
                    inlier_count=int(inlier_count),
                    inlier_ratio=float(inlier_ratio),
                    homography_found=bool(homography_found),
                    final_score=float(final_score),
                    score_det_term=float(det_term),
                    score_match_term=float(match_term),
                    score_inlier_term=float(inlier_term),
                    bbox=[int(value) for value in bbox],
                    crop_hw=[0, 0] if crop_bgr is None else [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])],
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
        capture = BACKEND_CAPTURE_REGISTRY.get(id(backend))
        if capture is not None:
            if capture._pending_verify:
                capture._pending_verify.popleft()
            det_term = float(yoloe_confidence) * float(capture.settings.task3_yoloe_score_confidence_weight)
            match_term = float(normalized_matches) * float(capture.settings.task3_yoloe_score_matches_weight)
            inlier_term = float(inlier_ratio) * float(capture.settings.task3_yoloe_score_inlier_weight)
            capture.records.append(
                YoloeProbeRecord(
                    scenario=str(scenario_id or ""),
                    frame_idx=int(frame_idx),
                    candidate_idx=int(candidate_idx),
                    object_id=str(object_id),
                    gate_pass=True,
                    yoloe_confidence=float(yoloe_confidence),
                    match_count=int(match_count),
                    normalized_matches=float(normalized_matches),
                    inlier_count=int(inlier_count),
                    inlier_ratio=float(inlier_ratio),
                    homography_found=bool(homography_found),
                    final_score=float(score),
                    score_det_term=float(det_term),
                    score_match_term=float(match_term),
                    score_inlier_term=float(inlier_term),
                    bbox=[int(value) for value in bbox],
                    crop_hw=[0, 0] if crop_bgr is None else [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])],
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

    LightGlueCropVerifier.verify = verify_wrapper  # type: ignore[assignment]
    YoloeVpLightGlueBackend._dump_rejected_candidate = reject_wrapper  # type: ignore[assignment]
    YoloeVpLightGlueBackend._dump_post_gate_candidate = post_wrapper  # type: ignore[assignment]
    GLOBAL_HOOKS_INSTALLED = True


def load_manifest_map() -> dict[str, dict[str, Any]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in payload.get("scenarios", [])}


def make_settings(reference_dir: str | Path) -> MvpRuntimeSettings:
    return MvpRuntimeSettings(
        task3_mode="yoloe_vp_lightglue",
        task3_reference_dir=Path(reference_dir),
        task3_eval_reference_dir=Path(reference_dir),
        task3_yoloe_allow_cpu=True,
    )


def _frame_envelope(scenario_id: str, decoded_frame: Any, ordinal: int) -> FrameEnvelope:
    return FrameEnvelope(
        frame_url=f"http://task3-d6/frames/{ordinal + 1}/",
        image_url=f"/task3-d6/{ordinal + 1}.jpg",
        video_name=scenario_id,
        translation_x=0.0,
        translation_y=0.0,
        translation_z=0.0,
        health_status="1",
        metadata={
            "frame_index": decoded_frame.frame_index,
            "image_width": decoded_frame.width,
            "image_height": decoded_frame.height,
        },
    )


def serialize_candidate(candidate: CanonicalUndefinedObject) -> dict[str, Any]:
    metadata = dict(candidate.metadata)
    task3_yoloe = metadata.get("task3_yoloe")
    if isinstance(task3_yoloe, dict):
        metadata["task3_yoloe"] = dict(task3_yoloe)
    return {
        "object_id": str(candidate.object_id),
        "bbox": [
            float(candidate.top_left_x),
            float(candidate.top_left_y),
            float(candidate.bottom_right_x),
            float(candidate.bottom_right_y),
        ],
        "matcher_source": str(metadata.get("matcher_source", "")),
        "match_score": float(metadata.get("match_score", 0.0)),
        "match_count": int(metadata.get("match_count", 0)),
        "inlier_count": int(metadata.get("inlier_count", 0)),
        "inlier_ratio": float(metadata.get("inlier_ratio", 0.0)),
        "verification_status": metadata.get("verification_status"),
        "bbox_sane": metadata.get("bbox_sane"),
        "scale_ok": metadata.get("scale_ok"),
        "similarity": metadata.get("similarity"),
        "corroboration": metadata.get("corroboration"),
        "task3_yoloe": metadata.get("task3_yoloe"),
    }


def serialize_yoloe_record(record: YoloeProbeRecord) -> dict[str, Any]:
    return {
        "scenario": record.scenario,
        "frame_idx": record.frame_idx,
        "candidate_idx": record.candidate_idx,
        "object_id": record.object_id,
        "gate_pass": record.gate_pass,
        "yoloe_confidence": record.yoloe_confidence,
        "match_count": record.match_count,
        "normalized_matches": record.normalized_matches,
        "inlier_count": record.inlier_count,
        "inlier_ratio": record.inlier_ratio,
        "homography_found": record.homography_found,
        "final_score": record.final_score,
        "score_det_term": record.score_det_term,
        "score_match_term": record.score_match_term,
        "score_inlier_term": record.score_inlier_term,
        "bbox": record.bbox,
        "crop_hw": record.crop_hw,
    }


def evaluate_frame(
    scenario_id: str,
    target_frame_index: int,
    *,
    reference_dir: str | Path,
) -> dict[str, Any]:
    settings = make_settings(reference_dir)
    manifest_map = load_manifest_map()
    if scenario_id not in manifest_map:
        raise KeyError(f"Scenario not found: {scenario_id}")
    scenario = manifest_map[scenario_id]

    cache = ReferenceCache()
    cache.preload_from_directory(Path(reference_dir), orb_features=settings.task3_orb_features)
    matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
    backend_captures: dict[int, YoloeFrameCapture] = {}
    threshold = resolve_min_score_for_mode(
        mode="yoloe_vp_lightglue",
        min_score=settings.task3_min_score,
        yoloe_min_score=settings.task3_yoloe_min_score,
        modality="thermal" if "thermal" in scenario_id else "rgb",
        yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
    )

    frames = iter_video_frames(
        scenario["video"],
        frame_stride=int(scenario["frame_stride"]),
        limit=int(scenario["frame_limit"]) if scenario.get("frame_limit") is not None else None,
        video_name=scenario_id,
    )
    for ordinal, decoded_frame in enumerate(frames):
        if int(decoded_frame.frame_index) != int(target_frame_index):
            continue

        frame = _frame_envelope(scenario_id, decoded_frame, ordinal)
        available_ids = cache.list_ids()
        active_ids = cache.filter_reference_ids_by_detector_modality(available_ids, modality=decoded_frame.modality)
        yoloe_ids, orb_ids = cache.split_reference_ids_by_detector(active_ids)

        yoloe_capture: YoloeFrameCapture | None = None
        if yoloe_ids:
            if matcher.experimental_backend is None:
                matcher.experimental_backend = YoloeVpLightGlueBackend(reference_cache=cache, runtime_settings=settings)
            backend = matcher.experimental_backend
            backend._ensure_ready(yoloe_ids)
            backend_key = id(backend)
            yoloe_capture = backend_captures.get(backend_key)
            if yoloe_capture is None:
                yoloe_capture = YoloeFrameCapture(backend, settings)
                yoloe_capture.ensure_installed()
                backend_captures[backend_key] = yoloe_capture
            yoloe_capture.start_frame()

        start = time.perf_counter()
        raw_matches = matcher.match(frame, b"", available_ids, decoded_frame=decoded_frame, mode="yoloe_vp_lightglue")
        task3_info = dict(matcher.last_run_info)
        filtered = filter_no_match_candidates(
            raw_matches,
            min_score=settings.task3_min_score,
            mode="yoloe_vp_lightglue",
            yoloe_min_score=settings.task3_yoloe_min_score,
            modality=decoded_frame.modality,
            yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
            ambiguity_margin=settings.task3_ambiguity_margin,
            suppression_mode=cache.get_candidate_suppression_mode(),
        )
        verified = verify_matches(
            frame,
            filtered,
            decoded_frame=decoded_frame,
            min_inliers=settings.task3_match_min_inliers,
        )
        wall_ms = (time.perf_counter() - start) * 1000.0
        yoloe_records = [] if yoloe_capture is None else [serialize_yoloe_record(record) for record in yoloe_capture.records]
        return {
            "scenario_id": scenario_id,
            "frame_index": int(decoded_frame.frame_index),
            "modality": str(decoded_frame.modality),
            "frame_width": int(decoded_frame.width),
            "frame_height": int(decoded_frame.height),
            "wall_ms": round(wall_ms, 6),
            "threshold": float(
                resolve_min_score_for_mode(
                    mode="yoloe_vp_lightglue",
                    min_score=settings.task3_min_score,
                    yoloe_min_score=settings.task3_yoloe_min_score,
                    modality=decoded_frame.modality,
                    yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
                )
            ),
            "suppression_mode": cache.get_candidate_suppression_mode(),
            "active_ids": active_ids,
            "yoloe_routed_refs": yoloe_ids,
            "orb_routed_refs": orb_ids,
            "auto_routing": cache.get_auto_routing_summary(),
            "task3_info": task3_info,
            "raw_matches": [serialize_candidate(item) for item in raw_matches],
            "filtered_matches": [serialize_candidate(item) for item in filtered],
            "verified_matches": [serialize_candidate(item) for item in verified],
            "yoloe_records": yoloe_records,
        }
    raise RuntimeError(f"Target frame {target_frame_index} not reached in scenario {scenario_id}")


def build_temp_reference_bank(
    *,
    selected_ref_ids: list[str],
    per_reference_suppression: bool,
    overrides: dict[str, dict[str, Any]] | None = None,
    suffix: str,
) -> Path:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{suffix}_", dir=str(TEMP_ROOT)))
    for reference_id in selected_ref_ids:
        metadata = REFERENCE_METADATA[reference_id]
        source = REFERENCE_DIR / str(metadata["file"])
        target = temp_dir / str(metadata["file"])
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    spec_payload = {
        "version": "d6_probe_v1",
        "created": "2026-05-02",
        "source": "temporary D6 diagnostic reference bank",
        "per_reference_suppression": bool(per_reference_suppression),
        "references": {
            reference_id: {
                "file": REFERENCE_METADATA[reference_id]["file"],
                "dimensions": list(REFERENCE_METADATA[reference_id]["dimensions"]),
                "modality": REFERENCE_METADATA[reference_id]["modality"],
                "source_exif": REFERENCE_METADATA[reference_id]["source_exif"],
            }
            for reference_id in selected_ref_ids
        },
        "overrides": overrides or {},
    }
    manifest_payload = {
        "version": "d6_probe_v1",
        "source": "temporary D6 diagnostic reference bank",
        "spec_path": "spec.json",
        "reference_count": len(selected_ref_ids),
        "items": [
            {
                "reference_id": reference_id,
                "output_file": REFERENCE_METADATA[reference_id]["file"],
                "dimensions": list(REFERENCE_METADATA[reference_id]["dimensions"]),
                "modality": REFERENCE_METADATA[reference_id]["modality"],
                "source_file": REFERENCE_METADATA[reference_id]["file"],
            }
            for reference_id in selected_ref_ids
        ],
    }
    (temp_dir / "spec.json").write_text(json.dumps(spec_payload, indent=2), encoding="utf-8")
    (temp_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return temp_dir


def cleanup_temp_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def reference_aspect_ratio(reference_id: str) -> float:
    width, height = REFERENCE_METADATA[reference_id]["dimensions"]
    return float(width) / float(max(height, 1))
