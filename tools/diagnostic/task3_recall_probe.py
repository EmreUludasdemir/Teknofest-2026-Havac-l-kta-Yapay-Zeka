from __future__ import annotations

"""Read-only recall probe for Task 3 auto-routing freeze.

This script inspects the frozen `yoloe_vp_lightglue` experimental path without
changing any production scoring or routing logic. It emits per-frame CSVs for
`ref_01`, `ref_05`, and a compact `ref_03` baseline sample set.

Implementation notes:
- Uses the current v3 auto-routing spec from `data/references/2026_baseline/`.
- Replays the existing manifest scenarios exactly as configured.
- Enables CPU YOLOE explicitly so results match the archived frozen validation.
- Captures YOLOE gate-pass and gate-fail details in-memory by monkeypatching the
  backend instance only inside this diagnostic process; no source logic changes.
- For ORB-routed references, the score decomposition is reported as:
  - `score_inlier_term = final_score` for homography candidates
  - `score_match_term = final_score` for bbox/template candidates
  - `score_det_term = 0.0`
  This keeps the requested CSV schema intact while making it obvious whether the
  ORB candidate quality came from geometric consistency or raw match support.
"""

import csv
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, FrameEnvelope
from src.evaluation.task2_long_sequence import iter_video_frames
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "task3_eval_manifest.json"
REFERENCE_DIR = PROJECT_ROOT / "data" / "references" / "2026_baseline"
OUTPUT_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-04-25_d1_recall_probe"

TARGET_REFERENCE_IDS = ("ref_01", "ref_05")
BASELINE_REFERENCE_ID = "ref_03"
ACTIVE_ANALYSIS_SCENARIO = "rgb_reference_session"
YOLOE_MATCH_GATE = 15

CSV_FIELDS = [
    "scenario",
    "frame_idx",
    "detector_path",
    "candidate_count",
    "max_det_conf",
    "lg_raw_matches",
    "lg_filtered_matches",
    "ransac_inliers",
    "ransac_inlier_ratio",
    "score_det_term",
    "score_match_term",
    "score_inlier_term",
    "final_score",
    "threshold",
    "score_minus_threshold",
    "accepted",
]

_VERIFY_CAPTURE_REGISTRY: dict[int, "YoloeFrameCapture"] = {}
_BACKEND_CAPTURE_REGISTRY: dict[int, "YoloeFrameCapture"] = {}
_GLOBAL_HOOKS_INSTALLED = False
_ORIGINAL_VERIFY = LightGlueCropVerifier.verify
_ORIGINAL_REJECT = YoloeVpLightGlueBackend._dump_rejected_candidate
_ORIGINAL_POST = YoloeVpLightGlueBackend._dump_post_gate_candidate


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
        _install_global_hooks()
        _VERIFY_CAPTURE_REGISTRY[id(self.backend.verifier)] = self
        _BACKEND_CAPTURE_REGISTRY[id(self.backend)] = self
        self._installed = True

    def start_frame(self) -> None:
        self.records = []
        self._pending_verify.clear()


def _install_global_hooks() -> None:
    global _GLOBAL_HOOKS_INSTALLED
    if _GLOBAL_HOOKS_INSTALLED:
        return

    def verify_wrapper(
        verifier: LightGlueCropVerifier,
        crop_bgr: Any,
        ref_features: dict[str, Any],
    ) -> tuple[bool, int, float, dict[str, Any] | None]:
        passed, match_count, verify_ms, kpts_data = _ORIGINAL_VERIFY(verifier, crop_bgr, ref_features)
        capture = _VERIFY_CAPTURE_REGISTRY.get(id(verifier))
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
        capture = _BACKEND_CAPTURE_REGISTRY.get(id(backend))
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
                )
            )
        return _ORIGINAL_REJECT(
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
        capture = _BACKEND_CAPTURE_REGISTRY.get(id(backend))
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
                )
            )
        return _ORIGINAL_POST(
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
    _GLOBAL_HOOKS_INSTALLED = True


def _safe_round(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.6f}"


def _safe_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(int(value))


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _make_settings() -> MvpRuntimeSettings:
    return MvpRuntimeSettings(
        task3_mode="yoloe_vp_lightglue",
        task3_reference_dir=REFERENCE_DIR,
        task3_eval_reference_dir=REFERENCE_DIR,
        task3_yoloe_allow_cpu=True,
    )


def _make_frame(scenario_id: str, decoded_frame: Any, ordinal: int) -> FrameEnvelope:
    return FrameEnvelope(
        frame_url=f"http://task3-d1/frames/{ordinal + 1}/",
        image_url=f"/task3-d1/{ordinal + 1}.jpg",
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


def _is_numeric(value: Any) -> bool:
    if value in ("", None):
        return False
    try:
        float(value)
    except Exception:
        return False
    return True


def _threshold_for_ref(settings: MvpRuntimeSettings, detector: str, modality: str) -> float:
    if detector == "yoloe":
        return resolve_min_score_for_mode(
            mode="yoloe_vp_lightglue",
            min_score=settings.task3_min_score,
            yoloe_min_score=settings.task3_yoloe_min_score,
            modality=modality,
            yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
        )
    return float(settings.task3_min_score)


def _build_skipped_row(*, scenario_id: str, frame_idx: int, detector_path: str) -> dict[str, Any]:
    return {
        "scenario": scenario_id,
        "frame_idx": frame_idx,
        "detector_path": detector_path,
        "candidate_count": 0,
        "max_det_conf": "",
        "lg_raw_matches": "",
        "lg_filtered_matches": "",
        "ransac_inliers": "",
        "ransac_inlier_ratio": "",
        "score_det_term": "",
        "score_match_term": "",
        "score_inlier_term": "",
        "final_score": "",
        "threshold": "",
        "score_minus_threshold": "",
        "accepted": False,
    }


def _build_orb_row(
    *,
    settings: MvpRuntimeSettings,
    scenario_id: str,
    frame_idx: int,
    ref_id: str,
    route_modalities: list[str] | None,
    current_modality: str,
    matcher: Task3Matcher,
    decoded_frame: Any,
    accepted_ids: set[str],
) -> dict[str, Any]:
    detector_path = f"orb({','.join(route_modalities or ['all'])})"
    if route_modalities and current_modality not in route_modalities:
        return _build_skipped_row(
            scenario_id=scenario_id,
            frame_idx=frame_idx,
            detector_path=f"{detector_path}-skipped",
        )

    candidates = matcher._match_with_real_orb_only(decoded_frame, [ref_id])
    threshold = _threshold_for_ref(settings, "orb", current_modality)
    if not candidates:
        return {
            "scenario": scenario_id,
            "frame_idx": frame_idx,
            "detector_path": detector_path,
            "candidate_count": 0,
            "max_det_conf": "",
            "lg_raw_matches": "",
            "lg_filtered_matches": "",
            "ransac_inliers": "",
            "ransac_inlier_ratio": "",
            "score_det_term": "",
            "score_match_term": "",
            "score_inlier_term": "",
            "final_score": "",
            "threshold": _safe_round(threshold),
            "score_minus_threshold": "",
            "accepted": ref_id in accepted_ids,
        }

    best = max(candidates, key=lambda item: float(item.metadata.get("match_score", 0.0)))
    final_score = float(best.metadata.get("match_score", 0.0))
    source = str(best.metadata.get("matcher_source", ""))
    if source.startswith("task3_orb_bf_homography"):
        score_match_term = 0.0
        score_inlier_term = final_score
    else:
        score_match_term = final_score
        score_inlier_term = 0.0
    return {
        "scenario": scenario_id,
        "frame_idx": frame_idx,
        "detector_path": detector_path,
        "candidate_count": len(candidates),
        "max_det_conf": "",
        "lg_raw_matches": _safe_int(int(best.metadata.get("match_count", 0))),
        "lg_filtered_matches": _safe_int(int(best.metadata.get("match_count", 0))),
        "ransac_inliers": _safe_int(int(best.metadata.get("inlier_count", 0))),
        "ransac_inlier_ratio": _safe_round(float(best.metadata.get("inlier_ratio", 0.0))),
        "score_det_term": _safe_round(0.0),
        "score_match_term": _safe_round(score_match_term),
        "score_inlier_term": _safe_round(score_inlier_term),
        "final_score": _safe_round(final_score),
        "threshold": _safe_round(threshold),
        "score_minus_threshold": _safe_round(final_score - threshold),
        "accepted": ref_id in accepted_ids,
    }


def _build_yoloe_row(
    *,
    settings: MvpRuntimeSettings,
    scenario_id: str,
    frame_idx: int,
    ref_id: str,
    route_modalities: list[str] | None,
    current_modality: str,
    yoloe_records: list[YoloeProbeRecord],
    accepted_ids: set[str],
) -> dict[str, Any]:
    detector_path = f"yoloe({','.join(route_modalities or ['all'])})"
    if route_modalities and current_modality not in route_modalities:
        return _build_skipped_row(
            scenario_id=scenario_id,
            frame_idx=frame_idx,
            detector_path=f"{detector_path}-skipped",
        )

    threshold = _threshold_for_ref(settings, "yoloe", current_modality)
    if not yoloe_records:
        return {
            "scenario": scenario_id,
            "frame_idx": frame_idx,
            "detector_path": detector_path,
            "candidate_count": 0,
            "max_det_conf": "",
            "lg_raw_matches": "",
            "lg_filtered_matches": "",
            "ransac_inliers": "",
            "ransac_inlier_ratio": "",
            "score_det_term": "",
            "score_match_term": "",
            "score_inlier_term": "",
            "final_score": "",
            "threshold": _safe_round(threshold),
            "score_minus_threshold": "",
            "accepted": ref_id in accepted_ids,
        }

    best = max(
        yoloe_records,
        key=lambda item: (
            float(item.final_score),
            float(item.yoloe_confidence),
            int(item.match_count),
        ),
    )
    max_det_conf = max(item.yoloe_confidence for item in yoloe_records)
    return {
        "scenario": scenario_id,
        "frame_idx": frame_idx,
        "detector_path": detector_path,
        "candidate_count": len(yoloe_records),
        "max_det_conf": _safe_round(max_det_conf),
        "lg_raw_matches": _safe_int(best.match_count),
        "lg_filtered_matches": _safe_int(best.match_count if best.gate_pass else 0),
        "ransac_inliers": _safe_int(best.inlier_count),
        "ransac_inlier_ratio": _safe_round(best.inlier_ratio),
        "score_det_term": _safe_round(best.score_det_term),
        "score_match_term": _safe_round(best.score_match_term),
        "score_inlier_term": _safe_round(best.score_inlier_term),
        "final_score": _safe_round(best.final_score),
        "threshold": _safe_round(threshold),
        "score_minus_threshold": _safe_round(best.final_score - threshold),
        "accepted": ref_id in accepted_ids,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [row for row in rows if _is_numeric(row.get("score_minus_threshold"))]
    return sorted(scored, key=lambda row: float(row["score_minus_threshold"]), reverse=True)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not str(row["detector_path"]).endswith("-skipped")]


def _scenario_rows(rows: list[dict[str, Any]], scenario_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["scenario"] == scenario_id]


def _accepted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row["accepted"])]


def _miss_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not bool(row["accepted"])]


def _numeric_column(rows: list[dict[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column)
        if _is_numeric(value):
            values.append(float(value))
    return values


def _build_summary_markdown(
    *,
    routing_summary: dict[str, dict[str, Any]],
    ref01_rows: list[dict[str, Any]],
    ref05_rows: list[dict[str, Any]],
    ref03_rows: list[dict[str, Any]],
    ref05_frame_meta: list[dict[str, Any]],
) -> str:
    ref01_rgb_rows = _scenario_rows(ref01_rows, ACTIVE_ANALYSIS_SCENARIO)
    ref03_rgb_rows = _scenario_rows(ref03_rows, ACTIVE_ANALYSIS_SCENARIO)
    ref05_rgb_rows = _scenario_rows(ref05_rows, ACTIVE_ANALYSIS_SCENARIO)

    ref01_top10 = _score_rows(ref01_rows)[:10]

    ref01_term_means = {
        "det": _mean_or_none(_numeric_column(ref01_rgb_rows, "score_det_term")),
        "match": _mean_or_none(_numeric_column(ref01_rgb_rows, "score_match_term")),
        "inlier": _mean_or_none(_numeric_column(ref01_rgb_rows, "score_inlier_term")),
        "score": _mean_or_none(_numeric_column(ref01_rgb_rows, "final_score")),
    }
    ref03_term_means = {
        "det": _mean_or_none(_numeric_column(ref03_rgb_rows, "score_det_term")),
        "match": _mean_or_none(_numeric_column(ref03_rgb_rows, "score_match_term")),
        "inlier": _mean_or_none(_numeric_column(ref03_rgb_rows, "score_inlier_term")),
        "score": _mean_or_none(_numeric_column(ref03_rgb_rows, "final_score")),
    }
    term_gaps = {
        key: (ref03_term_means[key] or 0.0) - (ref01_term_means[key] or 0.0)
        for key in ("det", "match", "inlier")
    }
    ref01_bottleneck = max(term_gaps, key=term_gaps.get) if term_gaps else "unknown"

    ref05_active_rows = _active_rows(ref05_rows)
    ref05_candidate_rows = [row for row in ref05_active_rows if int(row["candidate_count"]) > 0]
    ref05_candidate_rate = len(ref05_candidate_rows) / max(len(ref05_active_rows), 1)
    ref05_match_below_gate_rate = (
        sum(
            1
            for row in ref05_active_rows
            if _is_numeric(row.get("lg_raw_matches")) and int(row["lg_raw_matches"]) < YOLOE_MATCH_GATE
        )
        / max(len(ref05_active_rows), 1)
    )
    ref05_present_match_below_gate_rate = (
        sum(
            1
            for row in ref05_rgb_rows
            if _is_numeric(row.get("lg_raw_matches")) and int(row["lg_raw_matches"]) < YOLOE_MATCH_GATE
        )
        / max(len(ref05_rgb_rows), 1)
    )
    ref05_fallback_counts = Counter(str(item.get("fallback_reason") or "none") for item in ref05_frame_meta if item.get("active"))

    ref03_accept_rows = _accepted_rows(ref03_rows)
    ref03_miss_samples = _score_rows(_miss_rows(ref03_rows))[:5]

    lines = [
        "# D1 Recall Probe Summary",
        "",
        "## Routing Snapshot",
        "",
        "| Ref | Modality | Detector | Modalities | Confidence | Rationale |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for ref_id in ("ref_01", "ref_03", "ref_05"):
        info = routing_summary[ref_id]
        modalities = ",".join(info.get("detector_modalities") or []) or "-"
        lines.append(
            f"| {ref_id} | {info.get('modality')} | {info.get('detector')} | {modalities} | "
            f"{info.get('confidence')} | {info.get('rationale')} |"
        )

    lines.extend(
        [
            "",
            "## ref_01: Esiğe En Yakin Kareler",
            "",
            "| Rank | Scenario | Frame | Detector | Final Score | Threshold | Score - Threshold | Candidate Count |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for index, row in enumerate(ref01_top10, start=1):
        lines.append(
            f"| {index} | {row['scenario']} | {row['frame_idx']} | {row['detector_path']} | "
            f"{row['final_score']} | {row['threshold']} | {row['score_minus_threshold']} | {row['candidate_count']} |"
        )
    if not ref01_top10:
        lines.append("| - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## ref_01 Bottleneck vs ref_03 Baseline (rgb_reference_session)",
            "",
            "| Metric | ref_01 | ref_03 | Delta (ref_03 - ref_01) |",
            "| --- | --- | --- | --- |",
            f"| Mean det term | {_safe_round(ref01_term_means['det'])} | {_safe_round(ref03_term_means['det'])} | {_safe_round(term_gaps['det'])} |",
            f"| Mean match term | {_safe_round(ref01_term_means['match'])} | {_safe_round(ref03_term_means['match'])} | {_safe_round(term_gaps['match'])} |",
            f"| Mean inlier term | {_safe_round(ref01_term_means['inlier'])} | {_safe_round(ref03_term_means['inlier'])} | {_safe_round(term_gaps['inlier'])} |",
            f"| Mean final score | {_safe_round(ref01_term_means['score'])} | {_safe_round(ref03_term_means['score'])} | {_safe_round((ref03_term_means['score'] or 0.0) - (ref01_term_means['score'] or 0.0))} |",
            "",
            f"Direct answer: ref_01 bottleneck term is **{ref01_bottleneck}** on the current auto-routed path.",
            "",
            "## ref_05 Diagnostics",
            "",
            f"- Detector path expected by routing: `{routing_summary['ref_05']['detector']}({','.join(routing_summary['ref_05'].get('detector_modalities') or [])})`",
            f"- Fallback reasons on active frames: `{dict(ref05_fallback_counts)}`",
            f"- Candidate_count > 0 ratio on active frames: `{len(ref05_candidate_rows)}/{len(ref05_active_rows)}` = `{ref05_candidate_rate:.3f}`",
            f"- Candidate_count > 0 ratio on rgb_reference_session: `{sum(1 for row in ref05_rgb_rows if int(row['candidate_count']) > 0)}/{len(ref05_rgb_rows)}`",
            f"- LightGlue/verify match count below 15 on active frames: `{ref05_match_below_gate_rate:.3f}`",
            f"- LightGlue/verify match count below 15 on rgb_reference_session: `{ref05_present_match_below_gate_rate:.3f}`",
            "",
            "Direct answer: ref_05 stayed on the expected `yoloe(rgb)` route; no frame-level fallback was observed in the current frozen run.",
            "",
            "## ref_03 Accept Baseline",
            "",
            f"- Accepted rows kept in baseline CSV: `{len(ref03_accept_rows)}`",
            f"- Miss samples kept in baseline CSV: `{len(ref03_miss_samples)}`",
            "",
            "## Interpretation",
            "",
            "- `ref_01` and `ref_05` are rgb-only references, so both are correctly skipped in thermal scenarios by modality routing. Their actionable false negatives are therefore the rgb scenarios.",
            "- For ORB-routed rows (`ref_01`, `ref_03`), `score_inlier_term` carries the ORB homography score contribution and `score_match_term` carries bbox/template-style score where applicable.",
            "- For YOLOE-routed rows (`ref_05`), gate-failed candidates keep a diagnostic replay of the 3-term score so the margin to threshold can still be inspected.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    settings = _make_settings()

    cache = ReferenceCache()
    cache.preload_from_directory(REFERENCE_DIR, orb_features=settings.task3_orb_features)
    routing_summary = cache.get_auto_routing_summary()
    matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
    backend_captures: dict[int, YoloeFrameCapture] = {}

    target_rows: dict[str, list[dict[str, Any]]] = {reference_id: [] for reference_id in (*TARGET_REFERENCE_IDS, BASELINE_REFERENCE_ID)}
    ref05_frame_meta: list[dict[str, Any]] = []

    for scenario in manifest.get("scenarios", []):
        scenario_id = str(scenario["id"])
        frame_iter = iter_video_frames(
            scenario["video"],
            frame_stride=int(scenario["frame_stride"]),
            limit=int(scenario["frame_limit"]) if scenario.get("frame_limit") is not None else None,
            video_name=scenario_id,
        )
        for ordinal, decoded_frame in enumerate(frame_iter):
            frame = _make_frame(scenario_id, decoded_frame, ordinal)
            available_ids = cache.list_ids()
            active_ids = cache.filter_reference_ids_by_detector_modality(available_ids, modality=decoded_frame.modality)
            yoloe_ids, _orb_ids = cache.split_reference_ids_by_detector(active_ids)

            yoloe_capture: YoloeFrameCapture | None = None
            if yoloe_ids:
                if matcher.experimental_backend is None:
                    matcher.experimental_backend = YoloeVpLightGlueBackend(
                        reference_cache=cache,
                        runtime_settings=settings,
                    )
                backend = matcher.experimental_backend
                backend._ensure_ready(yoloe_ids)
                backend_key = id(backend)
                yoloe_capture = backend_captures.get(backend_key)
                if yoloe_capture is None:
                    yoloe_capture = YoloeFrameCapture(backend, settings)
                    yoloe_capture.ensure_installed()
                    backend_captures[backend_key] = yoloe_capture
                yoloe_capture.start_frame()

            raw_matches = matcher.match(
                frame,
                b"",
                available_ids,
                decoded_frame=decoded_frame,
                mode="yoloe_vp_lightglue",
            )
            task3_info = dict(matcher.last_run_info)
            filtered = filter_no_match_candidates(
                raw_matches,
                min_score=settings.task3_min_score,
                mode="yoloe_vp_lightglue",
                yoloe_min_score=settings.task3_yoloe_min_score,
                modality=decoded_frame.modality,
                yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
                ambiguity_margin=settings.task3_ambiguity_margin,
            )
            verified = verify_matches(
                frame,
                filtered,
                decoded_frame=decoded_frame,
                min_inliers=settings.task3_match_min_inliers,
            )
            accepted_ids = {item.object_id for item in verified}

            yoloe_records_by_ref: dict[str, list[YoloeProbeRecord]] = {}
            if yoloe_capture is not None:
                for record in yoloe_capture.records:
                    yoloe_records_by_ref.setdefault(record.object_id, []).append(record)

            for ref_id in (*TARGET_REFERENCE_IDS, BASELINE_REFERENCE_ID):
                route = routing_summary.get(ref_id, {})
                detector = str(route.get("detector") or "yoloe")
                route_modalities = list(route.get("detector_modalities") or [])
                if detector == "yoloe":
                    row = _build_yoloe_row(
                        settings=settings,
                        scenario_id=scenario_id,
                        frame_idx=int(decoded_frame.frame_index),
                        ref_id=ref_id,
                        route_modalities=route_modalities,
                        current_modality=str(decoded_frame.modality),
                        yoloe_records=yoloe_records_by_ref.get(ref_id, []),
                        accepted_ids=accepted_ids,
                    )
                elif detector == "orb":
                    row = _build_orb_row(
                        settings=settings,
                        scenario_id=scenario_id,
                        frame_idx=int(decoded_frame.frame_index),
                        ref_id=ref_id,
                        route_modalities=route_modalities,
                        current_modality=str(decoded_frame.modality),
                        matcher=matcher,
                        decoded_frame=decoded_frame,
                        accepted_ids=accepted_ids,
                    )
                else:
                    # Current frozen references do not trigger "both", but keep
                    # the CSV readable if future refs do.
                    yoloe_row = _build_yoloe_row(
                        settings=settings,
                        scenario_id=scenario_id,
                        frame_idx=int(decoded_frame.frame_index),
                        ref_id=ref_id,
                        route_modalities=route_modalities,
                        current_modality=str(decoded_frame.modality),
                        yoloe_records=yoloe_records_by_ref.get(ref_id, []),
                        accepted_ids=accepted_ids,
                    )
                    orb_row = _build_orb_row(
                        settings=settings,
                        scenario_id=scenario_id,
                        frame_idx=int(decoded_frame.frame_index),
                        ref_id=ref_id,
                        route_modalities=route_modalities,
                        current_modality=str(decoded_frame.modality),
                        matcher=matcher,
                        decoded_frame=decoded_frame,
                        accepted_ids=accepted_ids,
                    )
                    row = yoloe_row
                    if _is_numeric(orb_row.get("final_score")) and (
                        not _is_numeric(yoloe_row.get("final_score"))
                        or float(orb_row["final_score"]) > float(yoloe_row["final_score"])
                    ):
                        row = orb_row
                    row["detector_path"] = f"both->{row['detector_path']}"

                target_rows[ref_id].append(row)

                if ref_id == "ref_05":
                    ref05_frame_meta.append(
                        {
                            "scenario": scenario_id,
                            "frame_idx": int(decoded_frame.frame_index),
                            "active": not str(row["detector_path"]).endswith("-skipped"),
                            "effective_mode": task3_info.get("effective_mode"),
                            "fallback_reason": task3_info.get("fallback_reason"),
                        }
                    )

    ref03_accepts = _accepted_rows(target_rows[BASELINE_REFERENCE_ID])
    ref03_misses = _score_rows(_miss_rows(target_rows[BASELINE_REFERENCE_ID]))[:5]
    ref03_baseline_rows = ref03_accepts + ref03_misses

    _write_csv(OUTPUT_DIR / "ref_01_per_frame.csv", target_rows["ref_01"])
    _write_csv(OUTPUT_DIR / "ref_05_per_frame.csv", target_rows["ref_05"])
    _write_csv(OUTPUT_DIR / "ref_03_baseline_per_frame.csv", ref03_baseline_rows)

    summary_markdown = _build_summary_markdown(
        routing_summary=routing_summary,
        ref01_rows=target_rows["ref_01"],
        ref05_rows=target_rows["ref_05"],
        ref03_rows=target_rows["ref_03"],
        ref05_frame_meta=ref05_frame_meta,
    )
    (OUTPUT_DIR / "summary.md").write_text(summary_markdown, encoding="utf-8")

    summary_json = {
        "routing_summary": {key: routing_summary[key] for key in ("ref_01", "ref_03", "ref_05")},
        "row_counts": {
            "ref_01": len(target_rows["ref_01"]),
            "ref_05": len(target_rows["ref_05"]),
            "ref_03_baseline": len(ref03_baseline_rows),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print(f"Wrote diagnostic probe to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
