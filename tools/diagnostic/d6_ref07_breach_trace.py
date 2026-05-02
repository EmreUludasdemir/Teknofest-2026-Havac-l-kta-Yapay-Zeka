from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.core.vision import cv2, is_cv2_available, np
from src.evaluation.task2_long_sequence import iter_video_frames
from src.task3.reference_cache import ReferenceCache
from tools.diagnostic.d6_probe_common import MANIFEST_PATH, REFERENCE_DIR, evaluate_frame, reference_aspect_ratio

OUTPUT_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-04-27_d5_mixed_bank_integration"
REFERENCE_ID = "ref_07"
TARGETS = [
    ("thermal_cross_sensor_proxy", 450),
    ("thermal_cross_sensor_proxy", 540),
    ("thermal_absent_target_proxy_2025", 480),
    ("thermal_absent_target_proxy_2025", 960),
    ("thermal_absent_target_proxy_2025", 1080),
]


def _load_manifest_map() -> dict[str, dict[str, Any]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in payload.get("scenarios", [])}


def _load_frame(scenario_id: str, frame_index: int) -> Any:
    manifest_map = _load_manifest_map()
    scenario = manifest_map[scenario_id]
    frames = iter_video_frames(
        scenario["video"],
        frame_stride=int(scenario["frame_stride"]),
        limit=int(scenario["frame_limit"]) if scenario.get("frame_limit") is not None else None,
        video_name=scenario_id,
    )
    for decoded in frames:
        if int(decoded.frame_index) == int(frame_index):
            return decoded
    raise RuntimeError(f"Frame {frame_index} not found in {scenario_id}")


def _pct(value: float) -> float:
    return round(value * 100.0, 2)


def _area_ratio(bbox: list[float], frame_width: int, frame_height: int) -> float:
    x1, y1, x2, y2 = bbox
    area = max((x2 - x1), 0.0) * max((y2 - y1), 0.0)
    return area / max(float(frame_width * frame_height), 1.0)


def _bbox_aspect_ratio(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    width = max(float(x2 - x1), 1e-9)
    height = max(float(y2 - y1), 1e-9)
    return width / height


def _normalized_span(points: Any, width: float, height: float) -> dict[str, float]:
    xs = points[:, 0]
    ys = points[:, 1]
    return {
        "x_min": float(xs.min() / max(width, 1.0)),
        "x_max": float(xs.max() / max(width, 1.0)),
        "y_min": float(ys.min() / max(height, 1.0)),
        "y_max": float(ys.max() / max(height, 1.0)),
        "coverage_area": float(((xs.max() - xs.min()) * (ys.max() - ys.min())) / max(width * height, 1.0)),
    }


def _analyze_orb(decoded_frame: Any, cache: ReferenceCache, settings: MvpRuntimeSettings) -> dict[str, Any]:
    if not is_cv2_available() or decoded_frame.gray is None:
        return {"available": False, "reason": "opencv_or_gray_unavailable"}

    reference = cache.get(REFERENCE_ID)
    if not reference:
        return {"available": False, "reason": "missing_reference"}
    ref_descriptors = reference.get("descriptors")
    ref_keypoints = reference.get("keypoints")
    if ref_descriptors is None or not ref_keypoints:
        return {"available": False, "reason": "missing_reference_descriptors"}

    orb = cv2.ORB_create(nfeatures=settings.task3_orb_features)
    frame_keypoints, frame_descriptors = orb.detectAndCompute(decoded_frame.gray, None)
    if frame_descriptors is None or not frame_keypoints:
        return {"available": False, "reason": "missing_frame_descriptors"}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(ref_descriptors, frame_descriptors, k=2)
    good_matches: list[Any] = []
    distances: list[float] = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < settings.task3_match_ratio_threshold * second.distance:
            good_matches.append(first)
            distances.append(float(first.distance))

    result: dict[str, Any] = {
        "available": True,
        "frame_keypoint_count": int(len(frame_keypoints)),
        "ref_keypoint_count": int(len(ref_keypoints)),
        "good_match_count": int(len(good_matches)),
        "distance_mean": float(statistics.mean(distances)) if distances else None,
        "distance_median": float(statistics.median(distances)) if distances else None,
        "distance_max": float(max(distances)) if distances else None,
    }
    if len(good_matches) < settings.task3_match_min_inliers:
        result["accepted_like_runtime"] = False
        result["reason"] = "below_min_inliers"
        return result

    ref_points = np.float32([ref_keypoints[item.queryIdx].pt for item in good_matches]).reshape(-1, 1, 2)
    frame_points = np.float32([frame_keypoints[item.trainIdx].pt for item in good_matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(ref_points, frame_points, cv2.RANSAC, 5.0)
    ref_points_2d = ref_points.reshape(-1, 2)
    frame_points_2d = frame_points.reshape(-1, 2)
    ref_width = max(int(reference.get("width", 0)), 1)
    ref_height = max(int(reference.get("height", 0)), 1)

    result["ref_match_span"] = _normalized_span(ref_points_2d, float(ref_width), float(ref_height))
    result["frame_match_span"] = _normalized_span(frame_points_2d, float(decoded_frame.width), float(decoded_frame.height))

    if homography is None or inlier_mask is None:
        bbox = [
            float(frame_points_2d[:, 0].min()),
            float(frame_points_2d[:, 1].min()),
            float(frame_points_2d[:, 0].max()),
            float(frame_points_2d[:, 1].max()),
        ]
        score = min(0.95, 0.60 + 0.05 * len(good_matches))
        result.update(
            {
                "accepted_like_runtime": True,
                "matcher_source": "task3_orb_bf_bbox",
                "inlier_count": int(len(good_matches)),
                "inlier_ratio": float(min(len(good_matches) / max(settings.task3_match_min_inliers, 1), 1.0)),
                "score": float(score),
                "bbox": bbox,
                "homography_found": False,
            }
        )
        return result

    inlier_count = int(inlier_mask.ravel().sum())
    inlier_ratio = inlier_count / max(len(good_matches), 1)
    corners = np.float32([[0, 0], [ref_width - 1, 0], [ref_width - 1, ref_height - 1], [0, ref_height - 1]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    bbox = [
        float(projected[:, 0].min()),
        float(projected[:, 1].min()),
        float(projected[:, 0].max()),
        float(projected[:, 1].max()),
    ]
    score = min(0.99, 0.55 + (0.45 * inlier_ratio))
    result.update(
        {
            "accepted_like_runtime": bool(inlier_count >= settings.task3_match_min_inliers),
            "matcher_source": "task3_orb_bf_homography",
            "inlier_count": int(inlier_count),
            "inlier_ratio": float(inlier_ratio),
            "score": float(score),
            "bbox": bbox,
            "homography_found": True,
        }
    )
    return result


def _classify_pathology(verified_row: dict[str, Any] | None, yoloe_record: dict[str, Any] | None) -> str:
    if verified_row is None:
        return "no_accept"
    source = str(verified_row["matcher_source"])
    if source.startswith("task3_orb"):
        if int(verified_row["match_count"]) <= 6 and float(verified_row["inlier_ratio"]) >= 0.8:
            return "orb_low_match_high_inlier_ratio"
        return "orb_non_pathological"
    if yoloe_record is not None:
        if int(yoloe_record["match_count"]) <= 15 and float(yoloe_record["inlier_ratio"]) >= 0.4:
            return "yoloe_low_match_high_inlier_ratio"
        return "yoloe_match_term_dominant"
    return "accepted_without_trace"


def _serialize_frame(scenario_id: str, frame_index: int, probe: dict[str, Any], orb_analysis: dict[str, Any]) -> dict[str, Any]:
    verified_row = next((row for row in probe["verified_matches"] if row["object_id"] == REFERENCE_ID), None)
    yoloe_rows = [row for row in probe["yoloe_records"] if row["object_id"] == REFERENCE_ID]
    yoloe_row = max(yoloe_rows, key=lambda item: float(item["final_score"])) if yoloe_rows else None
    route_mode = "both" if REFERENCE_ID in probe["yoloe_routed_refs"] and REFERENCE_ID in probe["orb_routed_refs"] else "single"
    geometry = None
    if verified_row is not None:
        bbox = list(verified_row["bbox"])
        bbox_aspect = _bbox_aspect_ratio(bbox)
        ref_aspect = reference_aspect_ratio(REFERENCE_ID)
        geometry = {
            "area_ratio": _area_ratio(bbox, int(probe["frame_width"]), int(probe["frame_height"])),
            "bbox_aspect_ratio": bbox_aspect,
            "reference_aspect_ratio": ref_aspect,
            "aspect_ratio_scale": bbox_aspect / max(ref_aspect, 1e-9),
            "bbox_sane": verified_row["bbox_sane"],
            "scale_ok": verified_row["scale_ok"],
        }
    return {
        "scenario_id": scenario_id,
        "frame_index": frame_index,
        "modality": probe["modality"],
        "route_mode": route_mode,
        "routed_to_yoloe": REFERENCE_ID in probe["yoloe_routed_refs"],
        "routed_to_orb": REFERENCE_ID in probe["orb_routed_refs"],
        "raw_match_count": len([row for row in probe["raw_matches"] if row["object_id"] == REFERENCE_ID]),
        "filtered_match_count": len([row for row in probe["filtered_matches"] if row["object_id"] == REFERENCE_ID]),
        "verified_match": verified_row,
        "yoloe_record": yoloe_row,
        "orb_analysis": orb_analysis,
        "geometry": geometry,
        "pathology": _classify_pathology(verified_row, yoloe_row),
    }


def _render_frame_block(lines: list[str], payload: dict[str, Any]) -> None:
    verified = payload["verified_match"]
    yoloe = payload["yoloe_record"]
    geometry = payload["geometry"]
    orb = payload["orb_analysis"]
    lines.extend(
        [
            f"## {payload['scenario_id']} frame {payload['frame_index']}",
            "",
            f"- Route mode: `{payload['route_mode']}` (yoloe={payload['routed_to_yoloe']}, orb={payload['routed_to_orb']})",
            f"- Runtime pathology tag: `{payload['pathology']}`",
        ]
    )
    if verified is None:
        lines.extend(["- ref_07 verified accept: none", ""])
        return

    source = str(verified["matcher_source"])
    lines.append(f"- Accepted via: `{source}` with score `{float(verified['match_score']):.4f}`")
    lines.append(
        f"- Runtime bbox sanity: bbox_sane=`{verified['bbox_sane']}` scale_ok=`{verified['scale_ok']}` match_count=`{int(verified['match_count'])}` inlier_count=`{int(verified['inlier_count'])}` inlier_ratio=`{float(verified['inlier_ratio']):.4f}`"
    )
    if geometry is not None:
        lines.append(
            f"- Geometry: area_ratio=`{_pct(float(geometry['area_ratio']))}%` bbox_aspect=`{float(geometry['bbox_aspect_ratio']):.4f}` ref_aspect=`{float(geometry['reference_aspect_ratio']):.4f}` aspect_scale=`{float(geometry['aspect_ratio_scale']):.4f}`"
        )
    if source.startswith("task3_yoloe") and yoloe is not None:
        lines.append(
            f"- YOLOE score terms: conf_term=`{float(yoloe['score_det_term']):.4f}` match_term=`{float(yoloe['score_match_term']):.4f}` inlier_term=`{float(yoloe['score_inlier_term']):.4f}`"
        )
        lines.append(
            f"- YOLOE internals: confidence=`{float(yoloe['yoloe_confidence']):.4f}` matches=`{int(yoloe['match_count'])}` inliers=`{int(yoloe['inlier_count'])}` inlier_ratio=`{float(yoloe['inlier_ratio']):.4f}`"
        )
    if source == "task3_orb_bf_bbox":
        lines.append(f"- ORB score model: `0.60 + 0.05 * good_matches` -> `{float(verified['match_score']):.4f}`.")
    if source == "task3_orb_bf_homography":
        lines.append(f"- ORB score model: `0.55 + 0.45 * inlier_ratio` -> `{float(verified['match_score']):.4f}`.")
    if orb.get("available"):
        lines.append(
            f"- ORB forensic: good_matches=`{int(orb['good_match_count'])}` distance_mean=`{float(orb['distance_mean'] or 0.0):.2f}` distance_median=`{float(orb['distance_median'] or 0.0):.2f}` homography_found=`{orb.get('homography_found')}`"
        )
        ref_span = orb.get("ref_match_span")
        frame_span = orb.get("frame_match_span")
        if ref_span is not None and frame_span is not None:
            lines.append(
                f"- ORB feature span on ref_07: x=`{_pct(float(ref_span['x_min']))}-{_pct(float(ref_span['x_max']))}%` y=`{_pct(float(ref_span['y_min']))}-{_pct(float(ref_span['y_max']))}%` coverage=`{_pct(float(ref_span['coverage_area']))}%`"
            )
            lines.append(
                f"- ORB feature span on thermal frame: x=`{_pct(float(frame_span['x_min']))}-{_pct(float(frame_span['x_max']))}%` y=`{_pct(float(frame_span['y_min']))}-{_pct(float(frame_span['y_max']))}%` coverage=`{_pct(float(frame_span['coverage_area']))}%`"
            )
    lines.append("")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = MvpRuntimeSettings(
        task3_mode="yoloe_vp_lightglue",
        task3_reference_dir=REFERENCE_DIR,
        task3_eval_reference_dir=REFERENCE_DIR,
        task3_yoloe_allow_cpu=True,
    )
    cache = ReferenceCache()
    cache.preload_from_directory(REFERENCE_DIR, orb_features=settings.task3_orb_features)

    payloads: list[dict[str, Any]] = []
    for scenario_id, frame_index in TARGETS:
        probe = evaluate_frame(scenario_id, frame_index, reference_dir=REFERENCE_DIR)
        decoded = _load_frame(scenario_id, frame_index)
        orb_analysis = _analyze_orb(decoded, cache, settings)
        payloads.append(_serialize_frame(scenario_id, frame_index, probe, orb_analysis))

    summary_counts = {
        "accepted_via_orb": sum(1 for item in payloads if (item["verified_match"] or {}).get("matcher_source", "").startswith("task3_orb")),
        "accepted_via_yoloe": sum(1 for item in payloads if (item["verified_match"] or {}).get("matcher_source", "").startswith("task3_yoloe")),
        "orb_pathology_count": sum(1 for item in payloads if item["pathology"] == "orb_low_match_high_inlier_ratio"),
        "yoloe_match_term_dominant_count": sum(1 for item in payloads if item["pathology"] == "yoloe_match_term_dominant"),
    }
    (OUTPUT_DIR / "ref07_breach_trace.json").write_text(json.dumps({"frames": payloads, "summary": summary_counts}, indent=2), encoding="utf-8")

    lines = [
        "# D6 ref_07 Breach Trace",
        "",
        "- Scope: thermal video frames where D5 Pass A accepted `ref_07`.",
        f"- Target frames: `{TARGETS}`",
        "",
        "## Direct Answer",
        "",
        f"- `ref_07` is routed through `both`, but the five thermal accepts are mostly ORB-driven: ORB produces `{summary_counts['accepted_via_orb']}/5`, YOLOE produces `{summary_counts['accepted_via_yoloe']}/5`.",
        f"- ORB shows the D2'-A-style low-evidence pattern on `{summary_counts['orb_pathology_count']}/5` frames: only 5-6 good matches, yet inlier_ratio reaches 0.8-1.0 so the score jumps to 0.85-0.90 and clears the verifier.",
        "- YOLOE contributes only one accept. That pass is match-term dominant, not inlier-ratio saturation: LightGlue matches=`39`, inliers=`6`, inlier_ratio=`0.1538`, but the 0.61 match-count weight still carries the score above the thermal 0.50 gate.",
        "- Geometric validity is weak overall. The verifier checks bbox sanity and coarse scale, but aspect-ratio consistency with ref_07 is often poor, which points to accidental cross-modality structure instead of a stable object-level match.",
        "",
    ]
    for payload in payloads:
        _render_frame_block(lines, payload)

    (OUTPUT_DIR / "ref07_breach_trace.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote ref07 breach trace to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
