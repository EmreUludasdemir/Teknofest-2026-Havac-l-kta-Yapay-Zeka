from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.vision import cv2, np
from src.core.video_io import iter_video_frames
from tools.diagnostic.d6_probe_common import REFERENCE_DIR, evaluate_frame, reference_aspect_ratio

OUTPUT_DIR = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-05-02_d8c_ref04_isolation"
)
INPUT_RESULTS = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-05-02_d8b_ref07_routing_override"
    / "pass_a_gpu_post_ref07_fix.json"
)
SCENARIO_ID = "thermal_cross_sensor_proxy"
TARGET_REF = "ref_12"
REFERENCE_IMAGE_PATH = REFERENCE_DIR / "Referans_Nesne_12.png"


def _load_target_frames() -> list[int]:
    payload = json.loads(INPUT_RESULTS.read_text(encoding="utf-8"))
    frame_records = payload[SCENARIO_ID]["frame_records"]
    frames = [
        int(record["frame_index"])
        for record in frame_records
        if TARGET_REF in list(record.get("accepted_ids", []))
    ]
    return sorted(frames)


def _sample_frame(video_path: str, *, frame_index: int) -> Any:
    frames = iter_video_frames(video_path, frame_stride=1, limit=None, video_name=SCENARIO_ID)
    for decoded in frames:
        if int(decoded.frame_index) == int(frame_index):
            return decoded
    raise RuntimeError(f"Frame {frame_index} not found in {video_path}")


def _scenario_video_path() -> str:
    manifest = json.loads((PROJECT_ROOT / "data" / "task3_eval_manifest.json").read_text(encoding="utf-8"))
    for item in manifest.get("scenarios", []):
        scenario_id = str(item.get("id") or item.get("name") or item.get("scenario_id") or "")
        if scenario_id == SCENARIO_ID:
            return str(item["video"])
    raise KeyError(f"Scenario not found: {SCENARIO_ID}")


def _draw_bbox(frame_bgr: Any, bbox: list[float], *, label: str) -> Any:
    canvas = frame_bgr.copy()
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(canvas, label, (x1 + 4, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


def _fit_height(image_bgr: Any, target_height: int) -> Any:
    if image_bgr.shape[0] == target_height:
        return image_bgr
    scale = target_height / float(max(image_bgr.shape[0], 1))
    target_width = max(1, int(round(image_bgr.shape[1] * scale)))
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _write_contact_sheet(reference_bgr: Any, crops: list[dict[str, Any]]) -> Path:
    target_height = 220
    panels = []
    ref_panel = _fit_height(reference_bgr, target_height)
    cv2.putText(ref_panel, "ref_12", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    panels.append(ref_panel)
    for item in crops:
        crop = _fit_height(item["crop_bgr"], target_height)
        label = f"f{item['frame_index']} s={item['score']:.3f}"
        cv2.putText(crop, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        panels.append(crop)
    sheet = cv2.hconcat(panels)
    output_path = OUTPUT_DIR / "ref12_contact_sheet.png"
    cv2.imwrite(str(output_path), sheet)
    return output_path


def _classify_evidence(detail: dict[str, Any]) -> str:
    confidence = float(detail["accepted_candidate"]["task3_yoloe"]["confidence"])
    match_count = int(detail["accepted_candidate"]["task3_yoloe"]["match_count"])
    inlier_ratio = float(detail["accepted_candidate"]["task3_yoloe"]["inlier_ratio"])
    aspect_scale = float(detail["geometry"]["aspect_scale"])
    if confidence >= 0.30 and match_count >= 20 and inlier_ratio >= 0.40 and aspect_scale >= 0.40:
        return "legitimate_match"
    if match_count < 10 or inlier_ratio < 0.20 or aspect_scale < 0.25:
        return "false_match"
    return "ambiguous"


def _classify_final(detail: dict[str, Any]) -> str:
    evidence = str(detail["evidence_classification"])
    area_ratio = float(detail["geometry"]["area_ratio_percent"])
    aspect_scale = float(detail["geometry"]["aspect_scale"])
    if evidence == "legitimate_match" and area_ratio >= 35.0 and aspect_scale < 0.60:
        return "ambiguous"
    return evidence


def _classify_overall(details: list[dict[str, Any]]) -> str:
    labels = [detail["classification"] for detail in details]
    if labels and all(label == "legitimate_match" for label in labels):
        return "legitimate_match"
    if labels and all(label == "false_match" for label in labels):
        return "false_match"
    return "ambiguous"


def _frame_detail(frame_index: int, *, video_path: str, reference_bgr: Any) -> dict[str, Any]:
    probe = evaluate_frame(SCENARIO_ID, frame_index, reference_dir=REFERENCE_DIR)
    decoded = _sample_frame(video_path, frame_index=frame_index)
    accepted = next(item for item in probe["verified_matches"] if item["object_id"] == TARGET_REF)
    accepted_record = None
    for record in probe["yoloe_records"]:
        if record["object_id"] != TARGET_REF:
            continue
        if abs(float(record["final_score"]) - float(accepted["match_score"])) < 1e-4:
            accepted_record = record
            break
    if accepted_record is None:
        raise RuntimeError(f"Accepted YOLOE record not found for frame {frame_index}")

    bbox = list(accepted["bbox"])
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    crop = decoded.bgr[y1:y2, x1:x2].copy()
    bbox_width = max(x2 - x1, 1)
    bbox_height = max(y2 - y1, 1)
    frame_area = float(max(decoded.width * decoded.height, 1))
    area_ratio = (float(bbox_width * bbox_height) / frame_area) * 100.0
    bbox_aspect = float(bbox_width) / float(max(bbox_height, 1))
    ref_aspect = reference_aspect_ratio(TARGET_REF)
    aspect_scale = min(bbox_aspect, ref_aspect) / max(bbox_aspect, ref_aspect)
    reciprocal_aspect_scale = min(bbox_aspect, 1.0 / ref_aspect) / max(bbox_aspect, 1.0 / ref_aspect)
    path = "yoloe" if "yoloe" in str(accepted["matcher_source"]) else "orb"

    overlay = _draw_bbox(decoded.bgr, bbox, label=f"{TARGET_REF} {accepted['match_score']:.3f}")
    frame_path = OUTPUT_DIR / f"frame_{frame_index}_overlay.png"
    crop_path = OUTPUT_DIR / f"frame_{frame_index}_crop.png"
    cv2.imwrite(str(frame_path), overlay)
    cv2.imwrite(str(crop_path), crop)

    detail = {
        "frame_index": frame_index,
        "path": path,
        "route_mode": probe["task3_info"]["effective_mode"],
        "accepted_candidate": accepted,
        "accepted_record": accepted_record,
        "geometry": {
            "bbox_xyxy": bbox,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "bbox_aspect": round(bbox_aspect, 6),
            "ref_aspect": round(ref_aspect, 6),
            "aspect_scale": round(aspect_scale, 6),
            "reciprocal_aspect_scale": round(reciprocal_aspect_scale, 6),
            "area_ratio_percent": round(area_ratio, 6),
        },
        "frame_path": str(frame_path),
        "crop_path": str(crop_path),
        "crop_bgr": crop,
    }
    detail["evidence_classification"] = _classify_evidence(detail)
    detail["classification"] = _classify_final(detail)
    return detail


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_bgr = cv2.imread(str(REFERENCE_IMAGE_PATH))
    if reference_bgr is None:
        raise FileNotFoundError(REFERENCE_IMAGE_PATH)
    video_path = _scenario_video_path()
    target_frames = _load_target_frames()
    details = [_frame_detail(frame_index, video_path=video_path, reference_bgr=reference_bgr) for frame_index in target_frames]
    overall = _classify_overall(details)

    contact_sheet_path = _write_contact_sheet(
        reference_bgr,
        [
            {
                "frame_index": item["frame_index"],
                "score": float(item["accepted_candidate"]["match_score"]),
                "crop_bgr": item["crop_bgr"],
            }
            for item in details
        ],
    )

    serializable = []
    for item in details:
        payload = dict(item)
        del payload["crop_bgr"]
        serializable.append(payload)
    (OUTPUT_DIR / "ref12_attribution.json").write_text(
        json.dumps(
            {
                "scenario_id": SCENARIO_ID,
                "target_ref": TARGET_REF,
                "frames": serializable,
                "overall_classification": overall,
                "contact_sheet_path": str(contact_sheet_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# D8-C ref_12 Attribution",
        "",
        f"- Scenario: `{SCENARIO_ID}`",
        f"- Target ref: `{TARGET_REF}`",
        f"- Accepted frames: `{target_frames}`",
        f"- Contact sheet: `{contact_sheet_path}`",
        "",
        "## Direct Answer",
        "",
        f"- Final overall classification after visual-review heuristic: `{overall}`.",
        "- All three accepts route through `yoloe` on the thermal-only path; ORB does not contribute.",
        "- Evidence-only scoring marks all three frames as `legitimate_match`, but final classification is downgraded when a box spans a large scene patch with a strong aspect mismatch against the reference crop.",
        "",
    ]

    for item in details:
        candidate = item["accepted_candidate"]
        yoloe = candidate["task3_yoloe"]
        geo = item["geometry"]
        lines.extend(
            [
                f"## Frame {item['frame_index']}",
                "",
                f"- Evidence classification: `{item['evidence_classification']}`",
                f"- Final classification: `{item['classification']}`",
                f"- Routing path: `{item['path']}` via `{item['route_mode']}`",
                f"- Score: `{float(candidate['match_score']):.4f}` with confidence=`{float(yoloe['confidence']):.4f}` matches=`{int(yoloe['match_count'])}` inliers=`{int(yoloe['inlier_count'])}` inlier_ratio=`{float(yoloe['inlier_ratio']):.4f}`",
                f"- Geometry: bbox_aspect=`{float(geo['bbox_aspect']):.4f}` ref_aspect=`{float(geo['ref_aspect']):.4f}` aspect_scale=`{float(geo['aspect_scale']):.4f}` reciprocal_aspect_scale=`{float(geo['reciprocal_aspect_scale']):.4f}` area_ratio=`{float(geo['area_ratio_percent']):.2f}%`",
                f"- BBox: `{geo['bbox_xyxy']}`",
                f"- Overlay: `{item['frame_path']}`",
                f"- Crop: `{item['crop_path']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Decision",
            "",
            (
                "- `ref_12` has stable thermal-only YOLOE evidence and no ORB low-match saturation, "
                "but all three boxes cover large scene regions and are materially squarer than the reference crop."
            ),
            (
                "- This keeps Step 1 in the `ambiguous` lane: do not promote `ref_12` into manifest expectations yet, "
                "and do not treat it as the same mechanism as `ref_07` because no ORB low-evidence saturation appears here."
            ),
            "",
        ]
    )
    (OUTPUT_DIR / "ref12_attribution.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote ref_12 attribution outputs to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
