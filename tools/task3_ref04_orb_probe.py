"""Probe-only runner for routing ref_04 through ORB on thermal scenarios.

Creates a temporary reference bank manifest that points to the probe spec,
runs the 4-scenario manifest evaluator, and writes thermal-focused summaries
plus a few visual composites under the archive directory.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope
from src.core.video_io import iter_video_frames
from src.evaluation.task3_manifest_eval import evaluate_task3_manifest, load_task3_manifest
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches


ARCHIVE_DIR = Path("_logs/reports_generated/task3_manifest/2026-04-21_ref04_orb_probe")
REFERENCE_BANK_DIR = ARCHIVE_DIR / "reference_bank"
VISUAL_DIR = ARCHIVE_DIR / "visuals"
PROBE_SPEC = Path("data/references/specs/2026_baseline_v2_ref04_orb.json")
BASELINE_REFERENCE_DIR = Path("data/references/2026_baseline")
BASELINE_V2_ARCHIVE = Path("_logs/reports_generated/task3_manifest/2026-04-20_per_reference_routing_modality_v2")
MANIFEST_PATH = Path("data/task3_eval_manifest.json")


def _prepare_probe_reference_dir() -> Path:
    if REFERENCE_BANK_DIR.exists():
        shutil.rmtree(REFERENCE_BANK_DIR)
    REFERENCE_BANK_DIR.mkdir(parents=True, exist_ok=True)
    for image in sorted(BASELINE_REFERENCE_DIR.glob("ref_*.*")):
        shutil.copy2(image, REFERENCE_BANK_DIR / image.name)

    manifest_payload = json.loads((BASELINE_REFERENCE_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload["spec_path"] = str(PROBE_SPEC.resolve())
    (REFERENCE_BANK_DIR / "manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return REFERENCE_BANK_DIR


def _detector_family(candidate: Any) -> str:
    source = str(candidate.metadata.get("matcher_source", ""))
    if "orb" in source:
        return "orb"
    if "yoloe" in source:
        return "yoloe"
    if "placeholder" in source:
        return "placeholder"
    return "other"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_bgr(image):
    if image is None:
        raise RuntimeError("missing image")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _resize_fit(image, max_side: int = 400):
    image = _ensure_bgr(image)
    h, w = image.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _pad_to_height(image, target_height: int):
    if image.shape[0] >= target_height:
        return image
    return cv2.copyMakeBorder(image, 0, target_height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _annotate(image, lines: list[str]):
    canvas = image.copy()
    y = 22
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        y += 22
    return canvas


def _write_composite(reference_bgr, crop_bgr, *, out_path: Path, frame_idx: int, score: float, source: str):
    left = _resize_fit(reference_bgr)
    right = _resize_fit(crop_bgr)
    target_h = max(left.shape[0], right.shape[0])
    left = _pad_to_height(left, target_h)
    right = _pad_to_height(right, target_h)
    left = _annotate(left, ["REFERENCE", "ref_04.jpg"])
    right = _annotate(right, [f"frame={frame_idx}", f"score={score:.4f}", source])
    composite = cv2.hconcat([left, right])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite)


def _extract_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read frame {frame_idx} from {video_path}")
        return frame
    finally:
        cap.release()


def _run_detailed_probe(reference_dir: Path) -> dict[str, Any]:
    settings = replace(
        MvpRuntimeSettings(),
        task3_yoloe_allow_cpu=True,
        task3_reference_dir=reference_dir,
        task3_eval_reference_dir=reference_dir,
        task3_debug_dump_rejects=False,
    )
    manifest = load_task3_manifest(MANIFEST_PATH)
    per_frame_dump: dict[str, list[dict[str, Any]]] = {
        "thermal_cross_sensor_proxy": [],
        "thermal_absent_target_proxy_2025": [],
    }
    accepted_visual_candidates: list[dict[str, Any]] = []

    for scenario in manifest["scenarios"]:
        cache = ReferenceCache()
        cache.preload_from_directory(reference_dir, orb_features=settings.task3_orb_features)
        matcher = Task3Matcher(reference_cache=cache, runtime_settings=replace(settings, task3_mode="yoloe_vp_lightglue"))
        frames = iter_video_frames(
            scenario["video"],
            frame_stride=int(scenario["frame_stride"]),
            limit=int(scenario["frame_limit"]) if scenario["frame_limit"] is not None else None,
            video_name=scenario["id"],
        )
        for logical_idx, decoded in enumerate(frames):
            frame = FrameEnvelope(
                frame_url=f"http://task3-eval/frames/{logical_idx + 1}/",
                image_url=f"/task3/{logical_idx + 1}.jpg",
                video_name=scenario["id"],
                translation_x=0.0,
                translation_y=0.0,
                translation_z=0.0,
                health_status="1",
                metadata={"frame_index": decoded.frame_index},
            )
            raw = matcher.match(frame, b"", cache.list_ids(), decoded_frame=decoded, mode="yoloe_vp_lightglue")
            filtered = filter_no_match_candidates(
                raw,
                min_score=settings.task3_min_score,
                mode="yoloe_vp_lightglue",
                yoloe_min_score=settings.task3_yoloe_min_score,
                modality=decoded.modality,
                yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
                ambiguity_margin=settings.task3_ambiguity_margin,
            )
            verified = verify_matches(frame, filtered, decoded_frame=decoded, min_inliers=settings.task3_match_min_inliers)
            accepted_keys = {
                (item.object_id, str(item.metadata.get("matcher_source", "")), round(float(item.metadata.get("match_score", 0.0)), 6))
                for item in verified
            }

            if scenario["id"] in per_frame_dump:
                placeholder_count = sum(1 for item in raw if _detector_family(item) == "placeholder")
                real_orb_candidates = [item for item in raw if item.object_id == "ref_04" and _detector_family(item) == "orb"]
                for item in real_orb_candidates:
                    record = {
                        "frame_idx": decoded.frame_index,
                        "object_id": item.object_id,
                        "matcher_source": item.metadata.get("matcher_source"),
                        "score": float(item.metadata.get("match_score", 0.0)),
                        "match_count": int(item.metadata.get("match_count", 0)),
                        "placeholder_count_in_frame": placeholder_count,
                        "accepted": (item.object_id, str(item.metadata.get("matcher_source", "")), round(float(item.metadata.get("match_score", 0.0)), 6)) in accepted_keys,
                        "bbox": [
                            int(item.top_left_x),
                            int(item.top_left_y),
                            int(item.bottom_right_x),
                            int(item.bottom_right_y),
                        ],
                    }
                    per_frame_dump[scenario["id"]].append(record)
                    if record["accepted"] and len(accepted_visual_candidates) < 3:
                        accepted_visual_candidates.append(
                            {
                                "scenario_id": scenario["id"],
                                "video": scenario["video"],
                                "frame_idx": decoded.frame_index,
                                "score": record["score"],
                                "source": record["matcher_source"],
                                "bbox": record["bbox"],
                            }
                        )

    reference_bgr = cv2.imread(str(reference_dir / "ref_04.jpg"))
    for index, item in enumerate(accepted_visual_candidates, start=1):
        frame = _extract_frame(Path(item["video"]), int(item["frame_idx"]))
        x1, y1, x2, y2 = item["bbox"]
        crop = frame[y1:y2, x1:x2]
        _write_composite(
            reference_bgr,
            crop,
            out_path=VISUAL_DIR / f"accept_{index:02d}_{item['scenario_id']}_frame_{int(item['frame_idx']):04d}.png",
            frame_idx=int(item["frame_idx"]),
            score=float(item["score"]),
            source=str(item["source"]),
        )

    return {
        "per_frame_dump": per_frame_dump,
        "accepted_visual_candidates": accepted_visual_candidates,
    }


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    reference_dir = _prepare_probe_reference_dir()

    settings = replace(
        MvpRuntimeSettings(),
        task3_yoloe_allow_cpu=True,
        task3_reference_dir=reference_dir,
        task3_eval_reference_dir=reference_dir,
    )
    comparison = evaluate_task3_manifest(runtime_settings=settings, output_dir=ARCHIVE_DIR)
    detailed = _run_detailed_probe(reference_dir)

    baseline_summary = _load_json(BASELINE_V2_ARCHIVE / "per_reference_routing_summary.json")

    probe_summary = {}
    manifest = load_task3_manifest(MANIFEST_PATH)
    scenario_map = {item["id"]: item for item in manifest["scenarios"]}
    for scenario_id, rows in detailed["per_frame_dump"].items():
        accepted_by_ref = {"ref_04": sum(1 for row in rows if row["accepted"])}
        probe_summary[scenario_id] = {
            "accepted_by_ref": accepted_by_ref,
            "rows": rows,
            "placeholder_candidates_total": sum(int(row["placeholder_count_in_frame"]) for row in rows),
        }

    output = {
        "comparison_rows": comparison["comparison_rows"],
        "baseline_v2": {
            "thermal_cross_sensor_proxy": baseline_summary["thermal_cross_sensor_proxy"]["accepted_by_ref"],
            "thermal_absent_target_proxy_2025": baseline_summary["thermal_absent_target_proxy_2025"]["accepted_by_ref"],
            "rgb_reference_session": baseline_summary["rgb_reference_session"]["accepted_by_ref"],
            "rgb_absent_target_proxy_2025": baseline_summary["rgb_absent_target_proxy_2025"]["accepted_by_ref"],
        },
        "probe_summary": probe_summary,
        "accepted_visual_candidates": detailed["accepted_visual_candidates"],
    }
    (ARCHIVE_DIR / "ref04_orb_probe_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# ref_04 ORB Probe Summary",
        "",
        "## Thermal Cross Per-frame",
    ]
    for row in detailed["per_frame_dump"]["thermal_cross_sensor_proxy"]:
        lines.append(
            f"- frame={row['frame_idx']} score={row['score']:.4f} matches={row['match_count']} accepted={row['accepted']} source={row['matcher_source']} placeholder_in_frame={row['placeholder_count_in_frame']}"
        )
    lines.extend(["", "## Thermal Absent Per-frame"])
    for row in detailed["per_frame_dump"]["thermal_absent_target_proxy_2025"]:
        lines.append(
            f"- frame={row['frame_idx']} score={row['score']:.4f} matches={row['match_count']} accepted={row['accepted']} source={row['matcher_source']} placeholder_in_frame={row['placeholder_count_in_frame']}"
        )
    (ARCHIVE_DIR / "ref04_orb_probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
