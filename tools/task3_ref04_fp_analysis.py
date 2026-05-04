"""Root-cause analysis for ref_04 thermal absent false positives.

Measurement only. Reads archived data from the modality-limited routing v2 run,
extracts TP/FP populations for ref_04, generates composites, and writes a small
report bundle under _logs/ref04_fp_analysis/.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np

from src.evaluation.task3_manifest_eval import load_task3_manifest


ARCHIVE_DIR = Path("_logs/reports_generated/task3_manifest/2026-04-20_per_reference_routing_modality_v2")
DEBUG_DIR = Path("_logs/debug/task3_per_reference_routing_modality_v2")
OUTPUT_DIR = Path("_logs/ref04_fp_analysis")
REFERENCE_PATH = Path("data/references/2026_baseline/ref_04.jpg")
MANIFEST_PATH = Path("data/task3_eval_manifest.json")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scenario_map() -> dict[str, dict[str, Any]]:
    manifest = load_task3_manifest(MANIFEST_PATH)
    return {str(item["id"]): item for item in manifest["scenarios"]}


def _accepted_frames() -> tuple[set[int], set[int]]:
    summary = _load_json(ARCHIVE_DIR / "per_reference_routing_summary.json")
    tp_frames = set(int(value) for value in summary["thermal_cross_sensor_proxy"]["accepted_frames_by_ref"]["ref_04"])
    fp_frames = set(int(value) for value in summary["thermal_absent_target_proxy_2025"]["accepted_frames_by_ref"]["ref_04"])
    return tp_frames, fp_frames


def _pick_population(rows: list[dict[str, Any]], accepted_frames: set[int]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("object_id") != "ref_04":
            continue
        frame_idx = int(row["frame_idx"])
        if frame_idx not in accepted_frames:
            continue
        grouped.setdefault(frame_idx, []).append(row)

    selected: list[dict[str, Any]] = []
    for frame_idx in sorted(grouped):
        best = max(grouped[frame_idx], key=lambda item: float(item.get("score", 0.0)))
        selected.append(best)
    selected.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return selected


def _range(values: list[float]) -> tuple[float, float]:
    return (min(values), max(values)) if values else (0.0, 0.0)


def _format_range(values: list[float]) -> str:
    if not values:
        return "-"
    low, high = _range(values)
    return f"{low:.4f}-{high:.4f}"


def _overlap(tp_values: list[float], fp_values: list[float]) -> str:
    if not tp_values or not fp_values:
        return "-"
    tp_low, tp_high = _range(tp_values)
    fp_low, fp_high = _range(fp_values)
    low = max(tp_low, fp_low)
    high = min(tp_high, fp_high)
    if high < low:
        return "none"
    return f"{low:.4f}-{high:.4f}"


def _ascii_scatter(tp: list[dict[str, Any]], fp: list[dict[str, Any]], *, width: int = 28, height: int = 12) -> str:
    grid = [["." for _ in range(width)] for _ in range(height)]

    def _plot(points: list[dict[str, Any]], marker: str) -> None:
        for item in points:
            x = max(0.0, min(float(item["inlier_ratio"]), 1.0))
            y = max(0.0, min(float(item["score"]), 1.0))
            col = min(int(round(x * (width - 1))), width - 1)
            row = min(int(round((1.0 - y) * (height - 1))), height - 1)
            current = grid[row][col]
            if current != "." and current != marker:
                grid[row][col] = "*"
            else:
                grid[row][col] = marker

    _plot(tp, "T")
    _plot(fp, "F")
    lines = ["score^"]
    for row in grid:
        lines.append("".join(row))
    lines.append("+" + "-" * width + "> inlier_ratio")
    lines.append("Legend: T=true positive, F=false positive, *=overlap")
    return "\n".join(lines)


def _extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read frame {frame_idx} from {video_path}")
        return frame
    finally:
        cap.release()


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _resize_fit(image: np.ndarray, max_side: int = 400) -> np.ndarray:
    image = _ensure_bgr(image)
    height, width = image.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)


def _pad_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] >= target_height:
        return image
    bottom = target_height - image.shape[0]
    return cv2.copyMakeBorder(image, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = image.copy()
    y = 22
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        y += 22
    return canvas


def _write_composite(reference_bgr: np.ndarray, crop_bgr: np.ndarray, *, out_path: Path, label: str, frame_idx: int, score: float, inlier_ratio: float) -> None:
    ref_panel = _resize_fit(reference_bgr)
    crop_panel = _resize_fit(crop_bgr)
    target_height = max(ref_panel.shape[0], crop_panel.shape[0])
    ref_panel = _pad_to_height(ref_panel, target_height)
    crop_panel = _pad_to_height(crop_panel, target_height)
    ref_panel = _annotate(ref_panel, ["REFERENCE", f"{REFERENCE_PATH.name}", f"{reference_bgr.shape[1]}x{reference_bgr.shape[0]}"])
    crop_panel = _annotate(crop_panel, [label, f"frame={frame_idx}", f"score={score:.4f} inlier={inlier_ratio:.4f}"])
    composite = cv2.hconcat([ref_panel, crop_panel])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite)


def _reference_inspection(reference_gray: np.ndarray) -> dict[str, Any]:
    blurred = cv2.GaussianBlur(reference_gray, (5, 5), 0)
    _threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    components = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = components[2]
    image_area = reference_gray.shape[0] * reference_gray.shape[1]
    largest_area = 0
    largest_bbox = [0, 0, 0, 0]
    for label_idx in range(1, stats.shape[0]):
        x, y, w, h, area = [int(value) for value in stats[label_idx]]
        if area > largest_area:
            largest_area = area
            largest_bbox = [x, y, w, h]
    area_ratio = largest_area / max(image_area, 1)
    return {
        "image_width": int(reference_gray.shape[1]),
        "image_height": int(reference_gray.shape[0]),
        "largest_bright_component_bbox_xywh": largest_bbox,
        "largest_bright_component_area_ratio": round(float(area_ratio), 4),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = _scenario_map()
    tp_frames, fp_frames = _accepted_frames()
    tp_rows = _pick_population(_load_jsonl(DEBUG_DIR / "thermal_cross_sensor_proxy" / "post_gate.jsonl"), tp_frames)
    fp_rows = _pick_population(_load_jsonl(DEBUG_DIR / "thermal_absent_target_proxy_2025" / "post_gate.jsonl"), fp_frames)

    reference_bgr = cv2.imread(str(REFERENCE_PATH))
    if reference_bgr is None:
        raise RuntimeError(f"cannot load reference image: {REFERENCE_PATH}")

    tp_video = Path(scenarios["thermal_cross_sensor_proxy"]["video"])
    fp_video = Path(scenarios["thermal_absent_target_proxy_2025"]["video"])

    visual_dir = OUTPUT_DIR / "composites"
    for index, row in enumerate(tp_rows, start=1):
        frame = _extract_frame(tp_video, int(row["frame_idx"]))
        x1, y1, x2, y2 = [int(value) for value in row["bbox"]]
        crop = frame[y1:y2, x1:x2]
        _write_composite(
            reference_bgr,
            crop,
            out_path=visual_dir / f"tp_{index:02d}_frame_{int(row['frame_idx']):04d}.png",
            label="TP",
            frame_idx=int(row["frame_idx"]),
            score=float(row["score"]),
            inlier_ratio=float(row["inlier_ratio"]),
        )
    for index, row in enumerate(fp_rows, start=1):
        frame = _extract_frame(fp_video, int(row["frame_idx"]))
        x1, y1, x2, y2 = [int(value) for value in row["bbox"]]
        crop = frame[y1:y2, x1:x2]
        _write_composite(
            reference_bgr,
            crop,
            out_path=visual_dir / f"fp_{index:02d}_frame_{int(row['frame_idx']):04d}.png",
            label="FP",
            frame_idx=int(row["frame_idx"]),
            score=float(row["score"]),
            inlier_ratio=float(row["inlier_ratio"]),
        )

    # contact sheet
    ordered_paths = sorted(visual_dir.glob("*.png"))
    thumbs = []
    for path in ordered_paths:
        image = cv2.imread(str(path))
        thumbs.append(_resize_fit(image, max_side=260))
    rows: list[np.ndarray] = []
    for start in range(0, len(thumbs), 2):
        pair = thumbs[start : start + 2]
        max_h = max(item.shape[0] for item in pair)
        padded = [_pad_to_height(item, max_h) for item in pair]
        if len(padded) == 1:
            padded.append(np.zeros_like(padded[0]))
        rows.append(cv2.hconcat(padded))
    sheet = rows[0]
    for row in rows[1:]:
        max_w = max(sheet.shape[1], row.shape[1])
        if sheet.shape[1] < max_w:
            sheet = cv2.copyMakeBorder(sheet, 0, 0, 0, max_w - sheet.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if row.shape[1] < max_w:
            row = cv2.copyMakeBorder(row, 0, 0, 0, max_w - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        sheet = cv2.vconcat([sheet, row])
    cv2.imwrite(str(OUTPUT_DIR / "contact_sheet.png"), sheet)

    tp_conf = [float(item["yoloe_confidence"]) for item in tp_rows]
    fp_conf = [float(item["yoloe_confidence"]) for item in fp_rows]
    tp_matches = [float(item["match_count"]) for item in tp_rows]
    fp_matches = [float(item["match_count"]) for item in fp_rows]
    tp_inlier = [float(item["inlier_ratio"]) for item in tp_rows]
    fp_inlier = [float(item["inlier_ratio"]) for item in fp_rows]
    tp_score = [float(item["score"]) for item in tp_rows]
    fp_score = [float(item["score"]) for item in fp_rows]

    feature_summary = {
        "yoloe_confidence": {"tp_range": _format_range(tp_conf), "fp_range": _format_range(fp_conf), "overlap": _overlap(tp_conf, fp_conf)},
        "match_count": {"tp_range": _format_range(tp_matches), "fp_range": _format_range(fp_matches), "overlap": _overlap(tp_matches, fp_matches)},
        "inlier_ratio": {"tp_range": _format_range(tp_inlier), "fp_range": _format_range(fp_inlier), "overlap": _overlap(tp_inlier, fp_inlier)},
        "score_3term": {"tp_range": _format_range(tp_score), "fp_range": _format_range(fp_score), "overlap": _overlap(tp_score, fp_score)},
    }
    reference_notes = _reference_inspection(cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY))
    scatter = _ascii_scatter(tp_rows, fp_rows)

    payload = {
        "tp_population": tp_rows,
        "fp_population": fp_rows,
        "feature_summary": feature_summary,
        "ascii_scatter": scatter,
        "reference_inspection": reference_notes,
        "composite_dir": str(visual_dir),
    }
    (OUTPUT_DIR / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# ref_04 TP vs FP Analysis",
        "",
        "## Feature Summary",
        f"- YOLOE confidence: TP {feature_summary['yoloe_confidence']['tp_range']}, FP {feature_summary['yoloe_confidence']['fp_range']}, overlap {feature_summary['yoloe_confidence']['overlap']}",
        f"- match_count: TP {feature_summary['match_count']['tp_range']}, FP {feature_summary['match_count']['fp_range']}, overlap {feature_summary['match_count']['overlap']}",
        f"- inlier_ratio: TP {feature_summary['inlier_ratio']['tp_range']}, FP {feature_summary['inlier_ratio']['fp_range']}, overlap {feature_summary['inlier_ratio']['overlap']}",
        f"- score_3term: TP {feature_summary['score_3term']['tp_range']}, FP {feature_summary['score_3term']['fp_range']}, overlap {feature_summary['score_3term']['overlap']}",
        "",
        "## ASCII Scatter",
        "```text",
        scatter,
        "```",
        "",
        "## Reference Inspection",
        f"- Largest bright component bbox xywh: {reference_notes['largest_bright_component_bbox_xywh']}",
        f"- Largest bright component area ratio: {reference_notes['largest_bright_component_area_ratio']}",
        "",
        "## TP Population",
        "| frame_idx | conf | match_count | inlier_count | inlier_ratio | score | bbox |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in tp_rows:
        lines.append(
            f"| {row['frame_idx']} | {float(row['yoloe_confidence']):.6f} | {int(row['match_count'])} | {int(row['inlier_count'])} | {float(row['inlier_ratio']):.6f} | {float(row['score']):.6f} | {row['bbox']} |"
        )
    lines.extend(
        [
            "",
            "## FP Population",
            "| frame_idx | conf | match_count | inlier_count | inlier_ratio | score | bbox |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in fp_rows:
        lines.append(
            f"| {row['frame_idx']} | {float(row['yoloe_confidence']):.6f} | {int(row['match_count'])} | {int(row['inlier_count'])} | {float(row['inlier_ratio']):.6f} | {float(row['score']):.6f} | {row['bbox']} |"
        )
    (OUTPUT_DIR / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
