from __future__ import annotations

"""Analysis of sahi_per_frame_trace.csv for the D3 SAHI feasibility probe.

Reads the per-frame trace and emits machine-readable aggregates that the
human-written `recall_sinyal.md`, `false_match_riski.md`, `latency.md`, and
`summary.md` can quote verbatim.
"""

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARCHIVE_DIR = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-04-27_d3_sahi_feasibility"
)
CSV_PATH = ARCHIVE_DIR / "sahi_per_frame_trace.csv"


def _to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def main() -> int:
    rows = list(csv.DictReader(CSV_PATH.open("r", encoding="utf-8")))

    aggregates: dict[str, object] = {
        "row_count": len(rows),
        "by_scenario_ref_config": {},
    }

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["scenario"], row["ref_id"], row["config"])
        grouped.setdefault(key, []).append(row)

    for (scenario, ref_id, config), bucket in sorted(grouped.items()):
        gate_pass_count = sum(1 for row in bucket if _to_bool(row["yoloe_top1_passes_gate"]))
        score_pass_count = sum(1 for row in bucket if _to_bool(row["final_score_passes_threshold"]))
        candidate_pass_frame_count = sum(
            1 for row in bucket if (_to_int(row["candidate_count_post_gate"]) or 0) > 0
        )
        confidences = [_to_float(row["yoloe_top1_conf"]) for row in bucket]
        confidences = [value for value in confidences if value is not None]
        match_counts = [_to_int(row["lightglue_max_matches"]) for row in bucket]
        match_counts = [value for value in match_counts if value is not None]
        inlier_counts = [_to_int(row["ransac_max_inlier_count"]) for row in bucket]
        inlier_counts = [value for value in inlier_counts if value is not None]
        inlier_ratios = [_to_float(row["ransac_max_inlier_ratio"]) for row in bucket]
        inlier_ratios = [value for value in inlier_ratios if value is not None]
        scores = [_to_float(row["final_score_max"]) for row in bucket]
        scores = [value for value in scores if value is not None]
        infer_ms = [_to_float(row["yoloe_inference_ms"]) or 0.0 for row in bucket]
        verify_ms = [_to_float(row["lightglue_verify_ms"]) or 0.0 for row in bucket]
        total_ms = [_to_float(row["total_per_frame_ms"]) or 0.0 for row in bucket]

        bucket_summary = {
            "frame_count": len(bucket),
            "yoloe_gate_pass_frames": gate_pass_count,
            "candidate_pass_frame_count": candidate_pass_frame_count,
            "final_score_pass_frames": score_pass_count,
            "confidence_n": len(confidences),
            "confidence_mean": round(statistics.fmean(confidences), 6) if confidences else None,
            "confidence_max": round(max(confidences), 6) if confidences else None,
            "lightglue_match_count_distribution": dict(Counter(match_counts)),
            "lightglue_match_below_15_count": sum(1 for value in match_counts if value < 15),
            "ransac_inlier_count_distribution": dict(Counter(inlier_counts)),
            "ransac_inlier_count_below_10_count": sum(1 for value in inlier_counts if value < 10),
            "ransac_inlier_ratio_one_count": sum(1 for value in inlier_ratios if value >= 0.999),
            "ransac_inlier_ratio_above_0p9_count": sum(1 for value in inlier_ratios if value >= 0.9),
            "ransac_inlier_ratio_max": round(max(inlier_ratios), 6) if inlier_ratios else None,
            "score_max": round(max(scores), 6) if scores else None,
            "score_mean": round(statistics.fmean(scores), 6) if scores else None,
            "infer_ms_mean": round(statistics.fmean(infer_ms), 4),
            "verify_ms_mean": round(statistics.fmean(verify_ms), 4),
            "total_ms_mean": round(statistics.fmean(total_ms), 4),
            "total_ms_max": round(max(total_ms), 4),
        }
        aggregates["by_scenario_ref_config"][f"{scenario}::{ref_id}::{config}"] = bucket_summary

    sahi_low_inlier_accept = []
    sahi_inlier_one_accept = []
    sahi_absent_accepts = []
    for row in rows:
        if row["config"] != "sahi":
            continue
        if not _to_bool(row["final_score_passes_threshold"]):
            continue
        ic = _to_int(row["ransac_max_inlier_count"]) or 0
        ir = _to_float(row["ransac_max_inlier_ratio"]) or 0.0
        if ic < 10:
            sahi_low_inlier_accept.append(
                {
                    "scenario": row["scenario"],
                    "frame_idx": row["frame_idx"],
                    "ref": row["ref_id"],
                    "ic": ic,
                    "ir": ir,
                }
            )
        if ir >= 0.999:
            sahi_inlier_one_accept.append(
                {
                    "scenario": row["scenario"],
                    "frame_idx": row["frame_idx"],
                    "ref": row["ref_id"],
                    "ic": ic,
                    "ir": ir,
                }
            )
        if row["scenario"] == "rgb_absent_target_proxy_2025":
            sahi_absent_accepts.append(
                {
                    "scenario": row["scenario"],
                    "frame_idx": row["frame_idx"],
                    "ref": row["ref_id"],
                    "ic": ic,
                    "ir": ir,
                    "score": _to_float(row["final_score_max"]),
                }
            )
    aggregates["sahi_score_passes_with_low_inliers"] = sahi_low_inlier_accept
    aggregates["sahi_score_passes_with_inlier_ratio_one"] = sahi_inlier_one_accept
    aggregates["sahi_absent_scenario_score_passes"] = sahi_absent_accepts

    out_path = ARCHIVE_DIR / "analysis.json"
    out_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
