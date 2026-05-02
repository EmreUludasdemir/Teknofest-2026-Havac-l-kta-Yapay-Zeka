from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.diagnostic.d6_probe_common import (
    REFERENCE_DIR,
    build_temp_reference_bank,
    cleanup_temp_dir,
    evaluate_frame,
)

OUTPUT_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-04-27_d5_mixed_bank_integration"
TARGET_SCENARIO = "thermal_cross_sensor_proxy"
TARGET_FRAME_INDEX = 270
SIX_REF_IDS = [f"ref_{index:02d}" for index in range(1, 7)]
TWELVE_REF_IDS = [f"ref_{index:02d}" for index in range(1, 13)]
AMBIGUITY_MARGIN = 0.05


def _best_score(candidates: list[dict[str, Any]]) -> float | None:
    if not candidates:
        return None
    return max(float(item["match_score"]) for item in candidates)


def _match_rows(probe: dict[str, Any], reference_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = [item for item in probe["raw_matches"] if item["object_id"] == reference_id]
    filtered = [item for item in probe["filtered_matches"] if item["object_id"] == reference_id]
    verified = [item for item in probe["verified_matches"] if item["object_id"] == reference_id]
    yoloe = [item for item in probe["yoloe_records"] if item["object_id"] == reference_id]
    return raw, filtered, verified, yoloe


def _determine_ref04_status(probe: dict[str, Any]) -> tuple[str, str]:
    raw_ref04, filtered_ref04, verified_ref04, yoloe_ref04 = _match_rows(probe, "ref_04")
    if verified_ref04:
        return "verified", "ref_04 survives raw -> filtered -> verified."
    if filtered_ref04 and not verified_ref04:
        reason = filtered_ref04[0].get("verification_status") or "verifier_rejected"
        return "verifier_rejected", f"ref_04 survives suppression but fails verifier ({reason})."
    if raw_ref04 and not filtered_ref04:
        sorted_raw = sorted(raw_ref04, key=lambda item: float(item["match_score"]), reverse=True)
        best = float(sorted_raw[0]["match_score"])
        threshold = float(probe["threshold"])
        if best < threshold:
            return "below_threshold", f"ref_04 raw score {best:.4f} stays below threshold {threshold:.4f}."
        if probe["suppression_mode"] == "per_reference_top_1" and len(sorted_raw) > 1:
            second = float(sorted_raw[1]["match_score"])
            if (best - second) < AMBIGUITY_MARGIN:
                return "same_ref_ambiguity", f"ref_04 raw candidates are ambiguous within the same reference (gap {best - second:.4f})."
        if probe["suppression_mode"] == "global_top_1" and probe["filtered_matches"]:
            suppressor = probe["filtered_matches"][0]["object_id"]
            return "global_top1_suppressed", f"ref_04 clears threshold but loses to global top-1 candidate {suppressor}."
        return "suppressed_pre_verify", "ref_04 is present in raw matches but removed by suppression before verifier."
    if yoloe_ref04:
        gate_pass = any(bool(item["gate_pass"]) for item in yoloe_ref04)
        if not gate_pass:
            best = max(yoloe_ref04, key=lambda item: float(item["final_score"]))
            return (
                "lightglue_gate_reject",
                f"YOLOE sees ref_04 candidates, but all fail the LightGlue gate; best post-score is {float(best['final_score']):.4f}.",
            )
    competing_ids = [item["object_id"] for item in probe["raw_matches"]]
    if competing_ids:
        return "no_ref04_candidate", f"frame has raw candidates {competing_ids}, but none for ref_04."
    return "no_candidates", "no raw candidates survive matcher.match on this frame."


def _mechanism_summary(probe1: dict[str, Any], probe2: dict[str, Any], probe3: dict[str, Any]) -> list[str]:
    probe1_state, _ = _determine_ref04_status(probe1)
    probe2_state, _ = _determine_ref04_status(probe2)
    probe3_state, _ = _determine_ref04_status(probe3)
    lines: list[str] = []
    if probe1_state == "verified" and probe2_state in {"no_ref04_candidate", "lightglue_gate_reject", "no_candidates"} and probe3_state == probe2_state:
        lines.append("- M2 confirmed: 12-ref bank changes pre-suppression candidate generation for frame 270. ref_04 disappears before suppression, so D2 per-reference mode is not the root cause.")
    elif probe1_state == "verified" and probe2_state != "verified" and probe3_state == "verified":
        lines.append("- M3 confirmed: 12-ref frame 270 still generates ref_04, but the current suppression regime removes it. Switching back to global_top_1 restores the accept.")
    elif not math.isclose(float(probe1["threshold"]), float(probe2["threshold"]), rel_tol=0.0, abs_tol=1e-9):
        lines.append("- M4 confirmed: threshold dispatch differs between 6-ref and 12-ref probes.")
    else:
        lines.append("- No single mechanism is fully confirmed; frame 270 needs follow-up on candidate generation and verifier behavior.")

    if probe2["suppression_mode"] != "per_reference_top_1":
        lines.append("- Unexpected plumbing: Probe 2 did not load per_reference_top_1.")
    if probe3["suppression_mode"] != "global_top_1":
        lines.append("- Unexpected plumbing: Probe 3 did not flip to global_top_1 when per_reference_suppression=false.")
    else:
        lines.append("- M3 plumbing check: Probe 3 correctly maps per_reference_suppression=false to global_top_1.")
    return lines


def _render_candidate_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["| - | - | - | - | - | - |", "| - | - | - | - | - | - |"]
    lines = [
        "| Ref | Source | Score | Match Count | Inlier Count | Inlier Ratio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['object_id']} | {row['matcher_source']} | {float(row['match_score']):.4f} | "
            f"{int(row['match_count'])} | {int(row['inlier_count'])} | {float(row['inlier_ratio']):.4f} |"
        )
    return lines


def _append_probe_section(lines: list[str], title: str, probe: dict[str, Any]) -> None:
    ref04_state, ref04_reason = _determine_ref04_status(probe)
    lines.extend(
        [
            f"## {title}",
            "",
            f"- Suppression mode: `{probe['suppression_mode']}`",
            f"- Frame modality: `{probe['modality']}`",
            f"- Raw candidate count: `{len(probe['raw_matches'])}`",
            f"- Filtered candidate count: `{len(probe['filtered_matches'])}`",
            f"- Verified candidate count: `{len(probe['verified_matches'])}`",
            f"- ref_04 state: `{ref04_state}`",
            f"- ref_04 explanation: {ref04_reason}",
            f"- Active refs: `{probe['active_ids']}`",
            "",
            "### Raw Candidates",
        ]
    )
    lines.extend(_render_candidate_table(probe["raw_matches"]))
    lines.extend(["", "### Filtered Candidates"])
    lines.extend(_render_candidate_table(probe["filtered_matches"]))
    lines.extend(["", "### Verified Candidates"])
    lines.extend(_render_candidate_table(probe["verified_matches"]))

    raw_ref04, filtered_ref04, verified_ref04, yoloe_ref04 = _match_rows(probe, "ref_04")
    lines.extend(["", "### ref_04 Path"])
    lines.append(f"- raw_ref04_count: `{len(raw_ref04)}`")
    lines.append(f"- filtered_ref04_count: `{len(filtered_ref04)}`")
    lines.append(f"- verified_ref04_count: `{len(verified_ref04)}`")
    if yoloe_ref04:
        best_yoloe = max(yoloe_ref04, key=lambda item: float(item["final_score"]))
        lines.append(
            f"- best_yoloe_record: gate_pass=`{best_yoloe['gate_pass']}` conf=`{float(best_yoloe['yoloe_confidence']):.4f}` "
            f"matches=`{int(best_yoloe['match_count'])}` inlier_ratio=`{float(best_yoloe['inlier_ratio']):.4f}` score=`{float(best_yoloe['final_score']):.4f}`"
        )
    else:
        lines.append("- best_yoloe_record: none")
    lines.append("")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    probe1_dir = build_temp_reference_bank(
        selected_ref_ids=SIX_REF_IDS,
        per_reference_suppression=True,
        suffix="d6_ref04_probe1_six_ref",
    )
    probe3_dir = build_temp_reference_bank(
        selected_ref_ids=TWELVE_REF_IDS,
        per_reference_suppression=False,
        suffix="d6_ref04_probe3_global_top1",
    )

    try:
        probe1 = evaluate_frame(TARGET_SCENARIO, TARGET_FRAME_INDEX, reference_dir=probe1_dir)
        probe2 = evaluate_frame(TARGET_SCENARIO, TARGET_FRAME_INDEX, reference_dir=REFERENCE_DIR)
        probe3 = evaluate_frame(TARGET_SCENARIO, TARGET_FRAME_INDEX, reference_dir=probe3_dir)
    finally:
        cleanup_temp_dir(probe1_dir)
        cleanup_temp_dir(probe3_dir)

    payload = {"probe_1": probe1, "probe_2": probe2, "probe_3": probe3}
    (OUTPUT_DIR / "ref04_disappearance_trace.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# D6 ref_04 Disappearance Trace",
        "",
        f"- Target scenario: `{TARGET_SCENARIO}`",
        f"- Target frame: `{TARGET_FRAME_INDEX}`",
        "- Probe 1 = temporary 6-ref rollback bank with `per_reference_suppression=true`.",
        "- Probe 2 = current 12-ref D5 bank.",
        "- Probe 3 = temporary 12-ref bank with `per_reference_suppression=false` which maps to `global_top_1` in current plumbing.",
        "",
        "## Direct Answer",
        "",
    ]
    lines.extend(_mechanism_summary(probe1, probe2, probe3))
    lines.append("")
    _append_probe_section(lines, "Probe 1 — 6-ref rollback", probe1)
    _append_probe_section(lines, "Probe 2 — 12-ref current", probe2)
    _append_probe_section(lines, "Probe 3 — 12-ref global_top_1", probe3)

    lines.extend(
        [
            "## Comparison",
            "",
            "| Probe | ref_04 raw best | ref_04 filtered best | ref_04 verified best | Other raw refs |",
            "| --- | --- | --- | --- | --- |",
            f"| Probe 1 | {(_best_score(_match_rows(probe1, 'ref_04')[0]) or 0.0):.4f} | {(_best_score(_match_rows(probe1, 'ref_04')[1]) or 0.0):.4f} | {(_best_score(_match_rows(probe1, 'ref_04')[2]) or 0.0):.4f} | {sorted({item['object_id'] for item in probe1['raw_matches'] if item['object_id'] != 'ref_04'})} |",
            f"| Probe 2 | {(_best_score(_match_rows(probe2, 'ref_04')[0]) or 0.0):.4f} | {(_best_score(_match_rows(probe2, 'ref_04')[1]) or 0.0):.4f} | {(_best_score(_match_rows(probe2, 'ref_04')[2]) or 0.0):.4f} | {sorted({item['object_id'] for item in probe2['raw_matches'] if item['object_id'] != 'ref_04'})} |",
            f"| Probe 3 | {(_best_score(_match_rows(probe3, 'ref_04')[0]) or 0.0):.4f} | {(_best_score(_match_rows(probe3, 'ref_04')[1]) or 0.0):.4f} | {(_best_score(_match_rows(probe3, 'ref_04')[2]) or 0.0):.4f} | {sorted({item['object_id'] for item in probe3['raw_matches'] if item['object_id'] != 'ref_04'})} |",
            "",
        ]
    )

    (OUTPUT_DIR / "ref04_disappearance_trace.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote ref04 disappearance trace to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
