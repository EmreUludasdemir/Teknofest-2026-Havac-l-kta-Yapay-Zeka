"""Probe for Task 3 cross-detector merge and VPE composition sensitivity.

This is a measurement-only tool. It does not change production defaults.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import CanonicalUndefinedObject, FrameEnvelope
from src.core.video_io import iter_video_frames
from src.evaluation.task3_manifest_eval import load_task3_manifest
from src.task3.experimental.backend import YoloeVpLightGlueBackend
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches


ARCHIVE_ROOT = Path("_logs/reports_generated/task3_manifest/2026-04-21_cross_detector_diagnostics")
CURRENT_V2_ARCHIVE = Path("_logs/reports_generated/task3_manifest/2026-04-20_per_reference_routing_modality_v2")
ROUTING_V1_ARCHIVE = Path("_logs/reports_generated/task3_manifest/2026-04-20_per_reference_routing_v1")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_by_id(manifest: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    for scenario in manifest.get("scenarios", []):
        if str(scenario.get("id")) == scenario_id:
            return scenario
    raise KeyError(f"scenario not found: {scenario_id}")


def _frame_envelope(*, logical_index: int, decoded: Any, scenario_id: str) -> FrameEnvelope:
    return FrameEnvelope(
        frame_url=f"http://task3-eval/frames/{logical_index + 1}/",
        image_url=f"/task3/{logical_index + 1}.jpg",
        video_name=scenario_id,
        translation_x=0.0,
        translation_y=0.0,
        translation_z=0.0,
        health_status="1",
        metadata={"frame_index": decoded.frame_index, "image_width": decoded.width, "image_height": decoded.height},
    )


def _detector_family(candidate: CanonicalUndefinedObject) -> str:
    source = str(candidate.metadata.get("matcher_source", ""))
    if "yoloe" in source:
        return "yoloe"
    if "orb" in source:
        return "orb"
    return "other"


def _candidate_detail(candidate: CanonicalUndefinedObject) -> dict[str, Any]:
    yoloe_meta = candidate.metadata.get("task3_yoloe", {})
    return {
        "object_id": candidate.object_id,
        "source": candidate.metadata.get("matcher_source"),
        "score": float(candidate.metadata.get("match_score", 0.0)),
        "confidence": float(yoloe_meta.get("confidence", 0.0)) if isinstance(yoloe_meta, dict) else 0.0,
        "match_count": int(yoloe_meta.get("match_count", candidate.metadata.get("match_count", 0))) if isinstance(yoloe_meta, dict) else int(candidate.metadata.get("match_count", 0)),
        "inlier_ratio": float(yoloe_meta.get("inlier_ratio", candidate.metadata.get("inlier_ratio", 0.0))) if isinstance(yoloe_meta, dict) else float(candidate.metadata.get("inlier_ratio", 0.0)),
    }


def _score_summary(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(scores),
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "mean": round(mean(scores), 6),
        "median": round(median(scores), 6),
    }


def _run_detailed_v2(settings: MvpRuntimeSettings, manifest: dict[str, Any]) -> dict[str, Any]:
    suppression = {
        "frames_with_both_detectors": 0,
        "yoloe_suppressed_by_orb": 0,
        "orb_suppressed_by_yoloe": 0,
        "suppression_no_effect": 0,
        "yoloe_suppressed_cases": [],
    }
    accepted_scores = {"yoloe": [], "orb": []}
    scenario_summaries: dict[str, Any] = {}

    for scenario in manifest.get("scenarios", []):
        cache = ReferenceCache()
        cache.preload_from_directory(scenario["references_dir"], orb_features=settings.task3_orb_features)
        matcher = Task3Matcher(
            reference_cache=cache,
            runtime_settings=replace(
                settings,
                task3_mode="yoloe_vp_lightglue",
                task3_reference_dir=Path(scenario["references_dir"]),
                task3_eval_reference_dir=Path(scenario["references_dir"]),
            ),
        )

        accepted_by_ref: dict[str, int] = {}
        frames = iter_video_frames(
            scenario["video"],
            frame_stride=int(scenario["frame_stride"]),
            limit=int(scenario["frame_limit"]) if scenario["frame_limit"] is not None else None,
            video_name=scenario["id"],
        )
        for logical_index, decoded in enumerate(frames):
            frame = _frame_envelope(logical_index=logical_index, decoded=decoded, scenario_id=scenario["id"])
            raw = matcher.match(frame, b"", cache.list_ids(), decoded_frame=decoded, mode="yoloe_vp_lightglue")
            yoloe_raw = [item for item in raw if _detector_family(item) == "yoloe"]
            orb_raw = [item for item in raw if _detector_family(item) == "orb"]

            filtered_all = filter_no_match_candidates(
                raw,
                min_score=settings.task3_min_score,
                mode="yoloe_vp_lightglue",
                yoloe_min_score=settings.task3_yoloe_min_score,
                modality=decoded.modality,
                yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
                ambiguity_margin=settings.task3_ambiguity_margin,
            )
            filtered_yoloe = filter_no_match_candidates(
                yoloe_raw,
                min_score=settings.task3_min_score,
                mode="yoloe_vp_lightglue",
                yoloe_min_score=settings.task3_yoloe_min_score,
                modality=decoded.modality,
                yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
                ambiguity_margin=settings.task3_ambiguity_margin,
            )
            filtered_orb = filter_no_match_candidates(
                orb_raw,
                min_score=settings.task3_min_score,
                mode="yoloe_vp_lightglue",
                yoloe_min_score=settings.task3_yoloe_min_score,
                modality=decoded.modality,
                yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
                ambiguity_margin=settings.task3_ambiguity_margin,
            )

            if yoloe_raw and orb_raw:
                suppression["frames_with_both_detectors"] += 1
                if filtered_yoloe and filtered_orb and filtered_all:
                    winner = _detector_family(filtered_all[0])
                    if winner == "orb":
                        suppression["yoloe_suppressed_by_orb"] += 1
                        suppression["yoloe_suppressed_cases"].append(
                            {
                                "scenario_id": scenario["id"],
                                "frame_idx": decoded.frame_index,
                                "suppressed_yoloe": _candidate_detail(filtered_yoloe[0]),
                                "winning_orb": _candidate_detail(filtered_orb[0]),
                            }
                        )
                    elif winner == "yoloe":
                        suppression["orb_suppressed_by_yoloe"] += 1
                    else:
                        suppression["suppression_no_effect"] += 1
                else:
                    suppression["suppression_no_effect"] += 1

            verified = verify_matches(
                frame,
                filtered_all,
                decoded_frame=decoded,
                min_inliers=settings.task3_match_min_inliers,
            )
            for candidate in verified:
                family = _detector_family(candidate)
                if family in accepted_scores:
                    accepted_scores[family].append(float(candidate.metadata.get("match_score", 0.0)))
                accepted_by_ref[candidate.object_id] = accepted_by_ref.get(candidate.object_id, 0) + 1

        scenario_summaries[scenario["id"]] = {"accepted_by_ref": accepted_by_ref}

    return {
        "suppression": suppression,
        "score_scale": {key: _score_summary(value) for key, value in accepted_scores.items()},
        "scenario_summaries": scenario_summaries,
    }


def _run_variant_b(settings: MvpRuntimeSettings, scenario: dict[str, Any]) -> dict[str, Any]:
    cache = ReferenceCache()
    cache.preload_from_directory(scenario["references_dir"], orb_features=settings.task3_orb_features)
    backend = YoloeVpLightGlueBackend(
        reference_cache=cache,
        runtime_settings=replace(
            settings,
            task3_mode="yoloe_vp_lightglue",
            task3_reference_dir=Path(scenario["references_dir"]),
            task3_eval_reference_dir=Path(scenario["references_dir"]),
        ),
    )
    all_reference_ids = cache.list_ids()
    eligible_ids = cache.filter_reference_ids_by_detector_modality(all_reference_ids, modality="thermal")
    eligible_yoloe_ids = [reference_id for reference_id in eligible_ids if cache.get_detector(reference_id) != "orb"]

    accepted_total = 0
    accepted_by_ref: dict[str, int] = {}
    frames = iter_video_frames(
        scenario["video"],
        frame_stride=int(scenario["frame_stride"]),
        limit=int(scenario["frame_limit"]) if scenario["frame_limit"] is not None else None,
        video_name=scenario["id"],
    )
    for logical_index, decoded in enumerate(frames):
        frame = _frame_envelope(logical_index=logical_index, decoded=decoded, scenario_id=scenario["id"])
        raw, _task3_info = backend.match(decoded_frame=decoded, reference_ids=all_reference_ids, scenario_id=f"{scenario['id']}_variant_b")
        filtered_refs = [item for item in raw if item.object_id in eligible_yoloe_ids]
        filtered = filter_no_match_candidates(
            filtered_refs,
            min_score=settings.task3_min_score,
            mode="yoloe_vp_lightglue",
            yoloe_min_score=settings.task3_yoloe_min_score,
            modality=decoded.modality,
            yoloe_thermal_min_score=settings.task3_yoloe_thermal_min_score,
            ambiguity_margin=settings.task3_ambiguity_margin,
        )
        verified = verify_matches(frame, filtered, decoded_frame=decoded, min_inliers=settings.task3_match_min_inliers)
        accepted_total += len(verified)
        for candidate in verified:
            accepted_by_ref[candidate.object_id] = accepted_by_ref.get(candidate.object_id, 0) + 1

    return {
        "accepted_total": accepted_total,
        "accepted_by_ref": accepted_by_ref,
        "ref_04_false_positives": accepted_by_ref.get("ref_04", 0),
        "eligible_yoloe_ids": eligible_yoloe_ids,
        "full_vpe_reference_ids": all_reference_ids,
    }


def main() -> None:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    settings = replace(MvpRuntimeSettings(), task3_yoloe_allow_cpu=True, task3_debug_dump_rejects=False)
    manifest = load_task3_manifest(settings.task3_eval_manifest_path)

    detailed_v2 = _run_detailed_v2(settings, manifest)
    thermal_absent = _scenario_by_id(manifest, "thermal_absent_target_proxy_2025")
    variant_b = _run_variant_b(settings, thermal_absent)

    current_v2_summary = _load_json(CURRENT_V2_ARCHIVE / "per_reference_routing_summary.json")
    routing_v1_summary = _load_json(ROUTING_V1_ARCHIVE / "per_reference_routing_summary.json")

    variant_table = {
        "A_current_v2": {
            "thermal_absent_fp": int(current_v2_summary["thermal_absent_target_proxy_2025"]["accepted_by_ref"].get("ref_04", 0))
            + int(current_v2_summary["thermal_absent_target_proxy_2025"]["accepted_by_ref"].get("ref_02", 0))
            + int(current_v2_summary["thermal_absent_target_proxy_2025"]["accepted_by_ref"].get("ref_03", 0)),
            "ref_04_false_positives": int(current_v2_summary["thermal_absent_target_proxy_2025"]["accepted_by_ref"].get("ref_04", 0)),
            "description": "current archived modality-limited routing v2",
        },
        "A_prime_rerun_v2": {
            "thermal_absent_fp": sum(int(value) for value in detailed_v2["scenario_summaries"]["thermal_absent_target_proxy_2025"]["accepted_by_ref"].values()),
            "ref_04_false_positives": int(detailed_v2["scenario_summaries"]["thermal_absent_target_proxy_2025"]["accepted_by_ref"].get("ref_04", 0)),
            "description": "fresh rerun of current v2",
        },
        "B_full_vpe_bank_emission_filter": {
            "thermal_absent_fp": int(variant_b["accepted_total"]),
            "ref_04_false_positives": int(variant_b["ref_04_false_positives"]),
            "description": "all refs in VPE bank, ref_02/ref_03 filtered after emission",
        },
        "C_routing_v1_archived": {
            "thermal_absent_fp": sum(int(value) for value in routing_v1_summary["thermal_absent_target_proxy_2025"]["accepted_by_ref"].values()),
            "ref_04_false_positives": int(routing_v1_summary["thermal_absent_target_proxy_2025"]["accepted_by_ref"].get("ref_04", 0)),
            "description": "archived routing v1 reference point",
        },
    }

    report = {
        "variant_comparison": variant_table,
        "suppression_scope": detailed_v2["suppression"],
        "score_scale": detailed_v2["score_scale"],
    }
    (ARCHIVE_ROOT / "cross_detector_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Cross-Detector Diagnostics",
        "",
        "## Variant Comparison",
        "| Variant | Description | Thermal absent FP | ref_04 FPs |",
        "| --- | --- | --- | --- |",
    ]
    for key, payload in variant_table.items():
        lines.append(
            f"| {key} | {payload['description']} | {payload['thermal_absent_fp']} | {payload['ref_04_false_positives']} |"
        )

    suppression = detailed_v2["suppression"]
    lines.extend(
        [
            "",
            "## Suppression Scope",
            f"- Frames with both-detector candidates: {suppression['frames_with_both_detectors']}",
            f"- YOLOE suppressed by ORB: {suppression['yoloe_suppressed_by_orb']}",
            f"- ORB suppressed by YOLOE: {suppression['orb_suppressed_by_yoloe']}",
            f"- Suppression no effect: {suppression['suppression_no_effect']}",
            "",
            "## Score Scale",
            f"- YOLOE accepted: {json.dumps(detailed_v2['score_scale']['yoloe'])}",
            f"- ORB accepted: {json.dumps(detailed_v2['score_scale']['orb'])}",
            "",
            "## YOLOE Suppressed Cases",
        ]
    )
    if suppression["yoloe_suppressed_cases"]:
        for case in suppression["yoloe_suppressed_cases"]:
            lines.append(
                "- {scenario} frame {frame}: YOLOE {yoloe_obj} score={yoloe_score:.4f} conf={yoloe_conf:.4f} "
                "matches={yoloe_matches} inlier={yoloe_inlier:.4f} lost to ORB {orb_obj} score={orb_score:.4f} matches={orb_matches}".format(
                    scenario=case["scenario_id"],
                    frame=case["frame_idx"],
                    yoloe_obj=case["suppressed_yoloe"]["object_id"],
                    yoloe_score=case["suppressed_yoloe"]["score"],
                    yoloe_conf=case["suppressed_yoloe"]["confidence"],
                    yoloe_matches=case["suppressed_yoloe"]["match_count"],
                    yoloe_inlier=case["suppressed_yoloe"]["inlier_ratio"],
                    orb_obj=case["winning_orb"]["object_id"],
                    orb_score=case["winning_orb"]["score"],
                    orb_matches=case["winning_orb"]["match_count"],
                )
            )
    else:
        lines.append("- none")

    (ARCHIVE_ROOT / "cross_detector_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
