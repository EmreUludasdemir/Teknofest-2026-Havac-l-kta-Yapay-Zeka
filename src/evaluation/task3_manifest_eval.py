from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from src.config.settings import MvpRuntimeSettings
from src.evaluation.task2_long_sequence import iter_video_frames
from src.evaluation.task3_baseline import evaluate_task3_frames
from src.tools.report_paths import GENERATED_REPORTS_ROOT


def load_task3_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = [_normalize_manifest_scenario(item, index=index) for index, item in enumerate(payload.get("scenarios", []), start=1)]
    return {"scenarios": scenarios}


def evaluate_task3_manifest(
    *,
    runtime_settings: MvpRuntimeSettings | None = None,
    manifest_path: str | Path | None = None,
    output_dir: str | Path = GENERATED_REPORTS_ROOT,
    modes: tuple[str, ...] = ("orb_template", "yoloe_vp_lightglue"),
    scenario_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    settings = runtime_settings or MvpRuntimeSettings()
    manifest = load_task3_manifest(manifest_path or settings.task3_eval_manifest_path)
    scenarios = manifest.get("scenarios", [])
    if scenario_ids:
        allowed = set(scenario_ids)
        scenarios = [item for item in scenarios if item["id"] in allowed]
        if not scenarios:
            requested = ", ".join(sorted(allowed))
            raise ValueError(f"Task3 manifest scenario filter matched no entries: {requested}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mode_payloads: dict[str, dict[str, Any]] = {}
    for mode in modes:
        results: list[dict[str, Any]] = []
        for scenario in scenarios:
            scenario_settings = replace(
                settings,
                task3_mode=mode,
                task3_eval_reference_dir=Path(scenario["references_dir"]),
                task3_reference_dir=Path(scenario["references_dir"]),
            )
            try:
                frames = iter_video_frames(
                    scenario["video"],
                    frame_stride=int(scenario["frame_stride"]),
                    limit=int(scenario["frame_limit"]) if scenario["frame_limit"] is not None else None,
                    video_name=scenario["id"],
                )
                summary = evaluate_task3_frames(
                    frames,
                    runtime_settings=scenario_settings,
                    reference_dir=scenario["references_dir"],
                    video_name=scenario["id"],
                    mode=mode,
                )
            except Exception as exc:
                summary = {
                    "video_name": scenario["id"],
                    "status": "failed",
                    "error": str(exc),
                    "accepted_match_count": 0,
                    "false_positive_proxy_count": 0,
                    "no_match_suppression_rate": 1.0,
                }

            summary.update(
                {
                    "scenario_id": scenario["id"],
                    "video": scenario["video"],
                    "references_dir": scenario["references_dir"],
                    "reference_mode": scenario["reference_mode"],
                    "ground_truth": scenario["ground_truth"],
                    "expected_present_refs": scenario["expected_present_refs"],
                    "tags": scenario["tags"],
                }
            )
            if scenario["reference_mode"] == "synthetic_absent":
                summary["false_positive_proxy_count"] = int(summary.get("accepted_match_count", 0))
            results.append(summary)

        aggregate = {
            "scenario_count": len(results),
            "ok_count": sum(1 for item in results if item.get("status") == "ok"),
            "accepted_match_count": int(sum(int(item.get("accepted_match_count", 0)) for item in results)),
            "false_positive_proxy_count": int(sum(int(item.get("false_positive_proxy_count", 0)) for item in results)),
            "no_match_suppression_rate": round(_safe_mean(item.get("no_match_suppression_rate", 0.0) for item in results), 6),
            "mode": mode,
        }
        payload = {"mode": mode, "results": results, "aggregate": aggregate}
        suffix = mode.replace("-", "_")
        (output_path / f"task3_manifest_{suffix}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mode_payloads[mode] = payload

    comparison_rows = _build_comparison_rows(mode_payloads)
    comparison = {"modes": mode_payloads, "comparison_rows": comparison_rows}
    (output_path / "task3_manifest_comparison.md").write_text(_render_comparison_markdown(comparison_rows), encoding="utf-8")
    (output_path / "task3_manifest_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def _normalize_manifest_scenario(raw: dict[str, Any], *, index: int) -> dict[str, Any]:
    scenario_id = str(raw.get("id") or raw.get("scenario_id") or f"scenario_{index:02d}")
    video = str(raw.get("video") or raw.get("video_path") or "")
    references_dir = str(raw.get("references_dir") or raw.get("reference_dir") or MvpRuntimeSettings().task3_eval_reference_dir)
    if not video:
        raise ValueError(f"Task3 manifest scenario missing video path: {scenario_id}")
    target_expected = raw.get("target_expected")
    reference_mode = str(raw.get("reference_mode") or ("synthetic_absent" if target_expected is False else "present_targets"))
    return {
        "id": scenario_id,
        "video": video,
        "references_dir": references_dir,
        "reference_mode": reference_mode,
        "frame_stride": int(raw.get("frame_stride", MvpRuntimeSettings().task3_eval_frame_stride)),
        "frame_limit": raw.get("frame_limit", MvpRuntimeSettings().task3_eval_frame_limit),
        "ground_truth": raw.get("ground_truth"),
        "expected_present_refs": list(raw.get("expected_present_refs", [])),
        "tags": list(raw.get("tags", [])),
    }


def _build_comparison_rows(mode_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    orb_payload = mode_payloads.get("orb_template", {})
    yoloe_payload = mode_payloads.get("yoloe_vp_lightglue", {})
    orb_results = {str(item.get("scenario_id")): item for item in orb_payload.get("results", [])}
    yoloe_results = {str(item.get("scenario_id")): item for item in yoloe_payload.get("results", [])}
    scenario_ids = sorted(set(orb_results) | set(yoloe_results))

    rows: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        orb = orb_results.get(scenario_id, {})
        yoloe = yoloe_results.get(scenario_id, {})
        rows.append(
            {
                "scenario_id": scenario_id,
                "reference_mode": orb.get("reference_mode") or yoloe.get("reference_mode") or "-",
                "orb_accepted": orb.get("accepted_match_count", "-"),
                "yoloe_accepted": yoloe.get("accepted_match_count", "-"),
                "orb_false_positive_proxy": orb.get("false_positive_proxy_count", "-"),
                "yoloe_false_positive_proxy": yoloe.get("false_positive_proxy_count", "-"),
                "orb_no_match_rate": orb.get("no_match_suppression_rate", "-"),
                "yoloe_no_match_rate": yoloe.get("no_match_suppression_rate", "-"),
                "yoloe_effective_mode": _summarize_counts(yoloe.get("effective_mode_counts", {})),
                "yoloe_fallback_reason": yoloe.get("fallback_reason") or "-",
                "yoloe_candidates_generated": yoloe.get("candidates_generated", "-"),
                "yoloe_gate_rejected": yoloe.get("candidates_rejected_by_gate", "-"),
                "yoloe_score_filter_rejected": yoloe.get("candidates_rejected_by_score_filter", "-"),
                "yoloe_candidate_rejected_ratio": yoloe.get("candidate_rejected_ratio", "-"),
                "yoloe_inference_ms_per_frame_avg": yoloe.get("yoloe_inference_ms_per_frame_avg", "-"),
                "lightglue_verify_ms_total_per_frame_avg": yoloe.get("lightglue_verify_ms_total_per_frame_avg", "-"),
                "accepted_delta_yoloe_minus_orb": _safe_delta(yoloe.get("accepted_match_count"), orb.get("accepted_match_count")),
            }
        )
    return rows


def _render_comparison_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Scenario | Reference Mode | ORB Accepted | YOLOE Accepted | Delta | YOLOE Effective | YOLOE Fallback | YOLOE Cand Gen | YOLOE Gate Rej | YOLOE Score Rej | YOLOE Rej Ratio | YOLOE Infer ms/frame | LG Verify ms/frame | ORB FP Proxy | YOLOE FP Proxy | ORB No-match | YOLOE No-match |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {reference_mode} | {orb_acc} | {yoloe_acc} | {delta} | {effective} | {fallback} | {cand_gen} | {gate_rej} | {score_rej} | {rej_ratio} | {infer_ms} | {verify_ms} | {orb_fp} | {yoloe_fp} | {orb_nm} | {yoloe_nm} |".format(
                scenario=row.get("scenario_id"),
                reference_mode=row.get("reference_mode"),
                orb_acc=row.get("orb_accepted"),
                yoloe_acc=row.get("yoloe_accepted"),
                delta=row.get("accepted_delta_yoloe_minus_orb"),
                effective=row.get("yoloe_effective_mode"),
                fallback=row.get("yoloe_fallback_reason"),
                cand_gen=row.get("yoloe_candidates_generated"),
                gate_rej=row.get("yoloe_gate_rejected"),
                score_rej=row.get("yoloe_score_filter_rejected"),
                rej_ratio=row.get("yoloe_candidate_rejected_ratio"),
                infer_ms=row.get("yoloe_inference_ms_per_frame_avg"),
                verify_ms=row.get("lightglue_verify_ms_total_per_frame_avg"),
                orb_fp=row.get("orb_false_positive_proxy"),
                yoloe_fp=row.get("yoloe_false_positive_proxy"),
                orb_nm=row.get("orb_no_match_rate"),
                yoloe_nm=row.get("yoloe_no_match_rate"),
            )
        )
    return "\n".join(lines) + "\n"


def _safe_mean(values: Iterable[float]) -> float:
    filtered = [float(value) for value in values]
    return mean(filtered) if filtered else 0.0


def _summarize_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "-"
    if len(counts) == 1:
        return next(iter(counts))
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _safe_delta(left: Any, right: Any) -> str:
    try:
        return str(int(left) - int(right))
    except Exception:
        return "-"
