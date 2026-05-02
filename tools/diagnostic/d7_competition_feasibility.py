from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task3.experimental.backend import YoloeVpLightGlueBackend
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches
from tools.diagnostic.d7_common import OUTPUT_DIR, PASS_A_SCENARIO_IDS, REFERENCE_DIR, frame_envelope, iter_timed_video_frames, load_scenarios, make_settings

FRAMES_PER_COMPETITION = 2250
D5_OUTPUT_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-04-27_d5_mixed_bank_integration"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(percentile, 1.0)) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _minutes_from_ms(mean_ms: float) -> float:
    return (mean_ms * FRAMES_PER_COMPETITION) / 1000.0 / 60.0


def _load_current_baseline() -> dict[str, Any]:
    pass_a = json.loads((D5_OUTPUT_DIR / "pass_a_results.json").read_text(encoding="utf-8"))
    latency = json.loads((D5_OUTPUT_DIR / "latency_real.json").read_text(encoding="utf-8"))
    breakdown = json.loads((OUTPUT_DIR / "latency_breakdown.json").read_text(encoding="utf-8"))
    decode_mean = _safe_mean([float(item["decode_ms"]) for item in breakdown.get("frame_records", [])])
    decode_p50 = _percentile([float(item["decode_ms"]) for item in breakdown.get("frame_records", [])], 0.50)
    decode_p95 = _percentile([float(item["decode_ms"]) for item in breakdown.get("frame_records", [])], 0.95)
    homography_mean = _safe_mean(
        [
            float(frame["homography_ms"])
            for scenario in pass_a.values()
            for frame in scenario.get("frame_records", [])
        ]
    )
    return {
        "pass_a": pass_a,
        "latency": latency,
        "decode_mean_ms": decode_mean,
        "decode_p50_ms": decode_p50,
        "decode_p95_ms": decode_p95,
        "homography_mean_ms": homography_mean,
    }


def _current_accuracy_note(pass_a: dict[str, Any]) -> str:
    rgb_ref = int(pass_a["rgb_reference_session"]["guard_accept_count"])
    thermal_cross = int(pass_a["thermal_cross_sensor_proxy"]["guard_accept_count"])
    rgb_abs = int(pass_a["rgb_absent_target_proxy_2025"]["guard_accept_count"])
    thermal_abs = int(pass_a["thermal_absent_target_proxy_2025"]["guard_accept_count"])
    return f"guard=`{rgb_ref}/0/{rgb_abs}/{thermal_abs}` on rgb_ref/thermal_cross/rgb_abs/thermal_abs"


def _evaluate_matcher_config(
    *,
    config_id: str,
    mode: str,
    settings_overrides: dict[str, object] | None = None,
) -> dict[str, Any]:
    scenarios = load_scenarios(scenario_ids=PASS_A_SCENARIO_IDS)
    all_frame_times: list[float] = []
    per_scenario: dict[str, dict[str, Any]] = {}

    for scenario_id, scenario in scenarios.items():
        settings = make_settings(REFERENCE_DIR, **(settings_overrides or {}))
        cache = ReferenceCache()
        cache.preload_from_directory(REFERENCE_DIR, orb_features=settings.task3_orb_features)
        matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
        reference_ids = cache.list_ids()
        expected_set = set(scenario.expected_present_refs)

        total_accepts = 0
        guard_accepts = 0
        unexpected_accepts = 0
        frame_count = 0
        for decoded_frame, decode_ms in iter_timed_video_frames(
            scenario.video,
            frame_stride=scenario.frame_stride,
            limit=int(scenario.frame_limit) if scenario.frame_limit is not None else None,
            video_name=scenario.scenario_id,
        ):
            frame_count += 1
            frame = frame_envelope(scenario.scenario_id, decoded_frame.frame_index, decoded_frame.width, decoded_frame.height, tag=config_id)
            match_ids = reference_ids
            if mode != "yoloe_vp_lightglue":
                match_ids = cache.filter_reference_ids_by_detector_modality(reference_ids, modality=decoded_frame.modality)
            started = time.perf_counter()
            raw_matches = matcher.match(frame, b"", match_ids, decoded_frame=decoded_frame, mode=mode)
            filtered = filter_no_match_candidates(
                raw_matches,
                min_score=settings.task3_min_score,
                mode=mode,
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
            wall_ms = (time.perf_counter() - started) * 1000.0
            all_frame_times.append(float(decode_ms + wall_ms))
            accepted_ids = [str(item.object_id) for item in verified]
            total_accepts += len(accepted_ids)
            if scenario.reference_mode == "synthetic_absent":
                guard_accepts += len(accepted_ids)
            else:
                guard_accepts += len([item for item in accepted_ids if item in expected_set])
                unexpected_accepts += len([item for item in accepted_ids if item not in expected_set])
        per_scenario[scenario_id] = {
            "frames": frame_count,
            "total_accepts": total_accepts,
            "guard_accepts": guard_accepts,
            "unexpected_accepts": unexpected_accepts,
        }

    return {
        "config_id": config_id,
        "measurement": "measured",
        "mean_end_to_end_ms": _safe_mean(all_frame_times),
        "p50_end_to_end_ms": _percentile(all_frame_times, 0.50),
        "p95_end_to_end_ms": _percentile(all_frame_times, 0.95),
        "max_end_to_end_ms": float(max(all_frame_times)) if all_frame_times else 0.0,
        "competition_minutes": _minutes_from_ms(_safe_mean(all_frame_times)),
        "within_60_min": _minutes_from_ms(_safe_mean(all_frame_times)) <= 60.0,
        "per_scenario": per_scenario,
    }


def _evaluate_yoloe_detect_only() -> dict[str, Any]:
    scenarios = load_scenarios(scenario_ids=PASS_A_SCENARIO_IDS)
    all_frame_times: list[float] = []
    per_scenario: dict[str, dict[str, Any]] = {}

    for scenario_id, scenario in scenarios.items():
        settings = make_settings(REFERENCE_DIR)
        cache = ReferenceCache()
        cache.preload_from_directory(REFERENCE_DIR, orb_features=settings.task3_orb_features)
        backend = YoloeVpLightGlueBackend(reference_cache=cache, runtime_settings=settings)
        reference_ids = cache.list_ids()
        raw_kept_total = 0
        frame_count = 0
        for decoded_frame, decode_ms in iter_timed_video_frames(
            scenario.video,
            frame_stride=scenario.frame_stride,
            limit=int(scenario.frame_limit) if scenario.frame_limit is not None else None,
            video_name=scenario.scenario_id,
        ):
            frame_count += 1
            active_ids = cache.filter_reference_ids_by_detector_modality(reference_ids, modality=decoded_frame.modality)
            yoloe_ids, _orb_ids = cache.split_reference_ids_by_detector(active_ids)
            started = time.perf_counter()
            kept_count = 0
            if yoloe_ids:
                backend._ensure_ready(yoloe_ids)
                results = backend.model.predict(
                    decoded_frame.bgr,
                    conf=settings.task3_yoloe_conf,
                    iou=settings.task3_yoloe_iou,
                    imgsz=settings.task3_yoloe_imgsz,
                    max_det=max(settings.task3_yoloe_max_det_per_class * max(len(yoloe_ids), 1), 1),
                    verbose=False,
                    device=backend.device,
                )
                if results:
                    prediction = results[0]
                    boxes = getattr(prediction, "boxes", None)
                    if boxes is not None and len(boxes) > 0 and backend.reference_bank is not None and backend.reference_bank.ref_names is not None:
                        raw_classes = boxes.cls.cpu().numpy().astype(int)
                        raw_confidences = boxes.conf.cpu().numpy()
                        allowed_ids = set(yoloe_ids)
                        grouped: dict[int, list[int]] = {}
                        for index, class_id in enumerate(raw_classes):
                            if not 0 <= int(class_id) < len(backend.reference_bank.ref_names):
                                continue
                            ref_name = backend.reference_bank.ref_names[int(class_id)]
                            if ref_name not in allowed_ids:
                                continue
                            grouped.setdefault(int(class_id), []).append(index)
                        for indices in grouped.values():
                            sorted_indices = sorted(indices, key=lambda idx: -float(raw_confidences[idx]))
                            kept_count += len(sorted_indices[: settings.task3_yoloe_max_det_per_class])
            wall_ms = (time.perf_counter() - started) * 1000.0
            all_frame_times.append(float(decode_ms + wall_ms))
            raw_kept_total += int(kept_count)
        per_scenario[scenario_id] = {
            "frames": frame_count,
            "raw_kept_candidates": raw_kept_total,
            "raw_kept_per_frame": 0.0 if frame_count == 0 else raw_kept_total / frame_count,
        }

    return {
        "config_id": "yoloe_detect_only_estimate",
        "measurement": "measured_timing_not_comparable_accuracy",
        "mean_end_to_end_ms": _safe_mean(all_frame_times),
        "p50_end_to_end_ms": _percentile(all_frame_times, 0.50),
        "p95_end_to_end_ms": _percentile(all_frame_times, 0.95),
        "max_end_to_end_ms": float(max(all_frame_times)) if all_frame_times else 0.0,
        "competition_minutes": _minutes_from_ms(_safe_mean(all_frame_times)),
        "within_60_min": _minutes_from_ms(_safe_mean(all_frame_times)) <= 60.0,
        "per_scenario": per_scenario,
    }


def _scenario_note(per_scenario: dict[str, dict[str, Any]]) -> str:
    rgb_ref = per_scenario["rgb_reference_session"]["guard_accepts"]
    thermal_cross = per_scenario["thermal_cross_sensor_proxy"]["guard_accepts"]
    rgb_abs = per_scenario["rgb_absent_target_proxy_2025"]["guard_accepts"]
    thermal_abs = per_scenario["thermal_absent_target_proxy_2025"]["guard_accepts"]
    return f"guard=`{rgb_ref}/{thermal_cross}/{rgb_abs}/{thermal_abs}`"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _load_current_baseline()
    current_latency = baseline["latency"]["overall"]["current"]["wall_ms"]
    current_mean_ms = float(current_latency["mean"]) + float(baseline["decode_mean_ms"])
    current_p50_ms = float(current_latency["p50"]) + float(baseline["decode_p50_ms"])
    current_p95_ms = float(current_latency["p95"]) + float(baseline["decode_p95_ms"])
    current_minutes = _minutes_from_ms(current_mean_ms)

    orb_only = _evaluate_matcher_config(config_id="orb_only", mode="orb_template")
    lightglue_cap1 = _evaluate_matcher_config(
        config_id="lightglue_cap1",
        mode="yoloe_vp_lightglue",
        settings_overrides={"task3_yoloe_max_det_per_class": 1},
    )
    yoloe_only = _evaluate_yoloe_detect_only()
    gpu_runtime = json.loads((OUTPUT_DIR / "gpu_runtime_check.json").read_text(encoding="utf-8"))

    current_row = {
        "config_id": "current_cpu_full",
        "measurement": "measured_from_d6_current_plus_d7_decode",
        "mean_end_to_end_ms": current_mean_ms,
        "p50_end_to_end_ms": current_p50_ms,
        "p95_end_to_end_ms": current_p95_ms,
        "max_end_to_end_ms": float(current_latency["max"]),
        "competition_minutes": current_minutes,
        "within_60_min": current_minutes <= 60.0,
        "note": _current_accuracy_note(baseline["pass_a"]),
    }

    gpu_row = {
        "config_id": "gpu_full",
        "measurement": "unavailable",
        "mean_end_to_end_ms": None,
        "p50_end_to_end_ms": None,
        "p95_end_to_end_ms": None,
        "max_end_to_end_ms": None,
        "competition_minutes": None,
        "within_60_min": None,
        "note": "GPU visible via nvidia-smi but torch runtime is CPU-only",
    }
    if gpu_runtime.get("gpu_12ref_sample") is not None:
        sample = gpu_runtime["gpu_12ref_sample"]
        gpu_row.update(
            {
                "measurement": "single_frame_measured",
                "mean_end_to_end_ms": float(sample["wall_ms"]),
                "p50_end_to_end_ms": float(sample["wall_ms"]),
                "p95_end_to_end_ms": float(sample["wall_ms"]),
                "max_end_to_end_ms": float(sample["wall_ms"]),
                "competition_minutes": _minutes_from_ms(float(sample["wall_ms"])),
                "within_60_min": _minutes_from_ms(float(sample["wall_ms"])) <= 60.0,
                "note": "single-frame sample only",
            }
        )

    rows = [current_row, gpu_row, yoloe_only, orb_only, lightglue_cap1]
    (OUTPUT_DIR / "competition_feasibility.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    fastest = None
    measurable_rows = [row for row in rows if isinstance(row.get("mean_end_to_end_ms"), (float, int))]
    if measurable_rows:
        fastest = min(measurable_rows, key=lambda row: float(row["mean_end_to_end_ms"]))

    lines = [
        "# D7 Competition Feasibility",
        "",
        "## Direct Answer",
        "",
        f"- Current full CPU configuration is not competition-feasible. Estimated total time is `{current_minutes:.1f} minutes` for `{FRAMES_PER_COMPETITION}` frames.",
        f"- Fastest measurable configuration in this environment is `{fastest['config_id']}` at `{float(fastest['competition_minutes']):.1f} minutes`." if fastest is not None else "- No measurable configuration result was available.",
        "- `YOLOE detect only` timing is useful for speed planning, but its accept semantics are not comparable to the production pipeline because LightGlue scoring is removed.",
        "",
        "## Configuration Table",
        "",
        "| Config | Measurement | Mean ms/frame | P50 ms | P95 ms | Total Minutes | Within 60 min | Accuracy / Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        mean_value = "-" if row.get("mean_end_to_end_ms") is None else f"{float(row['mean_end_to_end_ms']):.1f}"
        p50_value = "-" if row.get("p50_end_to_end_ms") is None else f"{float(row['p50_end_to_end_ms']):.1f}"
        p95_value = "-" if row.get("p95_end_to_end_ms") is None else f"{float(row['p95_end_to_end_ms']):.1f}"
        minutes_value = "-" if row.get("competition_minutes") is None else f"{float(row['competition_minutes']):.1f}"
        within_value = "-" if row.get("within_60_min") is None else ("yes" if bool(row["within_60_min"]) else "no")
        if row.get("config_id") == "orb_only":
            note = _scenario_note(row["per_scenario"])
        elif row.get("config_id") == "lightglue_cap1":
            note = _scenario_note(row["per_scenario"])
        elif row.get("config_id") == "yoloe_detect_only_estimate":
            rgb_kept = row["per_scenario"]["rgb_reference_session"]["raw_kept_per_frame"]
            thermal_kept = row["per_scenario"]["thermal_cross_sensor_proxy"]["raw_kept_per_frame"]
            note = f"raw_kept/frame=`{rgb_kept:.2f}` rgb, `{thermal_kept:.2f}` thermal"
        else:
            note = str(row.get("note") or "")
        lines.append(
            f"| {row['config_id']} | {row['measurement']} | {mean_value} | {p50_value} | {p95_value} | {minutes_value} | {within_value} | {note} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- `current_cpu_full` uses the stable D6 current baseline latency and adds the D7 decode mean, so it avoids the intrusive profiling overhead from the breakdown probe.",
            "- `lightglue_cap1` means `task3_yoloe_max_det_per_class=1` only; scoring weights and thresholds stay unchanged.",
            "- `orb_only` is measured with the current modality filter still applied, then all active references are matched through the ORB/template path.",
            "- `gpu_full` is unavailable in the current environment because Torch cannot see CUDA even though the OS can see the RTX 5060 Laptop GPU.",
            "",
        ]
    )

    (OUTPUT_DIR / "competition_feasibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote competition feasibility to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
