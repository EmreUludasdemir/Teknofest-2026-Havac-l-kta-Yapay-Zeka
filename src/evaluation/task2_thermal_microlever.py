from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame
from src.core.utils import infer_modality, percentile
from src.core.vision import is_cv2_available
from src.evaluation.task2_long_sequence import Task2CsvRecord
from src.task2.estimator import Task2Estimator
from src.task2.health_logic import resolve_task2_translation

if is_cv2_available():  # pragma: no branch
    from src.core.vision import cv2, np
else:  # pragma: no cover
    cv2 = None
    np = None


TARGET_SCENARIO_ID = "thermal_long_degraded_1"
THERMAL_TARGET_MIN_IMPROVEMENT_RATIO = 0.05
NON_REGRESSION_MAX_RATIO = 0.01
RECOVERY_ERROR_MAX_DELTA = 0.05
RECOVERY_CONTINUITY_MAX_RATIO = 0.05
RUNTIME_P95_MAX_RATIO = 1.10


@dataclass(slots=True)
class Task2ThermalMicroleverScenario:
    scenario_id: str
    csv_path: Path
    video_path: Path
    frame_start: int
    frame_limit: int
    frame_stride: int
    health_segments: list[dict[str, Any]]
    transform_profile: str
    tags: list[str]

    def health_schedule(self) -> list[str]:
        schedule: list[str] = []
        for item in self.health_segments:
            schedule.extend([str(item["health"])] * int(item["length"]))
        return schedule[: self.frame_limit]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_manifest_path() -> Path:
    return Path(__file__).with_name("task2_thermal_microlever_manifest.json")


def _resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return _project_root() / path


def load_task2_thermal_microlever_manifest(
    manifest_path: str | Path | None = None,
) -> list[Task2ThermalMicroleverScenario]:
    payload = json.loads(Path(manifest_path or _default_manifest_path()).read_text(encoding="utf-8"))
    scenarios: list[Task2ThermalMicroleverScenario] = []
    for item in payload.get("scenarios", []):
        scenario = Task2ThermalMicroleverScenario(
            scenario_id=str(item["scenario_id"]),
            csv_path=_resolve_repo_path(item["csv_path"]),
            video_path=_resolve_repo_path(item["video_path"]),
            frame_start=int(item["frame_start"]),
            frame_limit=int(item["frame_limit"]),
            frame_stride=int(item["frame_stride"]),
            health_segments=[dict(segment) for segment in item["health_segments"]],
            transform_profile=str(item.get("transform_profile", "none")),
            tags=[str(tag) for tag in item.get("tags", [])],
        )
        if len(scenario.health_schedule()) != scenario.frame_limit:
            raise ValueError(f"health schedule length mismatch: {scenario.scenario_id}")
        scenarios.append(scenario)
    return scenarios


def build_variant_settings(base: MvpRuntimeSettings, variant: str) -> MvpRuntimeSettings:
    if variant == "baseline":
        return replace(base)
    if variant == "thermal_confidence_floor_045":
        return replace(base, task2_confidence_floor_thermal=0.45)
    if variant == "thermal_sensor_hint_weight_008":
        return replace(base, task2_sensor_hint_max_weight_thermal=0.08)
    raise ValueError(f"unknown variant: {variant}")


def load_scenario_records(scenario: Task2ThermalMicroleverScenario) -> list[Task2CsvRecord]:
    rows: list[dict[str, str]] = []
    with scenario.csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index < scenario.frame_start:
                continue
            if (row_index - scenario.frame_start) % max(scenario.frame_stride, 1) != 0:
                continue
            rows.append(row)
            if len(rows) >= scenario.frame_limit:
                break
    schedule = scenario.health_schedule()
    return [
        Task2CsvRecord(
            frame_id=str(row.get("frame_numbers", f"frame_{index:06d}")),
            translation_x=float(row["translation_x"]),
            translation_y=float(row["translation_y"]),
            translation_z=float(row["translation_z"]),
            health_status=schedule[index],
        )
        for index, row in enumerate(rows)
    ]


def iter_scenario_frames(scenario: Task2ThermalMicroleverScenario) -> Iterator[DecodedFrame]:
    if not is_cv2_available():
        raise RuntimeError("OpenCV gerekli")
    capture = cv2.VideoCapture(str(scenario.video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video acilamadi: {scenario.video_path}")
    frame_index = 0
    emitted = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index < scenario.frame_start:
                frame_index += 1
                continue
            if (frame_index - scenario.frame_start) % max(scenario.frame_stride, 1) != 0:
                frame_index += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape[:2]
            yield DecodedFrame(
                bgr=frame,
                gray=gray,
                width=int(width),
                height=int(height),
                channel_count=int(frame.shape[2]) if len(frame.shape) == 3 else 1,
                modality=infer_modality(scenario.video_path.stem, width=int(width), height=int(height)),
                frame_index=frame_index,
            )
            emitted += 1
            frame_index += 1
            if emitted >= scenario.frame_limit:
                break
    finally:
        capture.release()


def apply_eval_transform(
    decoded_frame: DecodedFrame,
    *,
    transform_profile: str,
    frame_offset: int,
    health_status: str,
    previous_transformed_gray=None,
    sensor_hint: dict[str, float] | None = None,
) -> tuple[DecodedFrame, dict[str, float], dict[str, object]]:
    hint = dict(sensor_hint or {})
    info = {
        "transform_profile": transform_profile,
        "frame_transform_applied": False,
        "sensor_hint_weakened": False,
    }
    if transform_profile == "none":
        return decoded_frame, hint, info
    if not is_cv2_available() or decoded_frame.gray is None:
        return decoded_frame, hint, info

    transformed = decoded_frame
    if transform_profile == "low_contrast_noise" and health_status == "0":
        rng = np.random.default_rng(frame_offset + 2026)
        compressed = cv2.normalize(decoded_frame.gray, None, alpha=96, beta=160, norm_type=cv2.NORM_MINMAX)
        noise = rng.normal(0.0, 10.0, size=compressed.shape).astype(np.float32)
        transformed_gray = np.clip(compressed.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        transformed = replace(decoded_frame, gray=transformed_gray)
        info["frame_transform_applied"] = True
    elif (
        transform_profile == "freeze_every_6"
        and health_status == "0"
        and previous_transformed_gray is not None
        and (frame_offset + 1) % 6 == 0
    ):
        transformed = replace(decoded_frame, gray=previous_transformed_gray.copy())
        info["frame_transform_applied"] = True

    if transform_profile == "weak_sensor_hint" and health_status == "0" and hint:
        sign = -1.0 if frame_offset % 2 == 0 else 1.0
        hint["translation_x"] = float(hint.get("translation_x", 0.0)) + (4.5 * sign)
        hint["translation_y"] = float(hint.get("translation_y", 0.0)) - (4.5 * sign)
        hint["translation_z"] = float(hint.get("translation_z", 0.0)) + (0.6 * sign)
        info["sensor_hint_weakened"] = True
    return transformed, hint, info


def evaluate_task2_thermal_microlever(
    *,
    manifest_path: str | Path | None = None,
    runtime_settings: MvpRuntimeSettings | None = None,
    output_dir: str | Path = "reports",
    scenario_id: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    scenarios = load_task2_thermal_microlever_manifest(manifest_path)
    if scenario_id is not None:
        scenarios = [item for item in scenarios if item.scenario_id == scenario_id]
    selected_variants = _selected_variants(variant)
    base_settings = runtime_settings or MvpRuntimeSettings()
    variant_payloads: dict[str, Any] = {}
    for current_variant in selected_variants:
        settings = build_variant_settings(base_settings, current_variant)
        scenario_results = [_evaluate_variant_scenario(item, settings=settings, variant=current_variant) for item in scenarios]
        variant_payloads[current_variant] = {
            "variant": current_variant,
            "runtime_settings": _render_settings(settings),
            "scenarios": scenario_results,
            "aggregate": _aggregate_variant(scenario_results),
        }
    comparisons = _build_comparisons(variant_payloads)
    final_recommendation = _final_recommendation(comparisons)
    payload = {
        "manifest_path": str(manifest_path or _default_manifest_path()),
        "variants": variant_payloads,
        "comparisons": comparisons,
        "final_recommendation": final_recommendation,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "task2_thermal_microlever_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_path / "task2_thermal_microlever_results.md").write_text(_render_results_markdown(payload), encoding="utf-8")
    (output_path / "task2_thermal_microlever_design.md").write_text(
        _render_design_markdown(scenarios, variant_payloads),
        encoding="utf-8",
    )
    (output_path / "task2_thermal_microlever_recommendation.md").write_text(
        _render_recommendation_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _selected_variants(variant: str | None) -> list[str]:
    ordered = ["baseline", "thermal_confidence_floor_045", "thermal_sensor_hint_weight_008"]
    if variant is None or variant == "baseline":
        return ordered if variant is None else ["baseline"]
    return ["baseline", variant]


def _evaluate_variant_scenario(
    scenario: Task2ThermalMicroleverScenario,
    *,
    settings: MvpRuntimeSettings,
    variant: str,
) -> dict[str, Any]:
    if not scenario.csv_path.exists():
        return {"scenario_id": scenario.scenario_id, "status": "missing_csv", "variant": variant, "tags": list(scenario.tags)}
    if not scenario.video_path.exists():
        return {"scenario_id": scenario.scenario_id, "status": "missing_video", "variant": variant, "tags": list(scenario.tags)}
    records = load_scenario_records(scenario)
    estimator = Task2Estimator(runtime_settings=settings)
    frames = iter_scenario_frames(scenario)
    transition_indices = _transition_window_indices([item.health_status for item in records])
    all_errors: list[float] = []
    health0_errors: list[float] = []
    transition_errors: list[float] = []
    recovery_errors: list[float] = []
    recovery_continuity_errors: list[float] = []
    latencies: list[float] = []
    catastrophic_jumps = 0
    previous_output: tuple[float, float, float] | None = None
    previous_ground_truth: tuple[float, float, float] | None = None
    previous_health: str | None = None
    previous_transformed_gray = None
    for index, record in enumerate(records):
        decoded = next(frames)
        sensor_hint = {
            "translation_x": record.translation_x,
            "translation_y": record.translation_y,
            "translation_z": record.translation_z,
        }
        decoded_frame, hint, _ = apply_eval_transform(
            decoded,
            transform_profile=scenario.transform_profile,
            frame_offset=index,
            health_status=record.health_status,
            previous_transformed_gray=previous_transformed_gray,
            sensor_hint=sensor_hint,
        )
        previous_transformed_gray = decoded_frame.gray if decoded_frame.gray is not None else previous_transformed_gray
        frame = replace(
            record.to_frame(index=index, video_name=scenario.video_path.stem),
            translation_x=float(hint.get("translation_x", record.translation_x)),
            translation_y=float(hint.get("translation_y", record.translation_y)),
            translation_z=float(hint.get("translation_z", record.translation_z)),
        )
        started = time.perf_counter()
        translation, diagnostics = resolve_task2_translation(frame, decoded_frame, estimator)
        latencies.append((time.perf_counter() - started) * 1000.0)
        output = (translation.translation_x, translation.translation_y, translation.translation_z)
        ground_truth = (record.translation_x, record.translation_y, record.translation_z)
        error = _l2(output, ground_truth)
        all_errors.append(error)
        if record.health_status == "0":
            health0_errors.append(error)
        elif previous_health == "0":
            recovery_errors.append(error)
        if index in transition_indices:
            transition_errors.append(error)
        if previous_output is not None and previous_ground_truth is not None:
            output_delta = _l2(output, previous_output)
            truth_delta = _l2(ground_truth, previous_ground_truth)
            if output_delta > ((settings.task2_max_step_xy * 1.5) + settings.task2_max_step_z):
                catastrophic_jumps += 1
            if previous_health == "0" and record.health_status == "1":
                recovery_continuity_errors.append(abs(output_delta - truth_delta))
        previous_output = output
        previous_ground_truth = ground_truth
        previous_health = record.health_status
    return {
        "scenario_id": scenario.scenario_id,
        "variant": variant,
        "status": "ok",
        "tags": list(scenario.tags),
        "evaluated_frames": len(all_errors),
        "health0_frame_count": len(health0_errors),
        "transition_window_count": len(transition_errors),
        "recovery_count": len(recovery_errors),
        "health0_drift_metric": round(_safe_mean(health0_errors), 6),
        "transition_window_drift_metric": round(_safe_mean(transition_errors), 6),
        "recovery_error": round(_safe_mean(recovery_errors), 6),
        "recovery_continuity_error": round(_safe_mean(recovery_continuity_errors), 6),
        "catastrophic_jump_count": catastrophic_jumps,
        "runtime_p95_ms": round(percentile(latencies, 95.0), 4),
    }


def _aggregate_variant(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [item for item in results if item.get("status") == "ok"]
    thermal = [item for item in ok if "thermal" in item.get("tags", [])]
    rgb_control = [item for item in ok if "rgb" in item.get("tags", [])]
    scenario_map = {item["scenario_id"]: item for item in ok}
    return {
        "scenario_count": len(results),
        "ok_count": len(ok),
        "aggregate_health0_drift": round(_weighted_mean(ok, "health0_drift_metric", "health0_frame_count"), 6),
        "thermal_only_drift": round(_weighted_mean(thermal, "health0_drift_metric", "health0_frame_count"), 6),
        "rgb_control_drift": round(_weighted_mean(rgb_control, "health0_drift_metric", "health0_frame_count"), 6),
        "transition_drift": round(_weighted_mean(ok, "transition_window_drift_metric", "transition_window_count"), 6),
        "catastrophic_jumps": int(sum(int(item.get("catastrophic_jump_count", 0)) for item in ok)),
        "recovery_continuity": round(_weighted_mean(ok, "recovery_continuity_error", "recovery_count"), 6),
        "recovery_error": round(_weighted_mean(ok, "recovery_error", "recovery_count"), 6),
        "runtime_p95_ms": round(_weighted_mean(ok, "runtime_p95_ms", "evaluated_frames"), 4),
        "target_scenario_health0_drift": round(
            float(scenario_map.get(TARGET_SCENARIO_ID, {}).get("health0_drift_metric", 0.0)),
            6,
        ),
    }


def _build_comparisons(variants: dict[str, Any]) -> dict[str, Any]:
    baseline = variants.get("baseline", {}).get("aggregate", {})
    baseline_scenarios = {
        item["scenario_id"]: item for item in variants.get("baseline", {}).get("scenarios", []) if item.get("status") == "ok"
    }
    comparisons: dict[str, Any] = {}
    baseline_target = float(baseline_scenarios.get(TARGET_SCENARIO_ID, {}).get("health0_drift_metric", 0.0))
    baseline_aggregate = float(baseline.get("aggregate_health0_drift", 0.0))
    baseline_thermal = float(baseline.get("thermal_only_drift", 0.0))
    baseline_rgb = float(baseline.get("rgb_control_drift", 0.0))
    baseline_transition = float(baseline.get("transition_drift", 0.0))
    baseline_recovery_continuity = float(baseline.get("recovery_continuity", 0.0))
    baseline_runtime_p95 = float(baseline.get("runtime_p95_ms", 0.0))
    baseline_jumps = int(baseline.get("catastrophic_jumps", 0))
    for name, payload in variants.items():
        if name == "baseline":
            continue
        aggregate = payload.get("aggregate", {})
        scenario_map = {
            item["scenario_id"]: item for item in payload.get("scenarios", []) if item.get("status") == "ok"
        }
        target_value = float(scenario_map.get(TARGET_SCENARIO_ID, {}).get("health0_drift_metric", 0.0))
        target_improvement_ratio = _improvement_ratio(target_value, baseline_target)
        aggregate_delta_ratio = _regression_ratio(float(aggregate.get("aggregate_health0_drift", 0.0)), baseline_aggregate)
        thermal_delta_ratio = _regression_ratio(float(aggregate.get("thermal_only_drift", 0.0)), baseline_thermal)
        rgb_delta_ratio = _regression_ratio(float(aggregate.get("rgb_control_drift", 0.0)), baseline_rgb)
        transition_delta_ratio = _regression_ratio(float(aggregate.get("transition_drift", 0.0)), baseline_transition)
        recovery_error_delta = float(aggregate.get("recovery_error", 0.0)) - float(baseline.get("recovery_error", 0.0))
        recovery_continuity_delta_ratio = _regression_ratio(
            float(aggregate.get("recovery_continuity", 0.0)),
            baseline_recovery_continuity,
        )
        runtime_p95_ratio = _ratio(float(aggregate.get("runtime_p95_ms", 0.0)), baseline_runtime_p95)
        catastrophic_jump_delta = int(aggregate.get("catastrophic_jumps", 0)) - baseline_jumps
        improved_scenarios = [
            scenario_id
            for scenario_id, baseline_item in baseline_scenarios.items()
            if scenario_id in scenario_map
            and float(scenario_map[scenario_id].get("health0_drift_metric", 0.0))
            < float(baseline_item.get("health0_drift_metric", 0.0))
        ]
        interesting = (
            target_improvement_ratio >= THERMAL_TARGET_MIN_IMPROVEMENT_RATIO
            and thermal_delta_ratio <= NON_REGRESSION_MAX_RATIO
            and aggregate_delta_ratio <= NON_REGRESSION_MAX_RATIO
            and rgb_delta_ratio <= NON_REGRESSION_MAX_RATIO
            and transition_delta_ratio <= NON_REGRESSION_MAX_RATIO
            and recovery_error_delta <= RECOVERY_ERROR_MAX_DELTA
            and recovery_continuity_delta_ratio <= RECOVERY_CONTINUITY_MAX_RATIO
            and catastrophic_jump_delta <= 0
            and runtime_p95_ratio <= RUNTIME_P95_MAX_RATIO
        )
        merge_ready = (
            interesting
            and float(aggregate.get("aggregate_health0_drift", 0.0)) <= baseline_aggregate
            and float(aggregate.get("rgb_control_drift", 0.0)) <= baseline_rgb
            and float(aggregate.get("transition_drift", 0.0)) <= baseline_transition
        )
        partial_ready = interesting and not merge_ready
        comparisons[name] = {
            "improved_scenarios": improved_scenarios,
            "target_scenario_improvement_ratio": round(target_improvement_ratio, 6),
            "aggregate_health0_delta_ratio": round(aggregate_delta_ratio, 6),
            "thermal_only_delta_ratio": round(thermal_delta_ratio, 6),
            "rgb_control_delta_ratio": round(rgb_delta_ratio, 6),
            "transition_delta_ratio": round(transition_delta_ratio, 6),
            "recovery_error_delta": round(recovery_error_delta, 6),
            "recovery_continuity_delta_ratio": round(recovery_continuity_delta_ratio, 6),
            "runtime_p95_ratio": round(runtime_p95_ratio, 6),
            "catastrophic_jump_delta": catastrophic_jump_delta,
            "interesting": interesting,
            "merge_ready": merge_ready,
            "partial_ready": partial_ready,
        }
    return comparisons


def _final_recommendation(comparisons: dict[str, Any]) -> str:
    if any(bool(item.get("merge_ready")) for item in comparisons.values()):
        return "MERGE CANDIDATE"
    if any(bool(item.get("partial_ready")) for item in comparisons.values()):
        return "PARTIAL MERGE CANDIDATE"
    return "NO MERGE CANDIDATE"


def _render_settings(settings: MvpRuntimeSettings) -> dict[str, Any]:
    return {
        "task2_confidence_floor_thermal": settings.task2_confidence_floor_thermal,
        "task2_sensor_hint_max_weight_thermal": settings.task2_sensor_hint_max_weight_thermal,
        "task2_phase_primary_response_min_thermal": settings.task2_phase_primary_response_min_thermal,
        "task2_z_update_scale_thermal": settings.task2_z_update_scale_thermal,
    }


def _render_results_markdown(payload: dict[str, Any]) -> str:
    variants = payload.get("variants", {})
    comparisons = payload.get("comparisons", {})
    baseline = variants.get("baseline", {}).get("aggregate", {})
    lines = [
        "# Task 2 Thermal Microlever Results",
        "",
        f"- Final recommendation: `{payload.get('final_recommendation')}`",
        f"- Baseline aggregate health0 drift: `{baseline.get('aggregate_health0_drift')}`",
        f"- Baseline thermal-only drift: `{baseline.get('thermal_only_drift')}`",
        f"- Baseline RGB/control drift: `{baseline.get('rgb_control_drift')}`",
        f"- Baseline transition drift: `{baseline.get('transition_drift')}`",
        f"- Baseline catastrophic jumps: `{baseline.get('catastrophic_jumps')}`",
        f"- Baseline recovery continuity: `{baseline.get('recovery_continuity')}`",
        f"- Baseline recovery error: `{baseline.get('recovery_error')}`",
        f"- Baseline runtime p95 ms: `{baseline.get('runtime_p95_ms')}`",
        "",
    ]
    for name, comparison in comparisons.items():
        aggregate = variants.get(name, {}).get("aggregate", {})
        lines.append(f"## {name}")
        lines.append(f"- Exact micro-lever tested: `{variants.get(name, {}).get('runtime_settings')}`")
        lines.append(f"- Improved scenarios: `{comparison.get('improved_scenarios')}`")
        lines.append(
            f"- Target scenario `{TARGET_SCENARIO_ID}` improvement ratio: `{comparison.get('target_scenario_improvement_ratio')}`"
        )
        lines.append(
            f"- Aggregate health0 drift: `{aggregate.get('aggregate_health0_drift')}` "
            f"(delta ratio `{comparison.get('aggregate_health0_delta_ratio')}`)"
        )
        lines.append(
            f"- Thermal-only drift: `{aggregate.get('thermal_only_drift')}` "
            f"(delta ratio `{comparison.get('thermal_only_delta_ratio')}`)"
        )
        lines.append(
            f"- RGB/control drift: `{aggregate.get('rgb_control_drift')}` "
            f"(delta ratio `{comparison.get('rgb_control_delta_ratio')}`)"
        )
        lines.append(
            f"- Transition drift: `{aggregate.get('transition_drift')}` "
            f"(delta ratio `{comparison.get('transition_delta_ratio')}`)"
        )
        lines.append(f"- Catastrophic jumps: `{aggregate.get('catastrophic_jumps')}` (delta `{comparison.get('catastrophic_jump_delta')}`)")
        lines.append(
            f"- Recovery continuity / error: `{aggregate.get('recovery_continuity')}` / `{aggregate.get('recovery_error')}`"
        )
        lines.append(f"- Runtime p95 ms: `{aggregate.get('runtime_p95_ms')}` (ratio `{comparison.get('runtime_p95_ratio')}`)")
        lines.append(f"- Interesting: `{comparison.get('interesting')}`")
        lines.append("")
    lines.extend(["## Per-Scenario Health0 Drift", ""])
    baseline_scenarios = {
        item["scenario_id"]: item for item in variants.get("baseline", {}).get("scenarios", []) if item.get("status") == "ok"
    }
    for scenario_id, baseline_item in baseline_scenarios.items():
        lines.append(f"### {scenario_id}")
        lines.append(
            f"- baseline: health0 `{baseline_item.get('health0_drift_metric')}`, transition `{baseline_item.get('transition_window_drift_metric')}`, "
            f"runtime_p95 `{baseline_item.get('runtime_p95_ms')}`, jumps `{baseline_item.get('catastrophic_jump_count')}`"
        )
        for name, variant in variants.items():
            if name == "baseline":
                continue
            scenario_item = next(
                (item for item in variant.get("scenarios", []) if item.get("scenario_id") == scenario_id and item.get("status") == "ok"),
                None,
            )
            if scenario_item is None:
                continue
            lines.append(
                f"- {name}: health0 `{scenario_item.get('health0_drift_metric')}`, "
                f"transition `{scenario_item.get('transition_window_drift_metric')}`, "
                f"runtime_p95 `{scenario_item.get('runtime_p95_ms')}`, jumps `{scenario_item.get('catastrophic_jump_count')}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_design_markdown(
    scenarios: list[Task2ThermalMicroleverScenario],
    variants: dict[str, Any],
) -> str:
    lines = [
        "# Task 2 Thermal Microlever Design",
        "",
        "## Approach",
        "- Start from the safe pre-drift-crush Task 2 baseline.",
        "- Reuse the previous Task 2 health0 slice harness shape, but test only thermal-only knob overrides.",
        "- Keep production defaults unchanged and avoid any new hold, quarantine, smoothing, or broad estimator-family logic.",
        "",
        "## Candidate Micro-Levers",
        "- `thermal_confidence_floor_045`: raise `task2_confidence_floor_thermal` from `0.40` to `0.45`.",
        "- `thermal_sensor_hint_weight_008`: reduce `task2_sensor_hint_max_weight_thermal` from `0.15` to `0.08`.",
        "",
        "## Scenarios",
    ]
    for item in scenarios:
        segment_text = ", ".join(f"{segment['length']}x{segment['health']}" for segment in item.health_segments)
        lines.append(
            f"- `{item.scenario_id}`: transform=`{item.transform_profile}`, "
            f"frame_limit=`{item.frame_limit}`, frame_stride=`{item.frame_stride}`, "
            f"health_segments=`{segment_text}`, tags=`{item.tags}`"
        )
    lines.extend(["", "## Variant Settings"])
    for name, payload in variants.items():
        lines.append(f"- `{name}` -> `{payload.get('runtime_settings')}`")
    return "\n".join(lines) + "\n"


def _render_recommendation_markdown(payload: dict[str, Any]) -> str:
    comparisons = payload.get("comparisons", {})
    best_name = _best_candidate_name(comparisons)
    lines = [
        "# Task 2 Thermal Microlever Recommendation",
        "",
        f"{payload.get('final_recommendation')}",
        "",
    ]
    if best_name is None:
        lines.append("- No candidate satisfied the strict thermal-target and non-regression gates.")
    else:
        comparison = comparisons.get(best_name, {})
        lines.append(f"- Best candidate: `{best_name}`")
        lines.append(f"- Target improvement ratio: `{comparison.get('target_scenario_improvement_ratio')}`")
        lines.append(f"- Aggregate health0 delta ratio: `{comparison.get('aggregate_health0_delta_ratio')}`")
        lines.append(f"- RGB/control delta ratio: `{comparison.get('rgb_control_delta_ratio')}`")
        lines.append(f"- Transition delta ratio: `{comparison.get('transition_delta_ratio')}`")
        lines.append(f"- Catastrophic jump delta: `{comparison.get('catastrophic_jump_delta')}`")
        lines.append(f"- Runtime p95 ratio: `{comparison.get('runtime_p95_ratio')}`")
    return "\n".join(lines) + "\n"


def _best_candidate_name(comparisons: dict[str, Any]) -> str | None:
    if not comparisons:
        return None
    ordered = sorted(
        comparisons.items(),
        key=lambda item: (
            int(bool(item[1].get("merge_ready"))),
            int(bool(item[1].get("partial_ready"))),
            float(item[1].get("target_scenario_improvement_ratio", 0.0)),
            -float(item[1].get("aggregate_health0_delta_ratio", 0.0)),
        ),
        reverse=True,
    )
    return str(ordered[0][0]) if ordered else None


def _transition_window_indices(schedule: list[str]) -> set[int]:
    indices: set[int] = set()
    start: int | None = None
    for index, health in enumerate(schedule + ["1"]):
        if health == "0" and start is None:
            start = index
            continue
        if health == "0" or start is None:
            continue
        end = index - 1
        window = list(range(start, min(start + 20, end + 1))) + list(range(max(end - 19, start), end + 1))
        indices.update(window)
        start = None
    return indices


def _l2(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 + (first[2] - second[2]) ** 2) ** 0.5


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _weighted_mean(items: list[dict[str, Any]], value_key: str, count_key: str) -> float:
    total = sum(int(item.get(count_key, 0)) for item in items)
    if total <= 0:
        return 0.0
    return sum(float(item.get(value_key, 0.0)) * int(item.get(count_key, 0)) for item in items) / total


def _improvement_ratio(current: float, baseline: float) -> float:
    if baseline <= 1e-9:
        return 0.0
    return (baseline - current) / baseline


def _regression_ratio(current: float, baseline: float) -> float:
    if baseline <= 1e-9:
        return 0.0 if current <= 1e-9 else 1.0
    return (current - baseline) / baseline


def _ratio(current: float, baseline: float) -> float:
    if baseline <= 1e-9:
        return 0.0
    return current / baseline
