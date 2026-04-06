from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import FrameEnvelope
from src.core.utils import percentile
from src.evaluation.task2_long_sequence import iter_video_frames
from src.task3.pipeline import Task3Pipeline
from src.tools.vram_monitor import query_vram


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_manifest_path() -> Path:
    return _project_root() / "data" / "task3_eval_manifest.json"


@dataclass(slots=True)
class Task3EvalScenario:
    scenario_id: str
    video_path: Path
    reference_dir: Path | None
    tags: list[str]
    frame_stride: int
    frame_limit: int | None
    target_expected: bool
    force_track_loss_frame: int | None = None
    reference_mode: str = "directory"


def load_task3_eval_manifest(manifest_path: str | Path | None = None) -> list[Task3EvalScenario]:
    payload = json.loads(Path(manifest_path or _default_manifest_path()).read_text(encoding="utf-8"))
    scenarios: list[Task3EvalScenario] = []
    for item in payload.get("scenarios", []):
        video_path = _resolve_repo_path(item["video_path"])
        reference_dir = _resolve_repo_path(item["reference_dir"]) if item.get("reference_dir") else None
        scenarios.append(
            Task3EvalScenario(
                scenario_id=str(item["scenario_id"]),
                video_path=video_path,
                reference_dir=reference_dir,
                tags=[str(tag) for tag in item.get("tags", [])],
                frame_stride=int(item.get("frame_stride", 60)),
                frame_limit=int(item["frame_limit"]) if item.get("frame_limit") is not None else None,
                target_expected=bool(item.get("target_expected", True)),
                force_track_loss_frame=(
                    int(item["force_track_loss_frame"]) if item.get("force_track_loss_frame") is not None else None
                ),
                reference_mode=str(item.get("reference_mode", "directory")),
            )
        )
    return scenarios


def apply_task3_settings_overrides(
    settings: MvpRuntimeSettings,
    overrides: dict[str, Any] | None,
) -> MvpRuntimeSettings:
    if not overrides:
        return settings
    normalized = dict(overrides)
    for path_field in (
        "task3_reference_dir",
        "task3_eval_reference_dir",
        "task3_experimental_model_path",
        "task3_experimental_verifier_weights_path",
    ):
        if normalized.get(path_field) is not None:
            normalized[path_field] = Path(str(normalized[path_field])).expanduser()
    return replace(settings, **normalized)


def evaluate_task3_mode(
    *,
    mode: str,
    manifest_path: str | Path | None = None,
    runtime_settings: MvpRuntimeSettings | None = None,
    output_dir: str | Path = "reports",
) -> dict[str, Any]:
    scenarios = load_task3_eval_manifest(manifest_path)
    settings = _resolve_mode_settings(mode, runtime_settings or MvpRuntimeSettings())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scenario_results: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_results.append(_evaluate_scenario(scenario, settings=settings, output_dir=output_path))
    aggregate = _aggregate_mode_results(mode, scenario_results)
    payload = {
        "mode": mode,
        "runtime_settings": _render_mode_settings(settings),
        "scenarios": scenario_results,
        "aggregate": aggregate,
    }
    (output_path / f"task3_experimental_{mode}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def compare_task3_baseline_vs_experimental(
    *,
    manifest_path: str | Path | None = None,
    runtime_settings: MvpRuntimeSettings | None = None,
    output_dir: str | Path = "reports",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    baseline = evaluate_task3_mode(
        mode="baseline",
        manifest_path=manifest_path,
        runtime_settings=runtime_settings,
        output_dir=output_path,
    )
    experimental = evaluate_task3_mode(
        mode="experimental",
        manifest_path=manifest_path,
        runtime_settings=runtime_settings,
        output_dir=output_path,
    )
    comparison = _compare_modes(baseline["aggregate"], experimental["aggregate"])
    payload = {
        "baseline": baseline,
        "experimental": experimental,
        "comparison": comparison,
    }
    (output_path / "task3_baseline_vs_experimental.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_path / "task3_baseline_vs_experimental.md").write_text(
        _render_comparison_markdown(payload),
        encoding="utf-8",
    )
    (output_path / "task3_experimental_design.md").write_text(
        _render_design_markdown(payload),
        encoding="utf-8",
    )
    (output_path / "task3_integration_risks.md").write_text(
        _render_risks_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _evaluate_scenario(
    scenario: Task3EvalScenario,
    *,
    settings: MvpRuntimeSettings,
    output_dir: Path,
) -> dict[str, Any]:
    reference_dir = _prepare_reference_dir(scenario, output_dir)
    scenario_settings = replace(
        settings,
        task3_reference_dir=reference_dir,
        task3_eval_reference_dir=reference_dir,
    )
    pipeline = Task3Pipeline(runtime_settings=scenario_settings)
    latencies_ms: list[float] = []
    peak_vram_mb = _initial_peak_vram(settings)
    accepted_match_count = 0
    detection_success_frames = 0
    no_match_frames = 0
    no_match_correct_frames = 0
    false_positive_proxy_count = 0
    rejected_frames = 0
    jitter_values: list[float] = []
    previous_centers: dict[str, tuple[float, float]] = {}
    re_detection_success = False
    forced_track_loss_applied = False
    track_loss_frame = scenario.force_track_loss_frame
    track_loss_deadline = (
        (track_loss_frame + max(scenario_settings.task3_experimental_track_lost_patience, 1) + 2)
        if track_loss_frame is not None
        else None
    )
    integration_statuses: list[str] = []
    tracker_modes: list[str] = []
    verifier_modes: list[str] = []
    results_seen = 0

    if not scenario.video_path.exists():
        return {
            "scenario_id": scenario.scenario_id,
            "status": "missing_video",
            "video_path": str(scenario.video_path),
            "reference_dir": str(reference_dir),
            "target_expected": scenario.target_expected,
            "tags": list(scenario.tags),
        }

    frame_iterator = iter_video_frames(
        scenario.video_path,
        frame_stride=scenario.frame_stride,
        limit=scenario.frame_limit,
        video_name=scenario.video_path.stem,
    )
    for local_index, decoded in enumerate(frame_iterator):
        if track_loss_frame is not None and local_index == track_loss_frame:
            pipeline.force_track_loss()
            forced_track_loss_applied = True
        frame = FrameEnvelope(
            frame_url=f"http://task3-eval/frames/{local_index + 1}/",
            image_url=f"/task3/{scenario.scenario_id}/{local_index + 1}.jpg",
            video_name=scenario.video_path.stem,
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
            metadata={
                "frame_index": decoded.frame_index,
                "image_width": decoded.width,
                "image_height": decoded.height,
            },
        )
        started = time.perf_counter()
        matches, diagnostics = pipeline.process(frame, b"", decoded_frame=decoded)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)
        peak_vram_mb = _update_peak_vram(settings, peak_vram_mb)
        results_seen += 1

        integration_statuses.append(str(diagnostics.get("integration_status", "baseline")))
        tracker_modes.append(str(diagnostics.get("tracker_active_mode", "off")))
        verifier_modes.append(str(diagnostics.get("verifier_active_mode", "baseline")))
        rejected_frames += int(bool(diagnostics.get("redetect_triggered"))) if not matches else 0

        if not matches:
            no_match_frames += 1
            if not scenario.target_expected:
                no_match_correct_frames += 1
            continue

        accepted_match_count += len(matches)
        if scenario.target_expected:
            detection_success_frames += 1
        else:
            false_positive_proxy_count += len(matches)

        if forced_track_loss_applied and track_loss_frame is not None and track_loss_deadline is not None:
            if local_index > track_loss_frame and local_index <= track_loss_deadline:
                re_detection_success = True

        for match in matches:
            if scenario.target_expected:
                fused_score = float(match.metadata.get("fused_score", match.metadata.get("match_score", 0.0)))
                if fused_score < max(scenario_settings.task3_min_score + 0.04, 0.75):
                    false_positive_proxy_count += 1
            center = (
                (float(match.top_left_x) + float(match.bottom_right_x)) / 2.0,
                (float(match.top_left_y) + float(match.bottom_right_y)) / 2.0,
            )
            previous = previous_centers.get(match.object_id)
            if previous is not None:
                jitter_values.append(math.dist(previous, center))
            previous_centers[match.object_id] = center

    return {
        "scenario_id": scenario.scenario_id,
        "status": "ok",
        "video_path": str(scenario.video_path),
        "reference_dir": str(reference_dir),
        "target_expected": scenario.target_expected,
        "tags": list(scenario.tags),
        "frame_count": results_seen,
        "accepted_match_count": accepted_match_count,
        "detection_success_frames": detection_success_frames,
        "no_match_frames": no_match_frames,
        "no_match_correct_frames": no_match_correct_frames,
        "false_positive_proxy_count": false_positive_proxy_count,
        "rejected_frames": rejected_frames,
        "re_detection_success": re_detection_success,
        "forced_track_loss_frame": track_loss_frame,
        "integration_status": _dominant_value(integration_statuses, "baseline"),
        "active_tracker_mode": _dominant_value(tracker_modes, "off"),
        "active_verifier_mode": _dominant_value(verifier_modes, "baseline"),
        "runtime_p50_ms": round(percentile(latencies_ms, 50.0), 4),
        "runtime_p95_ms": round(percentile(latencies_ms, 95.0), 4),
        "mean_jitter_px": round(_safe_mean(jitter_values), 4),
        "peak_vram_mb": peak_vram_mb,
    }


def _resolve_mode_settings(mode: str, settings: MvpRuntimeSettings) -> MvpRuntimeSettings:
    if mode == "experimental":
        verifier_mode = settings.task3_experimental_verifier if settings.task3_experimental_verifier != "off" else "orb_homography"
        tracking_mode = settings.task3_experimental_tracking if settings.task3_experimental_tracking != "off" else "light"
        experimental_mode = settings.task3_experimental_mode if settings.task3_experimental_mode != "baseline" else "yoloe_prompted"
        return replace(
            settings,
            task3_experimental_enabled=True,
            task3_experimental_mode=experimental_mode,
            task3_experimental_tracking=tracking_mode,
            task3_experimental_verifier=verifier_mode,
        )
    return replace(
        settings,
        task3_experimental_enabled=False,
        task3_experimental_mode="baseline",
        task3_experimental_tracking="off",
        task3_experimental_verifier="off",
    )


def _aggregate_mode_results(mode: str, scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [item for item in scenario_results if item.get("status") == "ok"]
    target_expected = [item for item in ok_results if item.get("target_expected")]
    target_absent = [item for item in ok_results if not item.get("target_expected")]
    integration_status = _dominant_value((item.get("integration_status", "baseline") for item in ok_results), "baseline")
    tracker_mode = _dominant_value((item.get("active_tracker_mode", "off") for item in ok_results), "off")
    verifier_mode = _dominant_value((item.get("active_verifier_mode", "baseline") for item in ok_results), "baseline")
    latencies = [float(item.get("runtime_p50_ms", 0.0)) for item in ok_results]
    p95_latencies = [float(item.get("runtime_p95_ms", 0.0)) for item in ok_results]
    return {
        "mode": mode,
        "scenario_count": len(scenario_results),
        "ok_count": len(ok_results),
        "integration_status": integration_status,
        "active_tracker_mode": tracker_mode,
        "active_verifier_mode": verifier_mode,
        "frame_count": int(sum(int(item.get("frame_count", 0)) for item in ok_results)),
        "accepted_match_count": int(sum(int(item.get("accepted_match_count", 0)) for item in ok_results)),
        "target_present_detection_success_frames": int(
            sum(int(item.get("detection_success_frames", 0)) for item in target_expected)
        ),
        "target_absent_no_match_correct_frames": int(
            sum(int(item.get("no_match_correct_frames", 0)) for item in target_absent)
        ),
        "target_absent_frame_count": int(sum(int(item.get("frame_count", 0)) for item in target_absent)),
        "false_positive_proxy_count": int(sum(int(item.get("false_positive_proxy_count", 0)) for item in ok_results)),
        "re_detection_success_count": int(sum(1 for item in ok_results if item.get("re_detection_success"))),
        "runtime_p50_ms": round(_safe_mean(latencies), 4),
        "runtime_p95_ms": round(_safe_mean(p95_latencies), 4),
        "mean_jitter_px": round(_safe_mean(item.get("mean_jitter_px", 0.0) for item in ok_results), 4),
        "peak_vram_mb": _max_known(item.get("peak_vram_mb") for item in ok_results),
        "absent_target_no_match_rate": round(
            sum(int(item.get("no_match_correct_frames", 0)) for item in target_absent)
            / max(sum(int(item.get("frame_count", 0)) for item in target_absent), 1),
            6,
        ),
    }


def _compare_modes(baseline: dict[str, Any], experimental: dict[str, Any]) -> dict[str, Any]:
    detection_gain = int(experimental.get("target_present_detection_success_frames", 0)) - int(
        baseline.get("target_present_detection_success_frames", 0)
    )
    false_positive_delta = int(experimental.get("false_positive_proxy_count", 0)) - int(
        baseline.get("false_positive_proxy_count", 0)
    )
    absent_gain = round(
        float(experimental.get("absent_target_no_match_rate", 0.0))
        - float(baseline.get("absent_target_no_match_rate", 0.0)),
        6,
    )
    redetect_gain = int(experimental.get("re_detection_success_count", 0)) - int(
        baseline.get("re_detection_success_count", 0)
    )
    runtime_ratio = 0.0
    baseline_runtime = float(baseline.get("runtime_p50_ms", 0.0) or 0.0)
    experimental_runtime = float(experimental.get("runtime_p50_ms", 0.0) or 0.0)
    if baseline_runtime > 0.0:
        runtime_ratio = round(experimental_runtime / baseline_runtime, 4)

    improvements = {
        "detection_rate": detection_gain >= 1,
        "false_positive_reduction": false_positive_delta <= -1,
        "absent_target_behavior": absent_gain >= 0.05,
        "re_detection": redetect_gain >= 1,
    }
    runtime_ok = baseline_runtime <= 1e-6 or experimental_runtime <= (baseline_runtime * 1.5)
    meaningful_gain = any(improvements.values()) and runtime_ok
    recommendation = "EXPERIMENTAL ONLY"
    if meaningful_gain:
        recommendation = (
            "MERGE CANDIDATE"
            if str(experimental.get("integration_status")) == "true_yoloe"
            else "PARTIAL MERGE CANDIDATE"
        )
    return {
        "integration_status": experimental.get("integration_status"),
        "detection_gain_frames": detection_gain,
        "false_positive_delta": false_positive_delta,
        "absent_target_no_match_rate_delta": absent_gain,
        "re_detection_gain": redetect_gain,
        "runtime_ratio_vs_baseline": runtime_ratio,
        "runtime_ok": runtime_ok,
        "improvements": improvements,
        "meaningful_gain": meaningful_gain,
        "recommendation": recommendation,
    }


def _render_mode_settings(settings: MvpRuntimeSettings) -> dict[str, Any]:
    return {
        "task3_experimental_enabled": settings.task3_experimental_enabled,
        "task3_experimental_mode": settings.task3_experimental_mode,
        "task3_experimental_tracking": settings.task3_experimental_tracking,
        "task3_experimental_verifier": settings.task3_experimental_verifier,
        "task3_experimental_tiled_inference": settings.task3_experimental_tiled_inference,
        "task3_experimental_multiscale": settings.task3_experimental_multiscale,
        "task3_experimental_model_path": str(settings.task3_experimental_model_path) if settings.task3_experimental_model_path else None,
    }


def _render_comparison_markdown(payload: dict[str, Any]) -> str:
    baseline = payload["baseline"]["aggregate"]
    experimental = payload["experimental"]["aggregate"]
    comparison = payload["comparison"]
    return (
        "# Task 3 Baseline vs Experimental\n\n"
        f"- Recommendation: `{comparison.get('recommendation')}`\n"
        f"- Integration status: `{comparison.get('integration_status')}`\n\n"
        "| Metric | Baseline | Experimental | Delta |\n"
        "| --- | --- | --- | --- |\n"
        f"| Accepted Match Count | {baseline.get('accepted_match_count')} | {experimental.get('accepted_match_count')} | {int(experimental.get('accepted_match_count', 0)) - int(baseline.get('accepted_match_count', 0))} |\n"
        f"| Present Detection Frames | {baseline.get('target_present_detection_success_frames')} | {experimental.get('target_present_detection_success_frames')} | {comparison.get('detection_gain_frames')} |\n"
        f"| False Positive Proxy | {baseline.get('false_positive_proxy_count')} | {experimental.get('false_positive_proxy_count')} | {comparison.get('false_positive_delta')} |\n"
        f"| Absent-target No-match Rate | {baseline.get('absent_target_no_match_rate')} | {experimental.get('absent_target_no_match_rate')} | {comparison.get('absent_target_no_match_rate_delta')} |\n"
        f"| Re-detection Success Count | {baseline.get('re_detection_success_count')} | {experimental.get('re_detection_success_count')} | {comparison.get('re_detection_gain')} |\n"
        f"| Runtime P50 ms | {baseline.get('runtime_p50_ms')} | {experimental.get('runtime_p50_ms')} | {round(float(experimental.get('runtime_p50_ms', 0.0)) - float(baseline.get('runtime_p50_ms', 0.0)), 4)} |\n"
        f"| Runtime P95 ms | {baseline.get('runtime_p95_ms')} | {experimental.get('runtime_p95_ms')} | {round(float(experimental.get('runtime_p95_ms', 0.0)) - float(baseline.get('runtime_p95_ms', 0.0)), 4)} |\n"
        f"| Peak VRAM MB | {baseline.get('peak_vram_mb')} | {experimental.get('peak_vram_mb')} | - |\n"
    )


def _render_design_markdown(payload: dict[str, Any]) -> str:
    experimental = payload["experimental"]["aggregate"]
    comparison = payload["comparison"]
    true_yoloe = str(experimental.get("integration_status")) == "true_yoloe"
    achieved = "true_yoloe" if true_yoloe else "prompt_approx"
    return (
        "# Task 3 Experimental Design\n\n"
        "## Freeze Boundary\n"
        "- Production Task 1 chain korunuyor.\n"
        "- `onnxruntime:yolo11n` production-disabled kaliyor.\n"
        "- Default sequential wire profile `official_current` olarak kaliyor.\n"
        "- Task 3 experimental path config-gated ve default-off.\n\n"
        "## Implemented Experimental Pipeline\n"
        f"- Detector path: `{achieved}`\n"
        f"- Requested mode: `{payload['experimental']['runtime_settings'].get('task3_experimental_mode')}`\n"
        f"- Active tracker: `{experimental.get('active_tracker_mode')}`\n"
        f"- Active verifier: `{experimental.get('active_verifier_mode')}`\n"
        f"- Tiled inference enabled: `{payload['experimental']['runtime_settings'].get('task3_experimental_tiled_inference')}`\n"
        f"- Multiscale enabled: `{payload['experimental']['runtime_settings'].get('task3_experimental_multiscale')}`\n\n"
        "## Outcome\n"
        f"- True YOLOE achieved: `{true_yoloe}`\n"
        f"- Recommendation: `{comparison.get('recommendation')}`\n"
        f"- Meaningful gain: `{comparison.get('meaningful_gain')}`\n"
    )


def _render_risks_markdown(payload: dict[str, Any]) -> str:
    experimental = payload["experimental"]["aggregate"]
    comparison = payload["comparison"]
    risks = [
        f"- True YOLOE integration status: `{experimental.get('integration_status')}`.",
        "- `sahi` ve `lightglue` bu branch'te opsiyonel; kurulu degilse otomatik degrade oluyor.",
        "- Experimental pipeline production default degil; merge edilse bile flag arkasinda kalmali.",
    ]
    if str(experimental.get("integration_status")) != "true_yoloe":
        risks.append("- YOLOE agirliklari sahaya alinmadan prompt-conditioned learned detector tam kapanmis sayilamaz.")
    if str(comparison.get("recommendation")) == "EXPERIMENTAL ONLY":
        risks.append("- Mevcut olcumler merge icin yeterli kazanimi gostermiyor.")
    return "# Task 3 Integration Risks\n\n" + "\n".join(risks) + "\n"


def _prepare_reference_dir(scenario: Task3EvalScenario, output_dir: Path) -> Path:
    if scenario.reference_mode != "synthetic_absent":
        if scenario.reference_dir is None:
            raise FileNotFoundError(f"Reference dir missing for scenario: {scenario.scenario_id}")
        return scenario.reference_dir
    cache_dir = output_dir / "_task3_eval_refs" / scenario.scenario_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    reference_path = cache_dir / "synthetic_absent_ref.pgm"
    if not reference_path.exists():
        reference_path.write_bytes(_synthetic_absent_reference_bytes())
    return cache_dir


def _synthetic_absent_reference_bytes() -> bytes:
    width = 64
    height = 64
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            diagonal = x == y or (x + y) == (width - 1)
            checker = ((x // 4) + (y // 4)) % 2 == 0
            if diagonal:
                pixels[(y * width) + x] = 255
            elif checker and 10 <= x < 54 and 10 <= y < 54:
                pixels[(y * width) + x] = 180
    return f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(pixels)


def _resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return _project_root() / path


def _initial_peak_vram(settings: MvpRuntimeSettings) -> float | None:
    snapshot = query_vram(settings.profiling_gpu_query_cmd)
    return float(snapshot.used_mb) if snapshot.used_mb is not None else None


def _update_peak_vram(settings: MvpRuntimeSettings, current_peak: float | None) -> float | None:
    snapshot = query_vram(settings.profiling_gpu_query_cmd)
    if snapshot.used_mb is None:
        return current_peak
    if current_peak is None:
        return float(snapshot.used_mb)
    return max(current_peak, float(snapshot.used_mb))


def _dominant_value(values: Iterable[Any], default: str) -> str:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return default
    return max(sorted(counts), key=lambda item: counts[item])


def _safe_mean(values: Iterable[float]) -> float:
    collected = [float(item) for item in values]
    return mean(collected) if collected else 0.0


def _max_known(values: Iterable[Any]) -> float | None:
    collected = [float(item) for item in values if item is not None]
    return round(max(collected), 4) if collected else None
