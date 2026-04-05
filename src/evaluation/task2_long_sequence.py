from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Iterator

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.core.utils import infer_modality
from src.core.vision import is_cv2_available
from src.task2.estimator import Task2Estimator
from src.task2.health_logic import resolve_task2_translation

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2
else:  # pragma: no cover - cv2 yoksa
    cv2 = None


@dataclass(slots=True)
class Task2CsvRecord:
    frame_id: str
    translation_x: float
    translation_y: float
    translation_z: float
    health_status: str

    def to_frame(self, *, index: int, video_name: str) -> FrameEnvelope:
        return FrameEnvelope(
            frame_url=f"http://replay/frames/{index + 1}/",
            image_url=f"/frames/{self.frame_id}.jpg",
            video_name=video_name,
            translation_x=self.translation_x,
            translation_y=self.translation_y,
            translation_z=self.translation_z,
            health_status=self.health_status,
            metadata={"frame_index": index},
        )


def discover_task2_sources(root_dir: str | Path | None = None) -> list[tuple[Path, Path | None]]:
    root = Path(root_dir) if root_dir is not None else Path("data") / "HYZ_2025_Ornek_Veriler-001"
    sources: list[tuple[Path, Path | None]] = []
    for csv_path in sorted(root.glob("*-translation.csv")):
        prefix = csv_path.stem.removesuffix("-translation")
        candidates = [csv_path.with_name(f"{prefix}.MP4"), csv_path.with_name(f"{prefix}.mp4")]
        video_path = next((candidate for candidate in candidates if candidate.exists()), None)
        sources.append((csv_path, video_path))
    return sources


def build_health_schedule(total_frames: int, reference_window: int = 450) -> list[str]:
    if total_frames <= 0:
        return []
    stable_window = min(reference_window, max(total_frames // 5, 1))
    recovery_window = min(reference_window, max(total_frames // 5, 1))
    schedule: list[str] = []
    for index in range(total_frames):
        is_reference = index < stable_window or index >= max(total_frames - recovery_window, stable_window)
        schedule.append("1" if is_reference else "0")
    return schedule


def load_translation_csv(
    csv_path: str | Path,
    *,
    frame_stride: int = 1,
    limit: int | None = None,
    reference_window: int = 450,
) -> list[Task2CsvRecord]:
    rows: list[dict[str, str]] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index % max(frame_stride, 1) != 0:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    schedule = build_health_schedule(len(rows), reference_window=reference_window)
    records: list[Task2CsvRecord] = []
    for index, row in enumerate(rows):
        records.append(
            Task2CsvRecord(
                frame_id=str(row.get("frame_numbers", f"frame_{index:06d}")),
                translation_x=float(row["translation_x"]),
                translation_y=float(row["translation_y"]),
                translation_z=float(row["translation_z"]),
                health_status=schedule[index],
            )
        )
    return records


def iter_video_frames(
    video_path: str | Path,
    *,
    frame_stride: int = 1,
    limit: int | None = None,
    video_name: str = "",
) -> Iterator[DecodedFrame]:
    if not is_cv2_available():
        raise RuntimeError("OpenCV gerekli")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video acilamadi: {video_path}")

    frame_index = 0
    sampled = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(frame_stride, 1) != 0:
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
                modality=infer_modality(video_name or str(video_path), width=int(width), height=int(height)),
                frame_index=frame_index,
            )
            sampled += 1
            frame_index += 1
            if limit is not None and sampled >= limit:
                break
    finally:
        capture.release()


def evaluate_records(
    records: Iterable[Task2CsvRecord],
    decoded_frames: Iterable[DecodedFrame],
    *,
    runtime_settings: MvpRuntimeSettings | None = None,
    sequence_name: str = "unknown",
    video_name: str = "replay",
) -> dict[str, object]:
    settings = runtime_settings or MvpRuntimeSettings()
    estimator = Task2Estimator(runtime_settings=settings)
    records_list = list(records)
    decoded_iter = iter(decoded_frames)
    reference_errors: list[float] = []
    estimated_errors: list[float] = []
    continuity_errors: list[float] = []
    confidence_values: list[float] = []
    fallback_count = 0
    step_jump_count = 0
    recovery_errors: list[float] = []
    previous_output: tuple[float, float, float] | None = None
    previous_ground_truth: tuple[float, float, float] | None = None
    previous_health: str | None = None
    last_health0_error: float | None = None
    branch_counts = {"reference": 0, "estimated": 0}

    evaluated_frames = 0
    for index, record in enumerate(records_list):
        try:
            decoded = next(decoded_iter)
        except StopIteration:
            break
        frame = record.to_frame(index=index, video_name=video_name)
        translation, diagnostics = resolve_task2_translation(frame, decoded, estimator)
        output = (translation.translation_x, translation.translation_y, translation.translation_z)
        ground_truth = (record.translation_x, record.translation_y, record.translation_z)
        error = _l2_distance(output, ground_truth)
        confidence_values.append(float(diagnostics.get("confidence", 0.0)))
        branch = str(diagnostics.get("task2_branch", "estimated"))
        branch_counts["reference" if branch == "reference" else "estimated"] += 1
        if diagnostics.get("fallback_source") is not None or "fallback" in translation.source:
            fallback_count += 1

        if record.health_status == "1":
            reference_errors.append(error)
            if previous_health == "0":
                recovery_errors.append(error)
        else:
            estimated_errors.append(error)
            last_health0_error = error

        if previous_output is not None and previous_ground_truth is not None:
            output_delta = _l2_distance(output, previous_output)
            ground_truth_delta = _l2_distance(ground_truth, previous_ground_truth)
            continuity_errors.append(abs(output_delta - ground_truth_delta))
            if output_delta > ((settings.task2_max_step_xy * 1.5) + settings.task2_max_step_z):
                step_jump_count += 1

        previous_output = output
        previous_ground_truth = ground_truth
        previous_health = record.health_status
        evaluated_frames += 1
    summary = {
        "sequence_name": sequence_name,
        "status": "ok",
        "evaluated_frames": evaluated_frames,
        "reference_frames": branch_counts["reference"],
        "estimated_frames": branch_counts["estimated"],
        "reference_period_accuracy": round(_safe_mean(reference_errors), 6),
        "health0_drift_accumulation": round(_safe_mean(estimated_errors), 6),
        "health0_terminal_error": round(float(last_health0_error or 0.0), 6),
        "step_jump_count": step_jump_count,
        "continuity_error": round(_safe_mean(continuity_errors), 6),
        "recovery_error_after_health_returns_to_1": round(_safe_mean(recovery_errors), 6),
        "confidence_mean": round(_safe_mean(confidence_values), 6),
        "confidence_min": round(min(confidence_values), 6) if confidence_values else 0.0,
        "fallback_rate": round(fallback_count / max(evaluated_frames, 1), 6),
        "branch_counts": branch_counts,
    }
    return summary


def evaluate_task2_long_sequences(
    *,
    runtime_settings: MvpRuntimeSettings | None = None,
    root_dir: str | Path | None = None,
    output_dir: str | Path = "reports",
) -> dict[str, object]:
    settings = runtime_settings or MvpRuntimeSettings()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for csv_path, video_path in discover_task2_sources(root_dir):
        sequence_name = csv_path.stem.removesuffix("-translation")
        records = load_translation_csv(
            csv_path,
            frame_stride=settings.task2_eval_frame_stride,
            limit=settings.task2_eval_sequence_limit,
        )
        if video_path is None:
            results.append({"sequence_name": sequence_name, "status": "missing_video", "evaluated_frames": len(records)})
            continue
        try:
            decoded_frames = iter_video_frames(
                video_path,
                frame_stride=settings.task2_eval_frame_stride,
                limit=settings.task2_eval_sequence_limit,
                video_name=sequence_name,
            )
            summary = evaluate_records(
                records,
                decoded_frames,
                runtime_settings=settings,
                sequence_name=sequence_name,
                video_name=sequence_name,
            )
        except Exception as exc:
            summary = {"sequence_name": sequence_name, "status": "failed", "error": str(exc), "evaluated_frames": 0}
        results.append(summary)

    payload = {
        "results": results,
        "aggregate": {
            "sequence_count": len(results),
            "ok_count": sum(1 for item in results if item.get("status") == "ok"),
            "reference_period_accuracy": round(_safe_mean(item.get("reference_period_accuracy", 0.0) for item in results if item.get("status") == "ok"), 6),
            "health0_drift_accumulation": round(_safe_mean(item.get("health0_drift_accumulation", 0.0) for item in results if item.get("status") == "ok"), 6),
            "continuity_error": round(_safe_mean(item.get("continuity_error", 0.0) for item in results if item.get("status") == "ok"), 6),
            "recovery_error_after_health_returns_to_1": round(_safe_mean(item.get("recovery_error_after_health_returns_to_1", 0.0) for item in results if item.get("status") == "ok"), 6),
            "fallback_rate": round(_safe_mean(item.get("fallback_rate", 0.0) for item in results if item.get("status") == "ok"), 6),
        },
    }
    (output_path / "task2_long_sequence_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_path / "task2_long_sequence_table.md").write_text(render_task2_table(results), encoding="utf-8")
    return payload


def snapshot_existing_task2_report(output_dir: str | Path, *, suffix: str) -> None:
    output_path = Path(output_dir)
    summary = output_path / "task2_long_sequence_summary.json"
    table = output_path / "task2_long_sequence_table.md"
    if summary.exists():
        shutil.copyfile(summary, output_path / f"task2_long_sequence_{suffix}_summary.json")
    if table.exists():
        shutil.copyfile(table, output_path / f"task2_long_sequence_{suffix}_table.md")


def write_task2_after_snapshot(output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    summary = output_path / "task2_long_sequence_summary.json"
    table = output_path / "task2_long_sequence_table.md"
    if summary.exists():
        shutil.copyfile(summary, output_path / "task2_long_sequence_after_summary.json")
    if table.exists():
        shutil.copyfile(table, output_path / "task2_long_sequence_after_table.md")


def write_task2_comparison(output_dir: str | Path) -> dict[str, object] | None:
    output_path = Path(output_dir)
    before_path = output_path / "task2_long_sequence_before_summary.json"
    after_path = output_path / "task2_long_sequence_after_summary.json"
    if not before_path.exists() or not after_path.exists():
        return None
    before_payload = json.loads(before_path.read_text(encoding="utf-8"))
    after_payload = json.loads(after_path.read_text(encoding="utf-8"))
    before_aggregate = before_payload.get("aggregate", {})
    after_aggregate = after_payload.get("aggregate", {})
    comparison = {
        "health0_drift_accumulation_before": before_aggregate.get("health0_drift_accumulation", 0.0),
        "health0_drift_accumulation_after": after_aggregate.get("health0_drift_accumulation", 0.0),
        "health0_drift_delta": round(
            float(after_aggregate.get("health0_drift_accumulation", 0.0))
            - float(before_aggregate.get("health0_drift_accumulation", 0.0)),
            6,
        ),
        "fallback_rate_before": before_aggregate.get("fallback_rate", 0.0),
        "fallback_rate_after": after_aggregate.get("fallback_rate", 0.0),
        "recovery_error_before": before_aggregate.get("recovery_error_after_health_returns_to_1", 0.0),
        "recovery_error_after": after_aggregate.get("recovery_error_after_health_returns_to_1", 0.0),
    }
    markdown = render_task2_comparison(before_aggregate, after_aggregate)
    (output_path / "task2_long_sequence_comparison.md").write_text(markdown, encoding="utf-8")
    return comparison


def render_task2_table(results: list[dict[str, object]]) -> str:
    lines = [
        "| Sequence | Status | Frames | Ref Accuracy | Health0 Drift | Continuity | Recovery | Fallback Rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            "| {sequence} | {status} | {frames} | {ref} | {drift} | {continuity} | {recovery} | {fallback} |".format(
                sequence=item.get("sequence_name"),
                status=item.get("status"),
                frames=item.get("evaluated_frames", 0),
                ref=item.get("reference_period_accuracy", "-"),
                drift=item.get("health0_drift_accumulation", "-"),
                continuity=item.get("continuity_error", "-"),
                recovery=item.get("recovery_error_after_health_returns_to_1", "-"),
                fallback=item.get("fallback_rate", "-"),
            )
        )
    return "\n".join(lines) + "\n"


def render_task2_comparison(before: dict[str, object], after: dict[str, object]) -> str:
    return (
        "| Metric | Before | After | Delta |\n"
        "| --- | --- | --- | --- |\n"
        f"| Health0 Drift | {before.get('health0_drift_accumulation', '-')} | {after.get('health0_drift_accumulation', '-')} | "
        f"{round(float(after.get('health0_drift_accumulation', 0.0)) - float(before.get('health0_drift_accumulation', 0.0)), 6)} |\n"
        f"| Fallback Rate | {before.get('fallback_rate', '-')} | {after.get('fallback_rate', '-')} | "
        f"{round(float(after.get('fallback_rate', 0.0)) - float(before.get('fallback_rate', 0.0)), 6)} |\n"
        f"| Recovery Error | {before.get('recovery_error_after_health_returns_to_1', '-')} | {after.get('recovery_error_after_health_returns_to_1', '-')} | "
        f"{round(float(after.get('recovery_error_after_health_returns_to_1', 0.0)) - float(before.get('recovery_error_after_health_returns_to_1', 0.0)), 6)} |\n"
    )


def _safe_mean(values: Iterable[float]) -> float:
    filtered = [float(value) for value in values]
    return mean(filtered) if filtered else 0.0


def _l2_distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    dx = float(first[0]) - float(second[0])
    dy = float(first[1]) - float(second[1])
    dz = float(first[2]) - float(second[2])
    return (dx * dx + dy * dy + dz * dz) ** 0.5
