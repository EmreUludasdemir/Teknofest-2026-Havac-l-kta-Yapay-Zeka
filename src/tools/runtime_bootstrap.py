from __future__ import annotations

from typing import Any

from src.core.frame_state import FrameEnvelope
from src.core.logger import StructuredLogger
from src.pipeline.mvp_processor import MvpFrameProcessor


def build_warmup_image_bytes() -> bytes:
    width = 640
    height = 640
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            value = 24
            if 96 <= x < 544 and 96 <= y < 544:
                value = 210 if ((x // 16 + y // 16) % 2 == 0) else 60
            if x == y or x + y == width - 1:
                value = 255
            pixels[(y * width) + x] = value
    return f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(pixels)


def build_warmup_frame(*, session_name: str = "warmup_session") -> FrameEnvelope:
    return FrameEnvelope(
        frame_url="warmup://frame/0",
        image_url="/warmup/frame_000000.pgm",
        video_name=session_name,
        translation_x=0.0,
        translation_y=0.0,
        translation_z=10.0,
        health_status="1",
        metadata={
            "image_width": 640,
            "image_height": 640,
            "camera_mode": "rgb",
            "frame_id": "warmup-0",
        },
    )
def execute_runtime_warmup(
    *,
    processor: MvpFrameProcessor,
    logger: StructuredLogger,
    session_payload: dict[str, Any] | None = None,
    session_name: str = "warmup_session",
    adapter: Any | None = None,
) -> dict[str, Any]:
    session_payload = session_payload or {}
    reference_manifest = list(session_payload.get("reference_manifest", []))
    if reference_manifest:
        processor.reference_cache.preload_from_manifest(
            reference_manifest,
            orb_features=processor.runtime_settings.task3_orb_features,
        )

    logger.log_runtime(
        event="warmup_policy_started",
        adapter=type(adapter).__name__ if adapter is not None else "RuntimeBootstrap",
        session_name=session_name,
        diagnostics={
            "task1_runtime_order": list(processor.runtime_settings.task1_runtime_order),
            "reference_manifest_count": len(reference_manifest),
            "reference_cache_size": len(processor.reference_cache.list_ids()),
            "task2_calibration_path": str(processor.runtime_settings.task2_calibration_path),
        },
    )

    warmup_frame = build_warmup_frame(session_name=session_name)
    warmup_bytes = build_warmup_image_bytes()
    selection = processor.task1_detector.warm_up_stage(
        warmup_frame,
        warmup_bytes,
    )
    _log_warmup_errors(selection.get("errors", []), logger=logger, session_name=session_name, adapter=adapter)
    attempted_stages = list(selection.get("attempted_stages", []))
    failed_stages = list(selection.get("failed_stages", []))

    selected_stage = str(selection["selected_stage"])
    selected_runtime = str(selection["runtime_name"])
    warmup_runs = _resolve_warmup_runs(processor, selected_runtime)
    logger.log_runtime(
        event="warmup_stage_selected",
        adapter=type(adapter).__name__ if adapter is not None else "RuntimeBootstrap",
        session_name=session_name,
        fallback_mode=selected_stage,
        diagnostics={
            "selected_stage": selected_stage,
            "runtime_name": selected_runtime,
            "warmup_runs": warmup_runs,
            "attempted_stages": attempted_stages,
            "failed_stages": failed_stages,
        },
    )

    for run_index in range(1, warmup_runs):
        run_info = processor.task1_detector.warm_up_stage(
            warmup_frame,
            warmup_bytes,
        )
        _log_warmup_errors(run_info.get("errors", []), logger=logger, session_name=session_name, adapter=adapter)
        attempted_stages = _merge_unique(attempted_stages, run_info.get("attempted_stages", []))
        failed_stages = _merge_unique(failed_stages, run_info.get("failed_stages", []))
        current_stage = str(run_info["selected_stage"])
        if current_stage != selected_stage:
            logger.log_runtime(
                event="warmup_stage_failed",
                adapter=type(adapter).__name__ if adapter is not None else "RuntimeBootstrap",
                session_name=session_name,
                fallback_mode=selected_stage,
                diagnostics={
                    "attempt": run_index + 1,
                    "rerouted_to": current_stage,
                },
            )
            selected_stage = current_stage
            selected_runtime = str(run_info["runtime_name"])
            warmup_runs = max(run_index + 1, _resolve_warmup_runs(processor, selected_runtime))

    if adapter is not None and hasattr(adapter, "mark_warmup_completed"):
        adapter.mark_warmup_completed(
            {
                "active_task1_stage": selected_stage,
                "reference_cache_size": len(processor.reference_cache.list_ids()),
                "attempted_stages": attempted_stages,
                "failed_stages": failed_stages,
            }
        )
    logger.log_runtime(
        event="warmup_stage_completed",
        adapter=type(adapter).__name__ if adapter is not None else "RuntimeBootstrap",
        session_name=session_name,
        fallback_mode=selected_stage,
        diagnostics={
            "active_task1_stage": selected_stage,
            "reference_cache_size": len(processor.reference_cache.list_ids()),
            "attempted_stages": attempted_stages,
            "failed_stages": failed_stages,
        },
    )
    return {
        "active_task1_stage": selected_stage,
        "reference_cache_size": len(processor.reference_cache.list_ids()),
        "task2_calibration_path": str(processor.runtime_settings.task2_calibration_path),
        "warmup_runs": warmup_runs,
        "attempted_stages": attempted_stages,
        "failed_stages": failed_stages,
    }


def _resolve_warmup_runs(processor: MvpFrameProcessor, runtime_name: str) -> int:
    if runtime_name == "tensorrt":
        return max(int(processor.runtime_settings.task1_trt_warmup_runs), 1)
    if runtime_name == "onnxruntime":
        return max(int(processor.runtime_settings.task1_onnx_warmup_runs), 1)
    if runtime_name == "ultralytics":
        return max(int(processor.runtime_settings.task1_native_warmup_runs), 1)
    return 1


def _log_warmup_errors(
    errors: list[str],
    *,
    logger: StructuredLogger,
    session_name: str,
    adapter: Any | None,
) -> None:
    for error_text in errors:
        stage_id, _, reason = str(error_text).partition(":")
        logger.log_runtime(
            event="warmup_stage_failed",
            adapter=type(adapter).__name__ if adapter is not None else "RuntimeBootstrap",
            session_name=session_name,
            fallback_mode=stage_id,
            diagnostics={"error": reason or error_text},
        )


def _merge_unique(existing: list[str], new_items: list[str]) -> list[str]:
    merged = list(existing)
    for item in new_items:
        text = str(item)
        if text not in merged:
            merged.append(text)
    return merged
