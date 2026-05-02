from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.core.utils import infer_modality
from src.core.vision import is_cv2_available

if is_cv2_available():  # pragma: no branch
    from src.core.vision import cv2
else:  # pragma: no cover
    cv2 = None

MANIFEST_PATH = PROJECT_ROOT / "data" / "task3_eval_manifest.json"
REFERENCE_DIR = PROJECT_ROOT / "data" / "references" / "2026_baseline"
OUTPUT_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-05-02_d7_reality_check"

PASS_A_SCENARIO_IDS = (
    "rgb_reference_session",
    "thermal_cross_sensor_proxy",
    "rgb_absent_target_proxy_2025",
    "thermal_absent_target_proxy_2025",
)
REFERENCE_IDS = tuple(f"ref_{index:02d}" for index in range(1, 13))


@dataclass(slots=True)
class ScenarioConfig:
    scenario_id: str
    name: str
    video: str
    references_dir: str
    reference_mode: str
    frame_stride: int
    frame_limit: int | None
    expected_present_refs: list[str]
    modality_routing_breach_refs: list[str]
    ambiguous_routing_refs: list[str]
    tags: list[str]


def load_manifest() -> list[ScenarioConfig]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    base_settings = MvpRuntimeSettings()
    scenarios: list[ScenarioConfig] = []
    for index, raw in enumerate(payload.get("scenarios", []), start=1):
        scenario_id = str(raw.get("id") or raw.get("name") or raw.get("scenario_id") or f"scenario_{index:02d}")
        scenarios.append(
            ScenarioConfig(
                scenario_id=scenario_id,
                name=str(raw.get("name") or scenario_id),
                video=str(raw["video"]),
                references_dir=str(raw.get("references_dir") or raw.get("reference_dir") or REFERENCE_DIR),
                reference_mode=str(raw.get("reference_mode") or "present_targets"),
                frame_stride=int(raw.get("frame_stride", base_settings.task3_eval_frame_stride)),
                frame_limit=raw.get("frame_limit", base_settings.task3_eval_frame_limit),
                expected_present_refs=list(raw.get("expected_present_refs", [])),
                modality_routing_breach_refs=list(raw.get("modality_routing_breach_refs", [])),
                ambiguous_routing_refs=list(raw.get("ambiguous_routing_refs", [])),
                tags=list(raw.get("tags", [])),
            )
        )
    return scenarios


def load_scenarios(*, scenario_ids: tuple[str, ...] = PASS_A_SCENARIO_IDS) -> dict[str, ScenarioConfig]:
    scenarios = {scenario.scenario_id: scenario for scenario in load_manifest()}
    return {scenario_id: scenarios[scenario_id] for scenario_id in scenario_ids}


def make_settings(
    reference_dir: str | Path = REFERENCE_DIR,
    **overrides: object,
) -> MvpRuntimeSettings:
    payload = {
        "task3_mode": "yoloe_vp_lightglue",
        "task3_reference_dir": Path(reference_dir),
        "task3_eval_reference_dir": Path(reference_dir),
        "task3_yoloe_allow_cpu": True,
    }
    payload.update(overrides)
    return MvpRuntimeSettings(**payload)


def frame_envelope(scenario_id: str, frame_index: int, width: int, height: int, *, tag: str = "d7") -> FrameEnvelope:
    return FrameEnvelope(
        frame_url=f"http://{tag}/frames/{frame_index + 1}/",
        image_url=f"/{tag}/{frame_index + 1}.jpg",
        video_name=scenario_id,
        translation_x=0.0,
        translation_y=0.0,
        translation_z=0.0,
        health_status="1",
        metadata={"frame_index": frame_index, "image_width": width, "image_height": height},
    )


def iter_timed_video_frames(
    video_path: str | Path,
    *,
    frame_stride: int = 1,
    limit: int | None = None,
    video_name: str = "",
) -> Iterator[tuple[DecodedFrame, float]]:
    if not is_cv2_available():
        raise RuntimeError("OpenCV gerekli")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video acilamadi: {video_path}")

    frame_index = 0
    sampled = 0
    try:
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            decode_ms = (time.perf_counter() - started) * 1000.0
            if not ok:
                break
            if frame_index % max(frame_stride, 1) != 0:
                frame_index += 1
                continue
            gray_started = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_ms = (time.perf_counter() - gray_started) * 1000.0
            height, width = frame.shape[:2]
            yield (
                DecodedFrame(
                    bgr=frame,
                    gray=gray,
                    width=int(width),
                    height=int(height),
                    channel_count=int(frame.shape[2]) if len(frame.shape) == 3 else 1,
                    modality=infer_modality(video_name or str(video_path), width=int(width), height=int(height)),
                    frame_index=frame_index,
                ),
                round(decode_ms + gray_ms, 6),
            )
            sampled += 1
            frame_index += 1
            if limit is not None and sampled >= limit:
                break
    finally:
        capture.release()
