from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.core.utils import infer_modality


@dataclass(slots=True)
class CameraCalibrationProfile:
    modality: str
    focal_x: float
    focal_y: float
    principal_x: float
    principal_y: float
    image_height: int
    image_width: int
    source: str


@dataclass(slots=True)
class CalibrationBundle:
    thermal: CameraCalibrationProfile
    rgb: CameraCalibrationProfile
    source_path: str


def _extract_numbers(line: str) -> list[float]:
    return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", line, flags=re.IGNORECASE)]


def _fallback_bundle(path: str | Path) -> CalibrationBundle:
    return CalibrationBundle(
        thermal=CameraCalibrationProfile(
            modality="thermal",
            focal_x=731.7965,
            focal_y=732.0172,
            principal_x=319.2367,
            principal_y=251.2424,
            image_height=512,
            image_width=640,
            source="fallback_defaults",
        ),
        rgb=CameraCalibrationProfile(
            modality="rgb",
            focal_x=2792.2,
            focal_y=2795.2,
            principal_x=1988.0,
            principal_y=1562.2,
            image_height=3000,
            image_width=4000,
            source="fallback_defaults",
        ),
        source_path=str(path),
    )


@lru_cache(maxsize=4)
def load_calibration_bundle(path: str | Path) -> CalibrationBundle:
    calibration_path = Path(path)
    if not calibration_path.exists():
        return _fallback_bundle(path)

    sections: dict[str, dict[str, list[float]]] = {"thermal": {}, "rgb": {}}
    current: str | None = None
    for raw_line in calibration_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("termal camera intrinsics"):
            current = "thermal"
            continue
        if lowered.startswith("rgb camera intrinsics"):
            current = "rgb"
            continue
        if current is None or ":" not in line:
            continue
        key, _ = line.split(":", 1)
        values = _extract_numbers(line)
        if values:
            sections[current][key.strip()] = values

    fallback = _fallback_bundle(path)

    def build_profile(name: str, fallback_profile: CameraCalibrationProfile) -> CameraCalibrationProfile:
        values = sections.get(name, {})
        focal = values.get("FocalLength", [fallback_profile.focal_x, fallback_profile.focal_y])
        principal = values.get("PrincipalPoint", [fallback_profile.principal_x, fallback_profile.principal_y])
        image_size = values.get("ImageSize", [fallback_profile.image_height, fallback_profile.image_width])
        return CameraCalibrationProfile(
            modality=name,
            focal_x=float(focal[0]),
            focal_y=float(focal[1]),
            principal_x=float(principal[0]),
            principal_y=float(principal[1]),
            image_height=int(image_size[0]),
            image_width=int(image_size[1]),
            source=str(calibration_path),
        )

    return CalibrationBundle(
        thermal=build_profile("thermal", fallback.thermal),
        rgb=build_profile("rgb", fallback.rgb),
        source_path=str(calibration_path),
    )


def select_calibration_profile(
    bundle: CalibrationBundle,
    frame: FrameEnvelope,
    decoded_frame: DecodedFrame | None = None,
) -> CameraCalibrationProfile:
    modality = infer_modality(
        frame.video_name,
        width=decoded_frame.width if decoded_frame is not None else int(frame.metadata.get("image_width", 0) or 0),
        height=decoded_frame.height if decoded_frame is not None else int(frame.metadata.get("image_height", 0) or 0),
        camera_mode=str(frame.metadata.get("camera_mode", "")) or None,
    )
    return bundle.thermal if modality == "thermal" else bundle.rgb


def build_calibration_state(frames: list[FrameEnvelope], runtime_settings: MvpRuntimeSettings | None = None) -> dict[str, object]:
    settings = runtime_settings or MvpRuntimeSettings()
    bundle = load_calibration_bundle(settings.task2_calibration_path)
    return {
        "frame_count": len(frames),
        "source_path": bundle.source_path,
        "thermal_size": [bundle.thermal.image_height, bundle.thermal.image_width],
        "rgb_size": [bundle.rgb.image_height, bundle.rgb.image_width],
    }


def pixel_shift_to_translation_delta(
    shift_x_px: float,
    shift_y_px: float,
    *,
    reference_z: float,
    profile: CameraCalibrationProfile,
) -> tuple[float, float]:
    depth_scale = max(abs(reference_z), 1.0)
    delta_x = (shift_x_px / max(profile.focal_x, 1.0)) * depth_scale
    delta_y = (shift_y_px / max(profile.focal_y, 1.0)) * depth_scale
    return float(delta_x), float(delta_y)
