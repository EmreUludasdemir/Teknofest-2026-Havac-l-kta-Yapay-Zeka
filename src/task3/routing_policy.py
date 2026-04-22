from __future__ import annotations

"""Standalone detector-assignment policy for Task 3 references.

Policy summary:

1. Modality is inferred through :mod:`src.task3.modality_detection`.
2. High-confidence thermal references route to YOLOE on thermal only.
3. High-confidence RGB references use fast visibility heuristics:
   - very large RGB references tend to downsample poorly for YOLOE VPE
   - extreme aspect ratios often correlate with scene-like captures
   - medium/large scene-like RGB captures are routed to ORB
4. Any uncertain classification falls back to a dual-path assignment so later
   phases can decide between detectors without losing coverage.

The thresholds below are intentionally named and colocated for follow-up
calibration. They were chosen to reproduce the known official-reference routing
decisions from 2026-04-21 measurements.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from src.core.vision import cv2, is_cv2_available
from src.task3.modality_detection import Modality, detect_modality

VISIBILITY_ANALYSIS_LONG_SIDE = 768
LARGE_REFERENCE_LONG_SIDE_PX = 1800
EXTREME_ASPECT_RATIO_THRESHOLD = 3.0
SCENE_LIKE_LONG_SIDE_MIN = 1400
SCENE_LIKE_CENTER_FRACTION_MAX = 0.26
SCENE_LIKE_CENTER_DENSITY_RATIO_MAX = 1.10
CENTER_WINDOW_FRACTION = 0.50


@dataclass(slots=True)
class RoutingAssignment:
    detector: Literal["yoloe", "orb", "both"]
    detector_modalities: list[str]
    confidence: Literal["high", "medium", "low"]
    rationale: str
    signals: dict[str, Any]


def assign_detector(image_path: Path) -> RoutingAssignment:
    """Determine detector routing for a reference image."""

    modality, modality_diagnostics = detect_modality(Path(image_path))
    modality_confidence = str(modality_diagnostics.get("confidence", "low"))

    base_signals = {
        "modality": modality.value,
        "modality_method": modality_diagnostics.get("method"),
        "modality_confidence": modality_confidence,
        "modality_reason": modality_diagnostics.get("reason"),
        "modality_exif_signals": list(modality_diagnostics.get("exif_signals", [])),
        "modality_pixel_signals": dict(modality_diagnostics.get("pixel_signals", {})),
    }

    if modality == Modality.THERMAL and modality_confidence == "high":
        return RoutingAssignment(
            detector="yoloe",
            detector_modalities=["thermal"],
            confidence="high",
            rationale="Thermal modality confirmed; keep YOLOE on thermal with the frozen strict threshold.",
            signals=base_signals,
        )

    if modality == Modality.RGB and modality_confidence in {"high", "medium"}:
        visibility_signals = _compute_visibility_signals(Path(image_path))
        signals = dict(base_signals)
        signals["visibility"] = visibility_signals

        prefer_orb_reasons: list[str] = []
        if visibility_signals["is_large"]:
            prefer_orb_reasons.append("large_rgb_reference")
        if visibility_signals["is_extreme_aspect"]:
            prefer_orb_reasons.append("extreme_aspect_ratio")
        if visibility_signals["is_scene_like"]:
            prefer_orb_reasons.append("scene_like_layout")

        if prefer_orb_reasons:
            return RoutingAssignment(
                detector="orb",
                detector_modalities=["rgb"],
                confidence="medium",
                rationale=(
                    "RGB reference is predicted to be weak for YOLOE visibility due to "
                    + ", ".join(prefer_orb_reasons)
                    + "."
                ),
                signals=signals,
            )

        return RoutingAssignment(
            detector="yoloe",
            detector_modalities=["rgb"],
            confidence="high" if modality_confidence == "high" else "medium",
            rationale="RGB reference is compact/object-focused enough for YOLOE visibility heuristics.",
            signals=signals,
        )

    return RoutingAssignment(
        detector="both",
        detector_modalities=["rgb", "thermal"],
        confidence="low",
        rationale="Modality confidence is not high enough for single-detector routing; use conservative dual-path fallback.",
        signals=base_signals,
    )


def _compute_visibility_signals(image_path: Path) -> dict[str, Any]:
    """Return deterministic visibility features for RGB routing decisions."""

    rgb_image = _load_rgb_image(Path(image_path))
    original_height, original_width = rgb_image.shape[:2]
    original_long_side = max(original_height, original_width)
    original_short_side = max(min(original_height, original_width), 1)
    original_aspect_ratio = float(original_long_side) / float(original_short_side)

    analysis_image = _resize_for_visibility_analysis(rgb_image)
    gray = _to_grayscale(analysis_image)
    gradient_magnitude = _compute_gradient_magnitude(gray)

    height, width = gray.shape[:2]
    center_fraction = CENTER_WINDOW_FRACTION
    center_height = max(int(round(height * center_fraction)), 1)
    center_width = max(int(round(width * center_fraction)), 1)
    top = max((height - center_height) // 2, 0)
    left = max((width - center_width) // 2, 0)
    bottom = min(top + center_height, height)
    right = min(left + center_width, width)

    center_patch = gradient_magnitude[top:bottom, left:right]
    total_sum = float(gradient_magnitude.sum())
    center_sum = float(center_patch.sum())
    total_area = max(int(gradient_magnitude.size), 1)
    center_area = max(int(center_patch.size), 1)
    outer_sum = max(total_sum - center_sum, 0.0)
    outer_area = max(total_area - center_area, 1)

    center_density = center_sum / float(center_area)
    outer_density = outer_sum / float(outer_area)
    saliency_concentration = center_sum / max(total_sum, 1e-9)
    center_density_ratio = center_density / max(outer_density, 1e-9)

    is_large = original_long_side > LARGE_REFERENCE_LONG_SIDE_PX
    is_extreme_aspect = original_aspect_ratio >= EXTREME_ASPECT_RATIO_THRESHOLD
    is_scene_like = (
        original_long_side >= SCENE_LIKE_LONG_SIDE_MIN
        and saliency_concentration < SCENE_LIKE_CENTER_FRACTION_MAX
        and center_density_ratio < SCENE_LIKE_CENTER_DENSITY_RATIO_MAX
    )

    return {
        "long_side_px": int(original_long_side),
        "aspect_ratio": round(original_aspect_ratio, 6),
        "saliency_concentration": round(float(saliency_concentration), 6),
        "center_density_ratio": round(float(center_density_ratio), 6),
        "is_large": bool(is_large),
        "is_extreme_aspect": bool(is_extreme_aspect),
        "is_scene_like": bool(is_scene_like),
        "analysis_shape": [int(height), int(width)],
    }


def _load_rgb_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resize_for_visibility_analysis(rgb_image: np.ndarray) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    long_side = max(height, width)
    if long_side <= VISIBILITY_ANALYSIS_LONG_SIDE:
        return rgb_image
    scale = VISIBILITY_ANALYSIS_LONG_SIDE / float(long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if is_cv2_available():
        return cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(rgb_image).resize((new_width, new_height), Image.Resampling.BILINEAR))


def _to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    if is_cv2_available():
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    red = rgb_image[..., 0].astype(np.float32)
    green = rgb_image[..., 1].astype(np.float32)
    blue = rgb_image[..., 2].astype(np.float32)
    return np.clip((0.299 * red) + (0.587 * green) + (0.114 * blue), 0, 255).astype(np.uint8)


def _compute_gradient_magnitude(gray_image: np.ndarray) -> np.ndarray:
    if is_cv2_available():
        gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        return cv2.GaussianBlur(magnitude, (0, 0), 1.0)

    gray = gray_image.astype(np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.sqrt((gx * gx) + (gy * gy))
    return magnitude
