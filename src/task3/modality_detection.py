from __future__ import annotations

"""Standalone Task 3 reference modality detection.

Decision policy:

1. EXIF wins when it provides an explicit capture-mode signal.
2. DJI `ImageDescription=default` is treated as an explicit RGB-mode marker for
   the official 2026 M3T reference set.
3. Thermal keywords such as `WhiteHot` are treated as explicit thermal markers.
4. When EXIF is absent or inconclusive, pixel analysis is used:
   - low saturation + high inter-channel correlation suggest grayscale content
   - a bimodal histogram strengthens the thermal hypothesis
   - higher saturation + less-correlated channels suggest RGB content

Named thresholds are kept at module scope so follow-up calibration can adjust
them without rewriting detection logic.
"""

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError

from src.core.vision import cv2, is_cv2_available

MAX_ANALYSIS_LONG_SIDE = 640
HSV_GRAY_SATURATION_MAX = 10.0
HSV_RGB_SATURATION_MIN = 20.0
CHANNEL_CORRELATION_GRAY_MIN = 0.98
CHANNEL_CORRELATION_RGB_MAX = 0.97
HISTOGRAM_BIN_COUNT = 32
HISTOGRAM_PEAK_MIN_FRACTION = 0.015
HISTOGRAM_MIN_PEAK_SEPARATION_BINS = 4
HISTOGRAM_DIP_RATIO_MAX = 0.60
HISTOGRAM_SPREAD_MIN_FRACTION = 0.18
DYNAMIC_RANGE_MIN = 40.0

THERMAL_TEXT_KEYWORDS = ("whitehot", "blackhot", "hothot", "thermal", "infrared", "lwir", "mwir")
THERMAL_EXACT_SHORT_KEYWORDS = {"ir"}
RGB_TEXT_KEYWORDS = ("default",)
THERMAL_ONLY_MODEL_KEYWORDS: tuple[str, ...] = ()
EXIF_TEXT_FIELDS = {
    "ImageDescription",
    "UserComment",
    "XPComment",
    "XPSubject",
    "XPTitle",
    "Software",
    "Make",
    "Model",
}
WEAK_RGB_EXIF_FIELDS = {"WhiteBalance", "Saturation", "ColorSpace"}


class Modality(Enum):
    RGB = "rgb"
    THERMAL = "thermal"
    UNKNOWN = "unknown"


def detect_modality(image_path: Path) -> tuple[Modality, dict[str, Any]]:
    """Classify a reference image as RGB, thermal, or unknown."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Image file is empty: {path}")

    try:
        with Image.open(path) as image:
            image.load()
            exif_info = _extract_exif_info(image)
            rgb_image = image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Invalid image file: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read image file: {path}") from exc

    pixel_signals = _analyze_pixels(np.asarray(rgb_image, dtype=np.uint8))
    diagnostics: dict[str, Any] = {
        "method": "none",
        "exif_signals": list(exif_info["signals"]),
        "pixel_signals": pixel_signals,
        "confidence": "low",
        "reason": "No modality signal found.",
    }

    if exif_info["strong_thermal"]:
        diagnostics.update(
            {
                "method": "exif",
                "confidence": "high",
                "reason": f"Thermal EXIF signal matched: {', '.join(exif_info['signals'])}.",
            }
        )
        return Modality.THERMAL, diagnostics

    if exif_info["strong_rgb"]:
        diagnostics.update(
            {
                "method": "exif",
                "confidence": "high",
                "reason": f"RGB EXIF signal matched: {', '.join(exif_info['signals'])}.",
            }
        )
        return Modality.RGB, diagnostics

    pixel_modality, pixel_confidence, pixel_reason = _classify_from_pixel_signals(pixel_signals)

    if exif_info["weak_rgb"] and pixel_modality == Modality.RGB:
        diagnostics.update(
            {
                "method": "both",
                "confidence": "high",
                "reason": f"Weak RGB EXIF evidence agrees with pixel analysis: {pixel_reason}",
            }
        )
        return Modality.RGB, diagnostics

    if exif_info["weak_rgb"]:
        diagnostics.update(
            {
                "method": "exif",
                "confidence": "medium",
                "reason": "Weak RGB EXIF evidence was present but pixel analysis was inconclusive or conflicting.",
            }
        )
        return Modality.RGB, diagnostics

    diagnostics.update(
        {
            "method": "pixel",
            "confidence": pixel_confidence,
            "reason": pixel_reason,
        }
    )
    return pixel_modality, diagnostics


def _extract_exif_info(image: Image.Image) -> dict[str, Any]:
    signals: list[str] = []
    strong_thermal = False
    strong_rgb = False
    weak_rgb = False

    try:
        exif = image.getexif()
    except Exception:
        exif = None

    if not exif:
        return {
            "signals": signals,
            "strong_thermal": strong_thermal,
            "strong_rgb": strong_rgb,
            "weak_rgb": weak_rgb,
        }

    tag_names = ExifTags.TAGS
    text_fields: dict[str, str] = {}
    for raw_key, raw_value in exif.items():
        tag_name = str(tag_names.get(raw_key, raw_key))
        value_text = _coerce_exif_value_to_text(raw_value)
        if tag_name in EXIF_TEXT_FIELDS and value_text:
            text_fields[tag_name] = value_text
        if tag_name in WEAK_RGB_EXIF_FIELDS:
            numeric_value = _coerce_exif_value_to_number(raw_value)
            if tag_name == "Saturation" and numeric_value is not None and numeric_value != 0:
                weak_rgb = True
                signals.append("saturation")
            elif tag_name in {"WhiteBalance", "ColorSpace"}:
                weak_rgb = True
                signals.append(tag_name.lower())

    for field_name, value_text in text_fields.items():
        lowered = value_text.lower()
        if any(keyword in lowered for keyword in RGB_TEXT_KEYWORDS):
            strong_rgb = True
            signals.append("default")
        for keyword in THERMAL_TEXT_KEYWORDS:
            if keyword in lowered:
                strong_thermal = True
                signals.append(keyword)
        tokenized = {token.strip(" ,;:_-") for token in lowered.split()}
        if any(keyword in tokenized for keyword in THERMAL_EXACT_SHORT_KEYWORDS):
            strong_thermal = True
            signals.extend(sorted(THERMAL_EXACT_SHORT_KEYWORDS.intersection(tokenized)))
        if field_name == "Model" and any(keyword in lowered for keyword in THERMAL_ONLY_MODEL_KEYWORDS):
            strong_thermal = True
            signals.append("thermal_model")

    deduped_signals: list[str] = []
    for signal in signals:
        if signal not in deduped_signals:
            deduped_signals.append(signal)

    return {
        "signals": deduped_signals,
        "strong_thermal": strong_thermal,
        "strong_rgb": strong_rgb,
        "weak_rgb": weak_rgb,
    }


def _analyze_pixels(rgb_image: np.ndarray) -> dict[str, Any]:
    analysis_image = _resize_for_analysis(rgb_image)
    mean_saturation = _compute_mean_saturation(analysis_image)
    correlations = _compute_channel_correlations(analysis_image)
    min_channel_correlation = min(correlations.values()) if correlations else 1.0
    gray = _to_grayscale(analysis_image)
    histogram_info = _compute_histogram_signals(gray)
    p5, p95 = np.percentile(gray, [5, 95])
    dynamic_range = float(p95 - p5)

    return {
        "analysis_shape": [int(analysis_image.shape[0]), int(analysis_image.shape[1])],
        "mean_saturation": round(mean_saturation, 4),
        "channel_correlations": {key: round(value, 6) for key, value in correlations.items()},
        "min_channel_correlation": round(float(min_channel_correlation), 6),
        "histogram_bimodal": bool(histogram_info["bimodal"]),
        "histogram_peak_bins": list(histogram_info["peak_bins"]),
        "histogram_peak_values": [round(float(value), 6) for value in histogram_info["peak_values"]],
        "histogram_dip_ratio": round(float(histogram_info["dip_ratio"]), 6),
        "histogram_spread_fraction": round(float(histogram_info["spread_fraction"]), 6),
        "dynamic_range_p5_p95": round(dynamic_range, 4),
    }


def _classify_from_pixel_signals(pixel_signals: dict[str, Any]) -> tuple[Modality, str, str]:
    mean_saturation = float(pixel_signals["mean_saturation"])
    min_corr = float(pixel_signals["min_channel_correlation"])
    bimodal = bool(pixel_signals["histogram_bimodal"])
    spread_fraction = float(pixel_signals["histogram_spread_fraction"])
    dynamic_range = float(pixel_signals["dynamic_range_p5_p95"])

    near_grayscale = mean_saturation < HSV_GRAY_SATURATION_MAX and min_corr >= CHANNEL_CORRELATION_GRAY_MIN
    clearly_color = mean_saturation >= HSV_RGB_SATURATION_MIN and min_corr <= CHANNEL_CORRELATION_GRAY_MIN

    if near_grayscale and bimodal:
        return (
            Modality.THERMAL,
            "high",
            (
                "Pixel analysis indicates thermal content: low saturation, very high channel correlation, "
                "and a bimodal grayscale histogram."
            ),
        )
    if near_grayscale and dynamic_range >= DYNAMIC_RANGE_MIN:
        return (
            Modality.THERMAL,
            "medium",
            "Pixel analysis indicates grayscale / thermal-like content with broad dynamic range.",
        )
    if clearly_color and spread_fraction >= HISTOGRAM_SPREAD_MIN_FRACTION:
        return (
            Modality.RGB,
            "high",
            "Pixel analysis indicates RGB content with meaningful saturation and non-grayscale channels.",
        )
    if clearly_color:
        return (
            Modality.RGB,
            "medium",
            "Pixel analysis indicates RGB-like channel diversity, but histogram spread is limited.",
        )
    return (
        Modality.UNKNOWN,
        "low",
        "Pixel analysis produced mixed signals; modality left unknown.",
    )


def _resize_for_analysis(rgb_image: np.ndarray) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    long_side = max(height, width)
    if long_side <= MAX_ANALYSIS_LONG_SIDE:
        return rgb_image
    scale = MAX_ANALYSIS_LONG_SIDE / float(long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if is_cv2_available():
        return cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(rgb_image).resize((new_width, new_height), Image.Resampling.BILINEAR))


def _compute_mean_saturation(rgb_image: np.ndarray) -> float:
    if is_cv2_available():
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        return float(hsv[..., 1].mean())
    hsv = np.asarray(Image.fromarray(rgb_image).convert("HSV"), dtype=np.uint8)
    return float(hsv[..., 1].mean())


def _compute_channel_correlations(rgb_image: np.ndarray) -> dict[str, float]:
    correlations: dict[str, float] = {}
    channels = [rgb_image[..., index].astype(np.float32).reshape(-1) for index in range(3)]
    labels = [("r_g", 0, 1), ("r_b", 0, 2), ("g_b", 1, 2)]
    for label, left_idx, right_idx in labels:
        left = channels[left_idx]
        right = channels[right_idx]
        left_std = float(np.std(left))
        right_std = float(np.std(right))
        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))
        if left_std < 1e-6 and right_std < 1e-6:
            correlations[label] = 1.0 if abs(left_mean - right_mean) < 1.0 else 0.0
            continue
        if left_std < 1e-6 or right_std < 1e-6:
            correlations[label] = 0.0
            continue
        correlations[label] = float(np.corrcoef(left, right)[0, 1])
    return correlations


def _to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    if is_cv2_available():
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    red = rgb_image[..., 0].astype(np.float32)
    green = rgb_image[..., 1].astype(np.float32)
    blue = rgb_image[..., 2].astype(np.float32)
    return np.clip((0.299 * red) + (0.587 * green) + (0.114 * blue), 0, 255).astype(np.uint8)


def _compute_histogram_signals(gray_image: np.ndarray) -> dict[str, Any]:
    histogram, _ = np.histogram(gray_image, bins=HISTOGRAM_BIN_COUNT, range=(0, 256))
    histogram = histogram.astype(np.float64)
    histogram_sum = float(histogram.sum())
    if histogram_sum <= 0.0:
        return {
            "bimodal": False,
            "peak_bins": [],
            "peak_values": [],
            "dip_ratio": 1.0,
            "spread_fraction": 0.0,
        }

    histogram /= histogram_sum
    spread_fraction = float(np.count_nonzero(histogram > HISTOGRAM_PEAK_MIN_FRACTION) / float(HISTOGRAM_BIN_COUNT))

    peaks: list[tuple[int, float]] = []
    for index, value in enumerate(histogram):
        left = histogram[index - 1] if index > 0 else -1.0
        right = histogram[index + 1] if index + 1 < len(histogram) else -1.0
        if value >= left and value >= right and value >= HISTOGRAM_PEAK_MIN_FRACTION:
            peaks.append((index, float(value)))

    best_pair: tuple[tuple[int, float], tuple[int, float]] | None = None
    best_pair_score = -1.0
    for left_index in range(len(peaks)):
        for right_index in range(left_index + 1, len(peaks)):
            left_peak = peaks[left_index]
            right_peak = peaks[right_index]
            separation = abs(left_peak[0] - right_peak[0])
            if separation < HISTOGRAM_MIN_PEAK_SEPARATION_BINS:
                continue
            pair_score = min(left_peak[1], right_peak[1])
            if pair_score > best_pair_score:
                best_pair = (left_peak, right_peak)
                best_pair_score = pair_score

    if best_pair is None:
        top_peaks = sorted(peaks, key=lambda item: item[1], reverse=True)[:2]
        return {
            "bimodal": False,
            "peak_bins": [item[0] for item in top_peaks],
            "peak_values": [item[1] for item in top_peaks],
            "dip_ratio": 1.0,
            "spread_fraction": spread_fraction,
        }

    left_peak, right_peak = sorted(best_pair, key=lambda item: item[0])
    valley = histogram[left_peak[0] : right_peak[0] + 1]
    valley_floor = float(valley.min()) if len(valley) else 1.0
    smaller_peak = min(left_peak[1], right_peak[1])
    dip_ratio = valley_floor / max(smaller_peak, 1e-9)
    return {
        "bimodal": dip_ratio <= HISTOGRAM_DIP_RATIO_MAX,
        "peak_bins": [left_peak[0], right_peak[0]],
        "peak_values": [left_peak[1], right_peak[1]],
        "dip_ratio": float(dip_ratio),
        "spread_fraction": spread_fraction,
    }


def _coerce_exif_value_to_text(value: Any) -> str:
    if isinstance(value, bytes):
        for encoding in ("utf-8", "utf-16", "latin1"):
            try:
                return value.decode(encoding, errors="ignore").strip("\x00 ").strip()
            except Exception:
                continue
        return ""
    if isinstance(value, (tuple, list)):
        return " ".join(_coerce_exif_value_to_text(item) for item in value if _coerce_exif_value_to_text(item))
    if value is None:
        return ""
    return str(value).strip()


def _coerce_exif_value_to_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
        denominator = float(value[1])
        if denominator == 0.0:
            return None
        return float(value[0]) / denominator
    return None
