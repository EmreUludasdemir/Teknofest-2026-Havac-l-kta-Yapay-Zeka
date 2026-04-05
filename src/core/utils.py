from __future__ import annotations

import re
from typing import Iterable


def extract_frame_index(frame_url: str) -> int:
    """URL veya kimlik icinden deterministik frame indeksi cikarir."""

    numbers = re.findall(r"\d+", frame_url)
    if not numbers:
        return 0
    return int(numbers[-1])


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def average_pair(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def infer_modality(
    video_name: str | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    camera_mode: str | None = None,
) -> str:
    if camera_mode:
        lowered = camera_mode.lower()
        if "term" in lowered or "thermal" in lowered:
            return "thermal"
        if "rgb" in lowered:
            return "rgb"

    if width == 640 and height == 512:
        return "thermal"

    if video_name:
        lowered = video_name.lower()
        if "term" in lowered or "thermal" in lowered:
            return "thermal"
        if "rgb" in lowered:
            return "rgb"

    return "rgb"


def percentile(values: Iterable[float], value: float) -> float:
    collected = sorted(float(item) for item in values)
    if not collected:
        return 0.0
    if len(collected) == 1:
        return collected[0]
    rank = clamp(value, 0.0, 100.0) / 100.0 * (len(collected) - 1)
    low_index = int(rank)
    high_index = min(low_index + 1, len(collected) - 1)
    weight = rank - low_index
    return collected[low_index] * (1.0 - weight) + collected[high_index] * weight


def bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def normalize_box(box: tuple[float, float, float, float], *, width: float, height: float) -> tuple[float, float, float, float]:
    return (
        clamp(box[0], 0.0, width),
        clamp(box[1], 0.0, height),
        clamp(box[2], 0.0, width),
        clamp(box[3], 0.0, height),
    )
