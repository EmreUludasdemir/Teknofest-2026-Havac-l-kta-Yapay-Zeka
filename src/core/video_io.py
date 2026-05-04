from __future__ import annotations

from pathlib import Path
from typing import Iterator

from src.core.frame_state import DecodedFrame
from src.core.utils import infer_modality
from src.core.vision import is_cv2_available

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2
else:  # pragma: no cover - cv2 yoksa
    cv2 = None


def iter_video_frames(
    video_path: str | Path,
    *,
    frame_stride: int = 1,
    limit: int | None = None,
    video_name: str = "",
) -> Iterator[DecodedFrame]:
    """Decode sampled video frames into the shared DecodedFrame shape."""

    if not is_cv2_available():
        raise RuntimeError("OpenCV gerekli")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video acilamadi: {video_path}")

    stride = max(int(frame_stride), 1)
    frame_index = 0
    sampled = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
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
