from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class VramSnapshot:
    available: bool
    used_mb: float | None
    total_mb: float | None
    raw_output: str = ""
    error: str | None = None


def query_vram(command: tuple[str, ...]) -> VramSnapshot:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # pragma: no cover - ortama bagli
        return VramSnapshot(available=False, used_mb=None, total_mb=None, error=str(exc))

    if completed.returncode != 0 or not completed.stdout.strip():
        return VramSnapshot(
            available=False,
            used_mb=None,
            total_mb=None,
            raw_output=completed.stdout.strip(),
            error=completed.stderr.strip() or f"returncode={completed.returncode}",
        )

    first_line = completed.stdout.strip().splitlines()[0]
    try:
        used_text, total_text = [part.strip() for part in first_line.split(",", maxsplit=1)]
        return VramSnapshot(
            available=True,
            used_mb=float(used_text),
            total_mb=float(total_text),
            raw_output=first_line,
        )
    except Exception as exc:  # pragma: no cover - parse hatasi ortam bagli
        return VramSnapshot(
            available=False,
            used_mb=None,
            total_mb=None,
            raw_output=first_line,
            error=str(exc),
        )


def recovery_mb(before_unload: VramSnapshot, after_unload: VramSnapshot) -> float | None:
    if before_unload.used_mb is None or after_unload.used_mb is None:
        return None
    return round(before_unload.used_mb - after_unload.used_mb, 3)
