from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StructuredLogger:
    """JSONL tabanli basit structured logger."""

    log_dir: Path | str = Path("_logs")
    console: bool = False
    runtime_path: Path = field(init=False)
    error_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_path = self.log_dir / "runtime.jsonl"
        self.error_path = self.log_dir / "error.jsonl"

    def log_runtime(
        self,
        *,
        event: str,
        adapter: str | None = None,
        session_name: str | None = None,
        frame_url: str | None = None,
        video_name: str | None = None,
        health_status: str | None = None,
        status_code: int | None = None,
        latency_ms: float | None = None,
        detected_objects: int | None = None,
        detected_translations: int | None = None,
        fallback_mode: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.log_event(
            level="INFO",
            event=event,
            adapter=adapter,
            session_name=session_name,
            frame_url=frame_url,
            video_name=video_name,
            health_status=health_status,
            status_code=status_code,
            latency_ms=latency_ms,
            detected_objects=detected_objects,
            detected_translations=detected_translations,
            fallback_mode=fallback_mode,
            diagnostics=diagnostics,
        )

    def log_error(
        self,
        *,
        event: str,
        adapter: str | None = None,
        session_name: str | None = None,
        frame_url: str | None = None,
        video_name: str | None = None,
        health_status: str | None = None,
        status_code: int | None = None,
        latency_ms: float | None = None,
        detected_objects: int | None = None,
        detected_translations: int | None = None,
        fallback_mode: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.log_event(
            level="ERROR",
            event=event,
            adapter=adapter,
            session_name=session_name,
            frame_url=frame_url,
            video_name=video_name,
            health_status=health_status,
            status_code=status_code,
            latency_ms=latency_ms,
            detected_objects=detected_objects,
            detected_translations=detected_translations,
            fallback_mode=fallback_mode,
            diagnostics=diagnostics,
        )

    def log_event(
        self,
        *,
        level: str,
        event: str,
        adapter: str | None = None,
        session_name: str | None = None,
        frame_url: str | None = None,
        video_name: str | None = None,
        health_status: str | None = None,
        status_code: int | None = None,
        latency_ms: float | None = None,
        detected_objects: int | None = None,
        detected_translations: int | None = None,
        fallback_mode: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            "adapter": adapter,
            "session_name": session_name,
            "frame_url": frame_url,
            "video_name": video_name,
            "health_status": health_status,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "detected_objects": detected_objects,
            "detected_translations": detected_translations,
            "fallback_mode": fallback_mode,
            "diagnostics": diagnostics or {},
        }

        target_path = self.error_path if level.upper() in {"ERROR", "CRITICAL"} else self.runtime_path
        with target_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        if self.console:
            print(f"[{record['level']}] {record['event']} {record['diagnostics']}")

        return record
