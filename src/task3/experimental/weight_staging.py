from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.config.settings import MvpRuntimeSettings

try:  # pragma: no cover - ortama bagli
    from ultralytics import YOLOE  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - ortama bagli
    YOLOE = None


STAGING_TRUE = "TRUE_YOLOE_VISUAL_PROMPT"
STAGING_WEIGHT_INCOMPATIBLE = "YOLOE_CLASS_PRESENT_BUT_WEIGHT_INCOMPATIBLE"
STAGING_RUNTIME_INCOMPATIBLE = "YOLOE_RUNTIME_INCOMPATIBLE"
STAGING_WEIGHT_MISSING = "WEIGHT_MISSING"


@dataclass(slots=True)
class WeightCandidate:
    path: Path
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "model_family": self.path.stem,
            "expected_input_mode": "frame_bgr + visual_prompts + refer_image",
        }


def default_search_roots(runtime_settings: MvpRuntimeSettings) -> list[Path]:
    configured = runtime_settings.task3_experimental_model_path
    roots: list[Path] = []
    if configured is not None:
        roots.append(Path(configured).expanduser().parent)
    roots.extend(
        [
            Path.home() / ".teknofest_models" / "task3",
            Path.home() / ".teknofest_models",
            Path.home() / "Downloads",
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path.home() / "OneDrive",
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for item in roots:
        key = str(item.resolve()) if item.exists() else str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def scan_weight_candidates(search_roots: Iterable[Path]) -> list[WeightCandidate]:
    patterns = ("*yoloe*.pt", "*yolo-e*.pt")
    discovered: dict[str, WeightCandidate] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for candidate in root.rglob(pattern):
                if not candidate.is_file():
                    continue
                resolved = str(candidate.resolve()).lower()
                discovered[resolved] = WeightCandidate(path=candidate, size_bytes=int(candidate.stat().st_size))
    return sorted(discovered.values(), key=lambda item: (-item.size_bytes, str(item.path).lower()))


def evaluate_weight_staging(
    runtime_settings: MvpRuntimeSettings | None = None,
    *,
    search_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    settings = runtime_settings or MvpRuntimeSettings()
    roots = [Path(item).expanduser() for item in search_roots] if search_roots is not None else default_search_roots(settings)
    predict_signature = _predict_signature()
    payload: dict[str, Any] = {
        "python_executable": sys.executable,
        "configured_model_path": str(settings.task3_experimental_model_path) if settings.task3_experimental_model_path else None,
        "searched_roots": [str(item) for item in roots],
        "yoloe_class_present": YOLOE is not None,
        "predict_signature": predict_signature,
        "visual_prompt_supported": bool(
            "visual_prompts" in predict_signature and "refer_image" in predict_signature
        ),
    }

    if YOLOE is None or not payload["visual_prompt_supported"]:
        payload.update(
            {
                "status": STAGING_RUNTIME_INCOMPATIBLE,
                "reason": "yoloe_class_or_visual_prompt_signature_missing",
                "candidate_weights": [],
                "chosen_weight": None,
                "weight_load_attempts": [],
            }
        )
        return payload

    candidates = scan_weight_candidates(roots)
    payload["candidate_weights"] = [item.to_dict() for item in candidates]
    if not candidates:
        payload.update(
            {
                "status": STAGING_WEIGHT_MISSING,
                "reason": "no_local_yoloe_weight_found",
                "chosen_weight": None,
                "weight_load_attempts": [],
            }
        )
        return payload

    attempts: list[dict[str, Any]] = []
    for candidate in candidates:
        attempt = candidate.to_dict()
        try:
            model = YOLOE(str(candidate.path), task="detect")
            attempt["load_ok"] = True
            attempt["model_class"] = type(model).__name__
            payload.update(
                {
                    "status": STAGING_TRUE,
                    "reason": "real_local_yoloe_weight_loaded",
                    "chosen_weight": attempt,
                    "weight_load_attempts": attempts + [attempt],
                }
            )
            return payload
        except Exception as exc:  # pragma: no cover - weight varsa ortama bagli
            attempt["load_ok"] = False
            attempt["load_error"] = str(exc)
            attempts.append(attempt)

    payload.update(
        {
            "status": STAGING_WEIGHT_INCOMPATIBLE,
            "reason": "local_weight_found_but_load_failed",
            "chosen_weight": None,
            "weight_load_attempts": attempts,
        }
    )
    return payload


def render_weight_staging_markdown(payload: dict[str, Any]) -> str:
    chosen = payload.get("chosen_weight") or {}
    attempts = payload.get("weight_load_attempts", [])
    lines = [
        "# Task 3 Weight Staging Status",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Reason: `{payload.get('reason')}`",
        f"- Python: `{payload.get('python_executable')}`",
        f"- YOLOE class present: `{payload.get('yoloe_class_present')}`",
        f"- Visual prompt supported: `{payload.get('visual_prompt_supported')}`",
        f"- Configured model path: `{payload.get('configured_model_path')}`",
    ]
    if chosen:
        lines.extend(
            [
                f"- Chosen weight: `{chosen.get('filename')}`",
                f"- Chosen path: `{chosen.get('path')}`",
                f"- Model family: `{chosen.get('model_family')}`",
                f"- Expected input mode: `{chosen.get('expected_input_mode')}`",
            ]
        )
    else:
        lines.append("- Chosen weight: `None`")
    lines.extend(["", "## Search Roots"])
    lines.extend(f"- `{item}`" for item in payload.get("searched_roots", []))
    lines.extend(["", "## Candidate Weights"])
    candidate_weights = payload.get("candidate_weights", [])
    if not candidate_weights:
        lines.append("- No local YOLOE candidate weight found.")
    else:
        for item in candidate_weights:
            lines.append(f"- `{item.get('filename')}` | `{item.get('path')}` | `{item.get('size_bytes')}` bytes")
    if attempts:
        lines.extend(["", "## Load Attempts"])
        for item in attempts:
            if item.get("load_ok"):
                lines.append(f"- `{item.get('filename')}` loaded successfully.")
            else:
                lines.append(f"- `{item.get('filename')}` failed: `{item.get('load_error')}`")
    return "\n".join(lines) + "\n"


def _predict_signature() -> str:
    if YOLOE is None:
        return ""
    try:
        return str(inspect.signature(YOLOE.predict))
    except Exception:  # pragma: no cover - ortama bagli
        return ""
