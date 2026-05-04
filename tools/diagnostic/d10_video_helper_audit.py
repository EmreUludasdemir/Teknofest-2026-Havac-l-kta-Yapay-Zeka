from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-05-03_d10_repo_cleanup"
REPORT_PATH = ARCHIVE_DIR / "video_helper_usage.md"


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout


def _grep_usage(pattern: str) -> list[str]:
    matches: list[str] = []
    for relative in list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "tests").rglob("*.py")) + list((PROJECT_ROOT / "tools").rglob("*.py")):
        text = relative.read_text(encoding="utf-8")
        if pattern in text:
            matches.append(str(relative.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return sorted(matches)


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    history_text = _run_git("show", "34a91ad^:src/evaluation/task2_long_sequence.py")
    api_surface = []
    for name in ("iter_video_frames", "discover_task2_sources", "load_translation_csv", "Task2CsvRecord", "evaluate_task2_long_sequences"):
        if f"def {name}" in history_text or f"class {name}" in history_text:
            api_surface.append(name)

    evaluator_importers = _grep_usage("from src.evaluation.task2_long_sequence import iter_video_frames")
    all_iter_callers = _grep_usage("iter_video_frames(")

    markdown = [
        "# D10 Video Helper Usage",
        "",
        "## Migration Target",
        "",
        "- Deleted source reviewed from `34a91ad^:src/evaluation/task2_long_sequence.py`.",
        "- Generic helper to migrate: `iter_video_frames(video_path, frame_stride=1, limit=None, video_name='')`.",
        "- Keep Task 2-only API deleted: `Task2CsvRecord`, `discover_task2_sources`, `load_translation_csv`, `evaluate_records`, `evaluate_task2_long_sequences`.",
        "",
        "## Historical API Surface",
        "",
        f"- Symbols found in deleted module: {', '.join(api_surface)}",
        "",
        "## Direct Importers To Repoint",
        "",
    ]
    if evaluator_importers:
        markdown.extend(f"- `{item}`" for item in evaluator_importers)
    else:
        markdown.append("- None")

    markdown.extend(
        [
            "",
            "## All Current iter_video_frames Callers",
            "",
        ]
    )
    if all_iter_callers:
        markdown.extend(f"- `{item}`" for item in all_iter_callers)
    else:
        markdown.append("- None")

    markdown.extend(
        [
            "",
            "## Conclusion",
            "",
            "- Minimal runtime API surface is a single shared helper: `iter_video_frames`.",
            "- Task 3 runtime evaluators depend on it directly.",
            "- Additional Task 3 diagnostics/tools also depend on it and should be repointed to `src.core.video_io` to keep the cleanup branch runnable.",
            "- No Task 2-specific evaluation classes need to return.",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
