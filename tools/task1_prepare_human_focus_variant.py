from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.experimental import (
    TARGET_CLASS_NAMES,
    TASK1_YOLO_ROOT,
    build_human_focus_dataset,
    collect_human_focus_audit,
    render_human_focus_audit_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a local-human-focused Task 1 dataset variant.")
    parser.add_argument("--base-root", default=str(TASK1_YOLO_ROOT / "combined"))
    parser.add_argument("--output-root", default=str(TASK1_YOLO_ROOT / "combined_human_focus"))
    parser.add_argument("--yaml-path", default=str(PROJECT_ROOT / "data" / "task1_combined_human_focus.yaml"))
    parser.add_argument("--repeat-factor", type=int, default=6)
    parser.add_argument("--tiny-threshold", type=float, default=0.001)
    args = parser.parse_args(argv)

    audit = collect_human_focus_audit(args.base_root)
    variant = build_human_focus_dataset(
        args.base_root,
        args.output_root,
        args.yaml_path,
        class_names=list(TARGET_CLASS_NAMES),
        local_human_repeat_factor=args.repeat_factor,
        tiny_human_threshold=args.tiny_threshold,
    )
    reports_root = PROJECT_ROOT / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "task1_human_focus_audit.json").write_text(
        json.dumps({"audit": audit, "variant": variant}, indent=2),
        encoding="utf-8",
    )
    (reports_root / "task1_human_focus_audit.md").write_text(
        render_human_focus_audit_markdown(audit),
        encoding="utf-8",
    )
    print(json.dumps({"audit": audit, "variant": variant}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
