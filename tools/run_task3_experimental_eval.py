from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MvpRuntimeSettings
from src.evaluation.task3_experimental import (
    apply_task3_settings_overrides,
    compare_task3_baseline_vs_experimental,
    evaluate_task3_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3 baseline vs experimental evaluation harness")
    parser.add_argument("--mode", choices=["baseline", "experimental", "compare"], default="compare")
    parser.add_argument("--manifest", default="data/task3_eval_manifest.json")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--config", help="Opsiyonel JSON override dosyasi")
    return parser.parse_args()


def load_overrides(path: str | None) -> dict[str, object] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    settings = apply_task3_settings_overrides(MvpRuntimeSettings(), load_overrides(args.config))
    if args.mode == "compare":
        payload = compare_task3_baseline_vs_experimental(
            manifest_path=args.manifest,
            runtime_settings=settings,
            output_dir=args.output_dir,
        )
        print(json.dumps(payload["comparison"], indent=2))
        return 0

    payload = evaluate_task3_mode(
        mode=args.mode,
        manifest_path=args.manifest,
        runtime_settings=settings,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
