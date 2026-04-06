from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.mock_server import OfficialRepoMockServer
from src.tools.runtime_package import load_runtime_bootstrap
from tools.run_competition_runtime import main as competition_runtime_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 10 operator drill")
    parser.add_argument("--config", default="final_runtime/config/runtime.toml")
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--output-json", default="final_runtime/logs/operator_drill_summary.json")
    args = parser.parse_args(argv)

    bootstrap = load_runtime_bootstrap(
        args.config,
        production=True,
        base_url_override=args.base_url,
        username_override=args.username,
        password_override=args.password,
    )
    log_dir = Path(bootstrap.runtime_config.get("paths", {}).get("log_dir", "final_runtime/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_path = log_dir / "runtime.jsonl"
    existing_log_size = runtime_log_path.stat().st_size if runtime_log_path.exists() else 0

    with maybe_mock_server(args.base_url) as server_base_url:
        runtime_args = [
            "--config",
            str(args.config),
            "--max-frames",
            str(args.max_frames),
        ]
        if server_base_url:
            runtime_args.extend(["--base-url", server_base_url])
        if args.username:
            runtime_args.extend(["--username", args.username])
        if args.password:
            runtime_args.extend(["--password", args.password])
        runtime_rc = competition_runtime_main(runtime_args)

    summary_path = log_dir / "competition_runtime_summary.json"
    runtime_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    runtime_events = read_appended_runtime_events(runtime_log_path, existing_log_size)
    event_names = [str(item.get("event")) for item in runtime_events]
    manifest = bootstrap.manifest
    required_artifacts = list(manifest.get("required", []))
    missing_required = [
        item.get("artifact_id")
        for item in required_artifacts
        if not item.get("path") or not Path(str(item.get("path"))).exists()
    ]

    warnings: list[str] = []
    active_stage = str(runtime_summary.get("active_task1_stage") or "")
    if active_stage in {"synthetic", "ultralytics:yolo11n"}:
        warnings.append(f"active_stage_requires_attention:{active_stage}")
    if bootstrap.sequential_settings.wire_profile != "official_current":
        warnings.append(f"non_default_wire_profile:{bootstrap.sequential_settings.wire_profile}")

    payload = {
        "python_executable": sys.executable,
        "config_path": str(args.config),
        "runtime_return_code": runtime_rc,
        "runtime_mode": bootstrap.runtime_config.get("runtime", {}).get("mode"),
        "wire_profile": bootstrap.sequential_settings.wire_profile,
        "required_artifacts_missing": missing_required,
        "competition_summary": runtime_summary,
        "expected_events_seen": {
            "warmup_policy_started": "warmup_policy_started" in event_names,
            "warmup_stage_completed": "warmup_stage_completed" in event_names,
            "sequential_prediction_sent": "sequential_prediction_sent" in event_names,
        },
        "warnings": warnings,
        "accepted": runtime_rc == 0 and not missing_required,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if bool(payload["accepted"]) else 1


def read_appended_runtime_events(path: Path, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(offset)
        content = handle.read().decode("utf-8")
    events: list[dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


@contextmanager
def maybe_mock_server(base_url: str | None):
    if base_url:
        yield base_url
        return
    server = OfficialRepoMockServer(mode="sequential", sequential_warmup_delay_s=0.05)
    server.start()
    try:
        yield server.base_url
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
