from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-05-03_d10_repo_cleanup"
)


@dataclass(slots=True)
class ItemRecord:
    label: str
    path: str
    category: str
    status: str
    file_count: int
    note: str = ""


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def _record_exact(path_str: str, *, category: str, label: str | None = None, note: str = "") -> ItemRecord:
    path = PROJECT_ROOT / path_str
    return ItemRecord(
        label=label or path.name,
        path=path_str,
        category=category,
        status="present" if path.exists() else "missing",
        file_count=_count_files(path),
        note=note,
    )


def _record_glob(pattern: str, *, category: str, label: str, note: str = "") -> list[ItemRecord]:
    matches = sorted(item for item in (PROJECT_ROOT / "tests").glob(pattern) if item.is_file())
    if not matches:
        return [ItemRecord(label=label, path=f"tests/{pattern}", category=category, status="missing", file_count=0, note=note)]
    return [
        ItemRecord(
            label=item.name,
            path=str(item.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            category=category,
            status="present",
            file_count=1,
            note=note,
        )
        for item in matches
    ]


def _sum_files(records: Iterable[ItemRecord]) -> int:
    return sum(int(record.file_count) for record in records if record.status == "present")


def _collect_removal_records() -> dict[str, list[ItemRecord]]:
    task1 = [
        _record_exact("src/task1", category="task1-remove", label="src/task1/"),
        _record_exact("src/exports", category="task1-remove", label="src/exports/"),
        _record_exact("src/evaluation/task1_onnx_validation.py", category="task1-remove"),
        _record_exact("src/evaluation/task1_trt_validation.py", category="task1-remove"),
        _record_exact("src/evaluation/export_readiness.py", category="task1-remove"),
        _record_exact("tools/run_phase7_export_validation.py", category="task1-remove"),
        _record_exact("tools/run_phase8_trt_validation.py", category="task1-remove"),
        _record_exact("tools/profiling_harness.py", category="task1-remove"),
        _record_exact("tools/run_runtime_smoke.py", category="task1-remove"),
        _record_exact("final_runtime", category="task1-remove", label="final_runtime/"),
        _record_exact("data/task1_combined.yaml", category="task1-remove"),
        _record_exact("data/task1_combined_human_focus.yaml", category="task1-remove"),
        _record_exact("data/task1_local_only.yaml", category="task1-remove"),
        _record_exact("data/task1_public_only.yaml", category="task1-remove"),
        _record_exact("data/task1_raw", category="task1-remove", label="data/task1_raw/"),
        _record_exact("data/task1_staging", category="task1-remove", label="data/task1_staging/"),
        _record_exact("data/task1_yolo", category="task1-remove", label="data/task1_yolo/"),
        _record_exact("data/HYZ_2025_Ornek_Veriler-001", category="task1-remove", label="data/HYZ_2025_Ornek_Veriler-001/"),
    ]
    task1.extend(_record_glob("test_task1_*.py", category="task1-remove", label="tests/test_task1_*.py"))
    task1.extend(
        [
            _record_exact("tests/test_export_readiness.py", category="task1-remove"),
            _record_exact("tests/test_runtime_package_smoke.py", category="task1-remove"),
            _record_exact("tests/test_profiling_harness.py", category="task1-remove"),
        ]
    )

    task2 = [
        _record_exact("src/task2", category="task2-remove", label="src/task2/"),
        _record_exact("src/evaluation/task2_long_sequence.py", category="task2-remove"),
        _record_exact("tools/run_phase5_reports.py", category="task2-remove"),
        _record_exact("tools/run_phase6_reports.py", category="task2-remove"),
        _record_exact("data/task2_health0_eval_manifest.json", category="task2-remove"),
    ]
    task2.extend(_record_glob("test_task2_*.py", category="task2-remove", label="tests/test_task2_*.py"))

    modify = [
        _record_exact("src/pipeline/mvp_processor.py", category="modify"),
        _record_exact("src/pipeline/orchestrator.py", category="modify"),
        _record_exact("src/config/settings.py", category="modify"),
        _record_exact("tests/test_mvp_processor.py", category="modify"),
        _record_exact("tests/test_task_scaffolds.py", category="modify"),
        _record_exact("tests/test_json_schema.py", category="modify"),
        _record_exact("tests/test_logger.py", category="modify"),
        _record_exact("README.md", category="modify", note="missing now; docs phase will be add/create instead of modify"),
        _record_exact("SETUP.md", category="modify"),
        _record_exact("agents.md", category="modify"),
    ]

    preserve = [
        _record_exact("src/task3", category="preserve", label="src/task3/"),
        _record_exact("src/server", category="preserve", label="src/server/"),
        _record_exact("src/data/validators.py", category="preserve"),
        _record_exact("src/core", category="preserve", label="src/core/"),
        _record_exact("src/config/settings.py", category="preserve", note="Task3 settings retained after cleanup"),
        _record_exact("src/tools/mock_server.py", category="preserve"),
        _record_exact("src/tools/runtime_package.py", category="preserve"),
        _record_exact("data/references/2026_baseline", category="preserve", label="data/references/2026_baseline/"),
        _record_exact("data/references/specs/2026_baseline_v3.json", category="preserve"),
        _record_exact("data/task3_eval_manifest.json", category="preserve"),
        _record_exact(
            "data/THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001",
            category="preserve",
            label="data/THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001/",
        ),
        _record_exact("tools/diagnostic", category="preserve", label="tools/diagnostic/"),
        _record_exact("_logs/reports_generated/task3_manifest", category="preserve", label="_logs/reports_generated/task3_manifest/"),
    ]
    preserve.extend(_record_glob("test_task3_*.py", category="preserve", label="tests/test_task3_*.py"))
    preserve.extend(
        [
            _record_exact("tests/test_contract.py", category="preserve"),
            _record_exact("tests/test_replay_runner.py", category="preserve"),
            _record_exact("tests/test_sequential_interface.py", category="preserve"),
            _record_exact("tests/test_sequential_mock_contract.py", category="preserve"),
            _record_exact("tests/test_sequential_warmup_retry.py", category="preserve"),
        ]
    )

    ambiguous = [
        _record_exact(
            "data/THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001/THYZ_2026_Ornek_Veri_Seti/Kamera_Kalibrasyon_Parametreleri_2026.txt",
            category="ambiguous",
            note="Task 3 runtime path currently does not reference this file directly; user decision needed before deletion.",
        ),
    ]

    extras = [
        _record_exact("runs", category="extra", label="runs/", note="Task1-style validation artefacts; not listed explicitly in prompt."),
        _record_exact("reports", category="extra", label="reports/", note="Contains mixed task reports; several task1/task2 docs are likely cleanup candidates."),
        _record_exact("ilk_modelim.pt", category="extra", note="Loose root weight; likely Task1-related or experiment artefact."),
        _record_exact("yolo11s.pt", category="extra", note="Referenced by task1 training support; likely removable in Task3-only repo."),
        _record_exact("yolo26n.pt", category="extra", note="Referenced by runtime/export/task1 tests; likely removable if Task1/export stack is removed."),
        _record_exact("yoloe-11m-seg.pt", category="extra", note="Loose duplicate root weight; Task3 code uses data/weights/task3/yoloe-11m-seg.pt."),
        _record_exact("data/weights/task3/yoloe-11m-seg.pt", category="extra", note="Keep: canonical Task3 YOLOE weight path."),
    ]
    return {
        "task1_remove": task1,
        "task2_remove": task2,
        "modify": modify,
        "preserve": preserve,
        "ambiguous": ambiguous,
        "extra_candidates": extras,
    }


def _render_table(records: list[ItemRecord]) -> list[str]:
    lines = [
        "| Item | Path | Status | Files | Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record.label} | `{record.path}` | `{record.status}` | {record.file_count} | {record.note or '-'} |"
        )
    return lines


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = _collect_removal_records()

    summary = {
        key: {
            "items": len(value),
            "present_items": sum(1 for item in value if item.status == "present"),
            "missing_items": sum(1 for item in value if item.status == "missing"),
            "present_file_count": _sum_files(value),
        }
        for key, value in groups.items()
    }

    payload = {
        "summary": summary,
        "groups": {key: [asdict(item) for item in value] for key, value in groups.items()},
    }
    (OUTPUT_DIR / "removal_inventory.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = [
        "# D10 Removal Inventory",
        "",
        "## Gate",
        "",
        "- This is a read-only inventory pass. No deletion, rename, branch creation, or pipeline cleanup was performed.",
        "- Stop condition remains active after this report: do not start Phase 2 until the user confirms ambiguous items.",
        "",
        "## Summary",
        "",
        f"- Task 1 explicit removal scope currently resolves to `{summary['task1_remove']['present_items']}` present entries and `{summary['task1_remove']['present_file_count']}` files.",
        f"- Task 2 explicit removal scope currently resolves to `{summary['task2_remove']['present_items']}` present entries and `{summary['task2_remove']['present_file_count']}` files.",
        f"- Modify list has `{summary['modify']['present_items']}` present files and `{summary['modify']['missing_items']}` missing targets. `README.md` is missing today, so docs phase would add/create it.",
        f"- Preserve list resolves to `{summary['preserve']['present_items']}` present entries and `{summary['preserve']['present_file_count']}` files.",
        f"- Ambiguous list has `{summary['ambiguous']['present_items']}` present item requiring user decision.",
        f"- Extra cleanup candidates outside the prompt were found in `runs/`, `reports/`, and loose root model weights.",
        "",
        "## Task 1 Removal",
        "",
    ]
    lines.extend(_render_table(groups["task1_remove"]))
    lines.extend(
        [
            "",
            "## Task 2 Removal",
            "",
        ]
    )
    lines.extend(_render_table(groups["task2_remove"]))
    lines.extend(
        [
            "",
            "## Modify In Place",
            "",
        ]
    )
    lines.extend(_render_table(groups["modify"]))
    lines.extend(
        [
            "",
            "## Preserve",
            "",
        ]
    )
    lines.extend(_render_table(groups["preserve"]))
    lines.extend(
        [
            "",
            "## Ambiguous",
            "",
        ]
    )
    lines.extend(_render_table(groups["ambiguous"]))
    lines.extend(
        [
            "",
            "## Extra Candidates",
            "",
            "- These are not in the explicit prompt list, but they look relevant to a Task3-only cleanup decision and should be reviewed before Phase 2.",
            "",
        ]
    )
    lines.extend(_render_table(groups["extra_candidates"]))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `tests/test_task1_*.py` expands to 13 files in this repo, not 8.",
            "- `tests/test_task2_*.py` expands to 7 files in this repo, not 5.",
            "- `data/HYZ_2025_Ornek_Veriler-001/` exists and includes Task2 calibration text files plus absent-target proxy videos currently used in Task3 diagnostics; deleting it will require either retiring those diagnostics or relocating that data.",
            "- `data/THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001/THYZ_2026_Ornek_Veri_Seti/Kamera_Kalibrasyon_Parametreleri_2026.txt` is the actual ambiguous calibration file path. The shorter root-level path in the prompt does not exist.",
            "",
        ]
    )

    (OUTPUT_DIR / "removal_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote removal inventory to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
