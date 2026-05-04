from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-05-03_d10_repo_cleanup"
)
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(slots=True)
class AuditRecord:
    name: str
    classification: str
    decision: str
    rationale: str


def _classify(path: Path) -> AuditRecord:
    name = path.name
    lowered = name.lower()
    if lowered.startswith("task3_"):
        return AuditRecord(name=name, classification="task3", decision="keep", rationale="Task 3 specific report.")
    if lowered.startswith("task1_"):
        return AuditRecord(name=name, classification="task1", decision="remove", rationale="Task 1 specific report.")
    if lowered.startswith("task2_"):
        return AuditRecord(name=name, classification="task2", decision="remove", rationale="Task 2 specific report.")
    if lowered.startswith("phase11_") or lowered.startswith("competition_"):
        return AuditRecord(
            name=name,
            classification="mixed",
            decision="remove",
            rationale="Cross-task or stale readiness document; not a Task3-only artefact.",
        )
    return AuditRecord(
        name=name,
        classification="unknown",
        decision="remove",
        rationale="Not explicitly Task3-scoped; remove under Task3-only repo policy.",
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [_classify(path) for path in sorted(REPORTS_DIR.glob("*")) if path.is_file()]
    payload = {"records": [asdict(record) for record in records]}
    (OUTPUT_DIR / "reports_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    keep_count = sum(1 for record in records if record.decision == "keep")
    remove_count = sum(1 for record in records if record.decision == "remove")
    lines = [
        "# D10 Reports Audit",
        "",
        f"- Reports scanned: `{len(records)}`",
        f"- Keep count: `{keep_count}`",
        f"- Remove count: `{remove_count}`",
        "",
        "| File | Classification | Decision | Rationale |",
        "| --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record.name} | `{record.classification}` | `{record.decision}` | {record.rationale} |"
        )
    (OUTPUT_DIR / "reports_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote reports audit to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
