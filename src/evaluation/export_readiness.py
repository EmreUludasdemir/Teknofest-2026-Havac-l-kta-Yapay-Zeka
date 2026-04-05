from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_export_readiness(
    *,
    reports_dir: str | Path = "reports",
    tests_ok: bool = True,
) -> dict[str, Any]:
    base = Path(reports_dir)
    blockers: list[str] = []

    task2_before = _load_json(base / "task2_long_sequence_before_summary.json")
    task2_after = _load_json(base / "task2_long_sequence_after_summary.json")
    task3_orb = _load_json(base / "task3_baseline_orb_summary.json")
    task3_learned = _load_json(base / "task3_baseline_learned_summary.json")
    task1_gpu = _load_json(base / "profiling" / "task1_real_profile_gpu.json")
    task1_cpu = _load_json(base / "profiling" / "task1_real_profile_cpu.json")

    task2_ready = False
    if task2_before and task2_after:
        before_aggregate = task2_before.get("aggregate", {})
        after_aggregate = task2_after.get("aggregate", {})
        before_drift = float(before_aggregate.get("health0_drift_accumulation", 0.0))
        after_drift = float(after_aggregate.get("health0_drift_accumulation", 0.0))
        before_fallback = float(before_aggregate.get("fallback_rate", 0.0))
        after_fallback = float(after_aggregate.get("fallback_rate", 0.0))
        before_recovery = float(before_aggregate.get("recovery_error_after_health_returns_to_1", 0.0))
        after_recovery = float(after_aggregate.get("recovery_error_after_health_returns_to_1", 0.0))
        task2_ready = (
            after_drift <= before_drift * 0.8
            and after_fallback <= before_fallback
            and after_recovery <= before_recovery
        )
        if not task2_ready:
            blockers.append("Task 2 drift hedefi saglanmadi.")
    else:
        blockers.append("Task 2 before/after raporlari eksik.")

    task1_ready = False
    if task1_gpu:
        gpu_results = task1_gpu.get("results", [])
        gpu_viable = [
            item
            for item in gpu_results
            if item.get("available")
            and item.get("diagnostics", {}).get("last_output", {}).get("runtime_device") not in {"cpu", "-", None}
            and float(item.get("peak_vram_mb") or 0.0) < 6500.0
        ]
        task1_ready = bool(gpu_viable)
        if not task1_ready:
            blockers.append("Task 1 GPU profiling dogrulanmadi veya VRAM siniri asildi.")
        if task1_gpu.get("recommendation", {}).get("model_fallback_candidate") is None:
            blockers.append("Task 1 ikinci gercek model adayi eksik.")
    else:
        blockers.append("Task 1 GPU profiling raporu eksik.")

    task3_ready = False
    if task3_orb and task3_learned:
        decision = _decide_task3_default(task3_orb, task3_learned)
        task3_ready = decision in {"orb_template", "learned_descriptor"}
        if not task3_ready:
            blockers.append("Task 3 learned descriptor karari net degil.")
    else:
        blockers.append("Task 3 ORB/learned comparison raporlari eksik.")

    if not tests_ok:
        blockers.append("Test suite yesil degil.")

    ready = task2_ready and task1_ready and task3_ready and tests_ok
    payload = {
        "ready": ready,
        "task2_ready": task2_ready,
        "task1_ready": task1_ready,
        "task3_ready": task3_ready,
        "tests_ok": tests_ok,
        "blockers": blockers,
    }
    (base / "export_readiness.md").write_text(render_export_readiness(payload), encoding="utf-8")
    return payload


def render_export_readiness(payload: dict[str, Any]) -> str:
    lines = [
        f"Export Readiness: {'EVET' if payload.get('ready') else 'HAYIR'}",
        "",
        f"- Task 2 ready: {payload.get('task2_ready')}",
        f"- Task 1 ready: {payload.get('task1_ready')}",
        f"- Task 3 ready: {payload.get('task3_ready')}",
        f"- Tests ok: {payload.get('tests_ok')}",
        "",
        "Blokajlar:",
    ]
    blockers = payload.get("blockers", [])
    if not blockers:
        lines.append("- Yok. ONNX hazirligina gecilebilir.")
    else:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    return "\n".join(lines) + "\n"


def _decide_task3_default(orb_payload: dict[str, Any], learned_payload: dict[str, Any]) -> str:
    orb_aggregate = orb_payload.get("aggregate", {})
    learned_aggregate = learned_payload.get("aggregate", {})
    orb_fp = int(orb_aggregate.get("false_positive_proxy_count", 0))
    learned_fp = int(learned_aggregate.get("false_positive_proxy_count", 0))
    orb_no_match = float(orb_aggregate.get("no_match_suppression_rate", 0.0))
    learned_no_match = float(learned_aggregate.get("no_match_suppression_rate", 0.0))
    orb_accept = int(orb_aggregate.get("accepted_match_count", 0))
    learned_accept = int(learned_aggregate.get("accepted_match_count", 0))

    if learned_fp <= int(round(orb_fp * 0.8)) and (learned_no_match - orb_no_match) <= 0.10 and learned_accept >= int(round(orb_accept * 0.85)):
        return "learned_descriptor"
    if orb_accept > 0:
        return "orb_template"
    return "belirsiz"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
