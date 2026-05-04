from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task3.reference_cache import ReferenceCache

REFERENCE_DIR = PROJECT_ROOT / "data" / "references" / "2026_baseline"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "_logs"
    / "reports_generated"
    / "task3_manifest"
    / "2026-04-27_d5_mixed_bank_integration"
)

REFERENCE_FILES = {
    "ref_01": "Referans_Nesne_01.JPG",
    "ref_02": "Referans_Nesne_02.JPG",
    "ref_03": "Referans_Nesne_03.JPG",
    "ref_04": "Referans_Nesne_04.JPG",
    "ref_05": "Referans_Nesne_05.jpg",
    "ref_06": "Referans_Nesne_06.jpg",
    "ref_07": "Referans_Nesne_07.png",
    "ref_08": "Referans_Nesne_08.png",
    "ref_09": "Referans_Nesne_09.png",
    "ref_10": "Referans_Nesne_10.png",
    "ref_11": "Referans_Nesne_11.png",
    "ref_12": "Referans_Nesne_12.png",
}

EXPECTED_MODALITIES = {
    "ref_01": "rgb",
    "ref_02": "rgb",
    "ref_03": "rgb",
    "ref_04": "thermal",
    "ref_05": "rgb",
    "ref_06": "rgb",
    "ref_07": "rgb",
    "ref_08": "rgb",
    "ref_09": "rgb",
    "ref_10": "rgb",
    "ref_11": "thermal",
    "ref_12": "thermal",
}


@dataclass(slots=True)
class AuditRow:
    reference_id: str
    source_file: str
    expected_modality: str
    detected_modality: str
    detector: str
    detector_modalities: list[str]
    confidence: str
    rationale: str
    exif_present: bool
    pixel_analysis_used: bool
    modality_method: str
    status: str
    decision: str


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_audit_manifest(temp_dir: Path) -> Path:
    manifest_path = temp_dir / "manifest.json"
    spec_path = temp_dir / "audit_spec.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "d5_audit_v1",
                "spec_path": spec_path.name,
                "reference_count": len(REFERENCE_FILES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    spec_payload = {
        "version": "d5_audit_v1",
        "created": "2026-05-02",
        "source": "temporary routing audit mirror for official 12-ref mixed bank",
        "references": {
            reference_id: {
                "file": source_file,
                "modality": EXPECTED_MODALITIES[reference_id],
                "source_exif": None,
            }
            for reference_id, source_file in REFERENCE_FILES.items()
        },
        "overrides": {},
    }
    spec_path.write_text(json.dumps(spec_payload, indent=2), encoding="utf-8")
    return manifest_path


def _build_audit_mirror() -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="task3_d5_routing_audit_", dir=str(PROJECT_ROOT / "_logs")))
    for source_file in REFERENCE_FILES.values():
        source_path = REFERENCE_DIR / source_file
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        _link_or_copy(source_path, temp_root / source_file)
    _write_audit_manifest(temp_root)
    return temp_root


def _has_exif(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            return bool(exif)
    except Exception:
        return False


def _evaluate_rgb_expectation(detected_modality: str, routed_modalities: list[str]) -> tuple[str, str]:
    if detected_modality == "rgb" and routed_modalities == ["rgb"]:
        return "pass", "RGB inspection matched rgb-only routing."
    if detected_modality == "unknown" and routed_modalities == ["rgb"]:
        return "pass", "RGB inspection fell back to rgb-only routing without opening the thermal path."
    if detected_modality == "thermal":
        return "fail", "RGB reference was misclassified as thermal."
    if "thermal" in routed_modalities:
        return "fail", "RGB reference opened the thermal route instead of rgb-only routing."
    return "fail", "RGB reference did not resolve to rgb or rgb-only fallback."


def _evaluate_thermal_expectation(detected_modality: str, routed_modalities: list[str]) -> tuple[str, str]:
    if detected_modality == "thermal" and routed_modalities == ["thermal"]:
        return "pass", "Thermal inspection matched thermal-only routing."
    if detected_modality != "thermal":
        return "fail", "Thermal reference was not classified as thermal."
    if "rgb" in routed_modalities:
        return "fail", "Thermal reference still opened the rgb route."
    return "fail", "Thermal reference did not resolve to thermal-only routing."


def _collect_rows(cache: ReferenceCache) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for reference_id in sorted(REFERENCE_FILES):
        item = cache.get(reference_id)
        if item is None:
            raise KeyError(f"Missing cached reference: {reference_id}")
        metadata = dict(item.get("reference_metadata") or {})
        routing = dict(metadata.get("routing_signals") or {})
        summary = cache.get_auto_routing_summary().get(reference_id, {})
        source_path = REFERENCE_DIR / REFERENCE_FILES[reference_id]
        detected_modality = str(summary.get("modality") or metadata.get("modality") or "unknown")
        detector = str(summary.get("detector") or metadata.get("detector") or "yoloe")
        detector_modalities = list(summary.get("detector_modalities") or metadata.get("detector_modalities") or [])
        expected_modality = EXPECTED_MODALITIES[reference_id]
        if expected_modality == "thermal":
            status, decision = _evaluate_thermal_expectation(detected_modality, detector_modalities)
        else:
            status, decision = _evaluate_rgb_expectation(detected_modality, detector_modalities)
        rows.append(
            AuditRow(
                reference_id=reference_id,
                source_file=REFERENCE_FILES[reference_id],
                expected_modality=expected_modality,
                detected_modality=detected_modality,
                detector=detector,
                detector_modalities=detector_modalities,
                confidence=str(summary.get("confidence") or metadata.get("routing_confidence") or "low"),
                rationale=str(summary.get("rationale") or metadata.get("routing_rationale") or ""),
                exif_present=_has_exif(source_path),
                pixel_analysis_used=str(routing.get("modality_method") or "none") in {"pixel", "both"},
                modality_method=str(routing.get("modality_method") or "none"),
                status=status,
                decision=decision,
            )
        )
    return rows


def _render_markdown(rows: list[AuditRow], *, native_preload_loaded: int, mirror_dir: Path) -> str:
    failures = [row for row in rows if row.status != "pass"]
    lines = [
        "# D5 Routing Audit",
        "",
        "## Gate",
        "",
        f"- Native `ReferenceCache.preload_from_directory(\"{REFERENCE_DIR.relative_to(PROJECT_ROOT)}\")` loaded `{native_preload_loaded}` refs with specless default routing because `manifest.json` is absent in the production bank.",
        f"- Audit auto-routing was therefore measured on a temporary mirror with generated manifest/spec at `{mirror_dir}`; production source files were not modified.",
        f"- Audit status: `{'FAIL' if failures else 'PASS'}`",
    ]
    if failures:
        lines.append("- Stop condition triggered: do not continue to manifest/spec integration or mixed-bank validation until routing heuristics are corrected.")
    else:
        lines.append("- No blocking routing mismatch observed; safe to continue to Step 2.")

    lines.extend(
        [
            "",
            "## Table",
            "",
            "| Ref | Source File | Expected | Detected | Detector | Route Modalities | Confidence | EXIF | Pixel | Method | Status | Decision |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.reference_id} | {row.source_file} | {row.expected_modality} | {row.detected_modality} | "
            f"{row.detector} | {','.join(row.detector_modalities) or '-'} | {row.confidence} | "
            f"{'yes' if row.exif_present else 'no'} | {'yes' if row.pixel_analysis_used else 'no'} | {row.modality_method} | "
            f"{row.status} | {row.decision} |"
        )

    lines.extend(["", "## Findings", ""])
    if failures:
        for row in failures:
            lines.append(
                f"- `{row.reference_id}`: expected `{row.expected_modality}`, got `{row.detected_modality}` with "
                f"`{row.detector}({','.join(row.detector_modalities) or '-'})`. {row.decision} Rationale: {row.rationale}"
            )
    else:
        lines.append("- All 12 references matched the expected mixed-bank routing behavior.")

    lines.extend(["", "## Summary", ""])
    thermal_count = sum(1 for row in rows if row.detected_modality == "thermal")
    rgb_count = sum(1 for row in rows if row.detected_modality == "rgb")
    unknown_count = sum(1 for row in rows if row.detected_modality == "unknown")
    lines.extend(
        [
            f"- Detected modality counts: rgb=`{rgb_count}`, thermal=`{thermal_count}`, unknown=`{unknown_count}`",
            f"- Pass count: `{len(rows) - len(failures)}/{len(rows)}`",
            f"- Fail count: `{len(failures)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json(rows: list[AuditRow], *, native_preload_loaded: int, audit_status: str, mirror_dir: Path) -> None:
    payload = {
        "audit_status": audit_status,
        "native_preload_loaded": native_preload_loaded,
        "source_reference_dir": str(REFERENCE_DIR),
        "temporary_mirror_dir": str(mirror_dir),
        "rows": [
            {
                "reference_id": row.reference_id,
                "source_file": row.source_file,
                "expected_modality": row.expected_modality,
                "detected_modality": row.detected_modality,
                "detector": row.detector,
                "detector_modalities": row.detector_modalities,
                "confidence": row.confidence,
                "rationale": row.rationale,
                "exif_present": row.exif_present,
                "pixel_analysis_used": row.pixel_analysis_used,
                "modality_method": row.modality_method,
                "status": row.status,
                "decision": row.decision,
            }
            for row in rows
        ],
    }
    (OUTPUT_DIR / "routing_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    native_cache = ReferenceCache()
    native_preload_loaded = native_cache.preload_from_directory(REFERENCE_DIR)

    mirror_dir = _build_audit_mirror()
    try:
        cache = ReferenceCache()
        loaded = cache.preload_from_directory(mirror_dir)
        if loaded != len(REFERENCE_FILES):
            raise RuntimeError(f"Audit mirror loaded {loaded} refs, expected {len(REFERENCE_FILES)}")
        rows = _collect_rows(cache)
        failures = [row for row in rows if row.status != "pass"]
        audit_status = "fail" if failures else "pass"
        markdown = _render_markdown(rows, native_preload_loaded=native_preload_loaded, mirror_dir=mirror_dir)
        (OUTPUT_DIR / "routing_audit.md").write_text(markdown, encoding="utf-8")
        _write_json(rows, native_preload_loaded=native_preload_loaded, audit_status=audit_status, mirror_dir=mirror_dir)
    finally:
        shutil.rmtree(mirror_dir, ignore_errors=True)

    print(f"Routing audit written to {OUTPUT_DIR}")
    return 2 if audit_status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
