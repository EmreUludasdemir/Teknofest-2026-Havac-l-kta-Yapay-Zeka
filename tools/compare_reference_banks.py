from __future__ import annotations

"""Pilot utility: compare the staged 2026 baseline reference bank to official references."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_baseline_dir() -> Path:
    return PROJECT_ROOT / "data" / "references" / "2026_baseline"


def _default_official_dir() -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001"
        / "THYZ_2026_Ornek_Veri_Seti"
        / "THYZ_2026_Ornek_Veri_1_Referans_Nesneler"
    )


def _default_output_dir() -> Path:
    return PROJECT_ROOT / "_logs" / "reference_bank_comparison"


@dataclass(slots=True)
class ReferenceComparisonResult:
    reference_id: str
    baseline_path: str | None
    official_path: str | None
    baseline_hw: list[int] | None
    official_hw: list[int] | None
    ssim: float | None
    mean_abs_diff: float | None
    classification: str
    note: str | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare baseline and official 2026 Task 3 reference banks")
    parser.add_argument("--baseline-dir", default=str(_default_baseline_dir()))
    parser.add_argument("--official-dir", default=str(_default_official_dir()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    args = parser.parse_args(argv)

    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    baseline_dir = Path(args.baseline_dir)
    official_dir = Path(args.official_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ReferenceComparisonResult] = []
    baseline_ids_seen: set[str] = set()
    official_ids_seen: set[str] = set()
    for index in range(1, 7):
        reference_id = f"ref_{index:02d}"
        baseline_path = _resolve_baseline_path(baseline_dir, reference_id)
        official_path = _resolve_official_path(official_dir, index)
        if baseline_path is not None:
            baseline_ids_seen.add(reference_id)
        if official_path is not None:
            official_ids_seen.add(reference_id)

        if baseline_path is None and official_path is None:
            continue
        if baseline_path is None and official_path is not None:
            official_image = _load_grayscale(cv2, official_path)
            results.append(
                ReferenceComparisonResult(
                    reference_id=reference_id,
                    baseline_path=None,
                    official_path=str(official_path),
                    baseline_hw=None,
                    official_hw=list(official_image.shape[:2]),
                    ssim=None,
                    mean_abs_diff=None,
                    classification="MISSING",
                    note="MISSING_FROM_BASELINE",
                )
            )
            continue
        if baseline_path is not None and official_path is None:
            baseline_image = _load_grayscale(cv2, baseline_path)
            results.append(
                ReferenceComparisonResult(
                    reference_id=reference_id,
                    baseline_path=str(baseline_path),
                    official_path=None,
                    baseline_hw=list(baseline_image.shape[:2]),
                    official_hw=None,
                    ssim=None,
                    mean_abs_diff=None,
                    classification="EXTRA",
                    note="EXTRA_IN_BASELINE",
                )
            )
            continue

        assert baseline_path is not None
        assert official_path is not None
        baseline_image = _load_grayscale(cv2, baseline_path)
        official_image = _load_grayscale(cv2, official_path)
        baseline_hw = list(baseline_image.shape[:2])
        official_hw = list(official_image.shape[:2])
        aligned_baseline, aligned_official = _align_to_common_shape(cv2, baseline_image, official_image)
        ssim = _compute_ssim(cv2, np, aligned_baseline, aligned_official)
        mean_abs_diff = float(np.mean(np.abs(aligned_baseline.astype(np.float32) - aligned_official.astype(np.float32))))
        classification = _classify_pair(ssim, mean_abs_diff)
        results.append(
            ReferenceComparisonResult(
                reference_id=reference_id,
                baseline_path=str(baseline_path),
                official_path=str(official_path),
                baseline_hw=baseline_hw,
                official_hw=official_hw,
                ssim=round(float(ssim), 6),
                mean_abs_diff=round(mean_abs_diff, 6),
                classification=classification,
            )
        )

    extra_candidates = sorted(
        item for item in baseline_dir.glob("ref_*.*") if item.is_file() and _extract_reference_id(item.name) not in official_ids_seen
    )
    for path in extra_candidates:
        reference_id = _extract_reference_id(path.name)
        if not reference_id:
            continue
        baseline_image = _load_grayscale(cv2, path)
        results.append(
            ReferenceComparisonResult(
                reference_id=reference_id,
                baseline_path=str(path),
                official_path=None,
                baseline_hw=list(baseline_image.shape[:2]),
                official_hw=None,
                ssim=None,
                mean_abs_diff=None,
                classification="EXTRA",
                note="EXTRA_IN_BASELINE",
            )
        )

    summary = _summarize_results(results)
    payload = {
        "baseline_dir": str(baseline_dir),
        "official_dir": str(official_dir),
        "results": [asdict(item) for item in results],
        "summary": summary,
    }
    (output_dir / "reference_bank_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "reference_bank_comparison.md").write_text(_render_markdown(results, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _resolve_baseline_path(baseline_dir: Path, reference_id: str) -> Path | None:
    for suffix in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        candidate = baseline_dir / f"{reference_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _resolve_official_path(official_dir: Path, index: int) -> Path | None:
    stem = f"Referans_Nesne_{index:02d}"
    for suffix in (".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"):
        candidate = official_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_grayscale(cv2: Any, path: Path) -> Any:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed_to_load_image:{path}")
    return image


def _align_to_common_shape(cv2: Any, first: Any, second: Any) -> tuple[Any, Any]:
    first_h, first_w = first.shape[:2]
    second_h, second_w = second.shape[:2]
    target_h = min(first_h, second_h)
    target_w = min(first_w, second_w)
    if (first_h, first_w) != (target_h, target_w):
        first = cv2.resize(first, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if (second_h, second_w) != (target_h, target_w):
        second = cv2.resize(second, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return first, second


def _compute_ssim(cv2: Any, np: Any, first: Any, second: Any) -> float:
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = (11, 11)
    sigma = 1.5
    mu1 = cv2.GaussianBlur(first, kernel, sigma)
    mu2 = cv2.GaussianBlur(second, kernel, sigma)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(first * first, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(second * second, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(first * second, kernel, sigma) - mu1_mu2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(ssim_map))


def _classify_pair(ssim: float, mean_abs_diff: float) -> str:
    if ssim > 0.98 and mean_abs_diff < 5.0:
        return "IDENTICAL"
    if ssim > 0.90 and mean_abs_diff < 15.0:
        return "CLOSE"
    if 0.70 <= ssim <= 0.90:
        return "DRIFT"
    return "DIFFERENT"


def _extract_reference_id(name: str) -> str | None:
    lower = name.lower()
    if not lower.startswith("ref_"):
        return None
    stem = Path(lower).stem
    return stem if len(stem) == 6 else None


def _summarize_results(results: list[ReferenceComparisonResult]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in results:
        counts[item.classification] = counts.get(item.classification, 0) + 1
    good_count = counts.get("IDENTICAL", 0) + counts.get("CLOSE", 0)
    drift_count = counts.get("DRIFT", 0)
    missing_count = counts.get("MISSING", 0)
    gate_pass = (
        good_count >= 5
        and counts.get("DIFFERENT", 0) == 0
        and drift_count <= 1
        and missing_count <= 1
    )
    return {
        "classification_counts": counts,
        "gate_decision": "PASS" if gate_pass else "FAIL",
        "gate_reason": _build_gate_reason(counts, good_count=good_count),
    }


def _build_gate_reason(counts: dict[str, int], *, good_count: int) -> str:
    if counts.get("DIFFERENT", 0) > 0:
        return "Any DIFFERENT reference fails the gate"
    if counts.get("DRIFT", 0) > 1:
        return "More than one DRIFT reference fails the gate"
    if counts.get("MISSING", 0) > 1:
        return "More than one MISSING reference fails the gate"
    if good_count < 5:
        return "Fewer than five IDENTICAL/CLOSE references fails the gate"
    return "At least five references are IDENTICAL/CLOSE with no blocking divergence"


def _render_markdown(results: list[ReferenceComparisonResult], summary: dict[str, Any]) -> str:
    lines = [
        "| Reference | Baseline HW | Official HW | SSIM | Mean Abs Diff | Classification | Note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(results, key=lambda item: item.reference_id)
    for item in ordered:
        lines.append(
            "| {reference_id} | {baseline_hw} | {official_hw} | {ssim} | {mean_abs_diff} | {classification} | {note} |".format(
                reference_id=item.reference_id,
                baseline_hw=_format_hw(item.baseline_hw),
                official_hw=_format_hw(item.official_hw),
                ssim="-" if item.ssim is None else f"{item.ssim:.6f}",
                mean_abs_diff="-" if item.mean_abs_diff is None else f"{item.mean_abs_diff:.6f}",
                classification=item.classification,
                note=item.note or "-",
            )
        )
    lines.append("")
    lines.append(f"Gate decision: {summary['gate_decision']}")
    lines.append(summary["gate_reason"])
    return "\n".join(lines) + "\n"


def _format_hw(value: list[int] | None) -> str:
    if value is None:
        return "-"
    return f"{value[0]}x{value[1]}"


if __name__ == "__main__":
    raise SystemExit(main())
