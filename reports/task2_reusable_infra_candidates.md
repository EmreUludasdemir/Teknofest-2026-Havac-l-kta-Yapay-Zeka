# Task 2 Reusable Infrastructure Candidates

**Date:** 2026-04-07  
**Purpose:** Identify cherry-pickable items that are neutral evaluation/reporting infrastructure

## Safe to Cherry-Pick

These items do not alter production estimator behavior and provide useful evaluation capabilities:

### Already on Main (No Action Needed)

| File | Purpose | Status |
|------|---------|--------|
| `src/evaluation/task2_long_sequence.py` | Long-sequence replay evaluation harness | ✅ Already on main |
| `tests/test_task2_long_sequence.py` | Tests for long-sequence evaluator | ✅ Already on main |
| `tests/test_task2_drift_reduction.py` | Drift boundary tests | ✅ Already on main |
| `tests/test_task2_calibration.py` | Calibration tests | ✅ Already on main |
| `tests/test_task2_health_transition.py` | Health transition tests | ✅ Already on main |
| `tests/test_task2_mvp.py` | MVP processor tests | ✅ Already on main |
| `tests/test_task2_processor_fallback.py` | Fallback logic tests | ✅ Already on main |
| `tests/test_task2_quality_proxy.py` | Quality proxy tests | ✅ Already on main |

### Potentially Cherry-Pickable from Experimental Branches

These items exist only on experimental branches but are **evaluation-only** infrastructure:

| File | Source Branch | Purpose | Cherry-Pick Risk |
|------|---------------|---------|------------------|
| `src/evaluation/task2_thermal_microlever.py` | feature/task2-thermal-only-microlever | Microlever evaluation harness | LOW - pure evaluation |
| `src/evaluation/task2_thermal_microlever_manifest.json` | feature/task2-thermal-only-microlever | Scenario manifest format | LOW - data only |
| `tools/run_task2_thermal_microlever_eval.py` | feature/task2-thermal-only-microlever | CLI for microlever eval | LOW - CLI wrapper |
| `tests/test_task2_thermal_microlever_eval.py` | feature/task2-thermal-only-microlever | Tests for microlever eval | LOW - test only |
| `tools/run_phase10_task2_dress_rehearsal.py` | feature/task2-thermal-only-microlever | Phase 10 dress rehearsal | LOW - eval runner |
| `tools/run_phase10_gpu_validation.py` | feature/task2-thermal-only-microlever | GPU validation tool | LOW - validation only |

## DO NOT Cherry-Pick

These items alter estimator behavior or expose experimental variants:

| Item | Reason |
|------|--------|
| Any changes to `src/task2/estimator.py` | Production estimator logic |
| Any changes to `src/task2/drift_control.py` | Production drift guard |
| Any changes to `src/task2/health_logic.py` | Production health handling |
| Any thermal knob settings changes | Experimental variants not validated |
| Drift-crush quarantine/smoothing logic | Rejected experiment |
| `build_variant_settings()` with experimental variants | Exposes rejected config |

## Recommendation

1. **No immediate cherry-pick required** - Main branch has sufficient evaluation infrastructure
2. **If future eval needed** - The thermal microlever evaluation harness pattern is clean but should have experimental variant references removed before any cherry-pick
3. **Manifest format is useful** - The scenario manifest JSON format could be extracted as a generic pattern

## Infrastructure Patterns Worth Preserving (Design Only)

Even without cherry-picking, these patterns from experimental work are worth documenting:

### Scenario Manifest Format
```json
{
  "scenarios": [
    {
      "scenario_id": "thermal_long_degraded_1",
      "csv_path": "path/to/translation.csv",
      "video_path": "path/to/video.mp4",
      "frame_limit": 720,
      "frame_stride": 2,
      "health_segments": [
        {"health": "1", "length": 120},
        {"health": "0", "length": 480},
        {"health": "1", "length": 120}
      ],
      "transform_profile": "none",
      "tags": ["thermal", "long_degraded"]
    }
  ]
}
```

### Evaluation Report Structure
- JSON summary with aggregate metrics
- Markdown table for human review
- Per-scenario breakdown with drift/jump/recovery metrics

---

*This document catalogs infrastructure that could be reused without affecting production behavior.*
