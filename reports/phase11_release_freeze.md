# Phase 11: Release Freeze

**Date:** 2026-04-07  
**Status:** PRODUCTION FROZEN

## Release State Summary

| Component | Branch | Status | Merge to Main |
|-----------|--------|--------|---------------|
| Task 1 detector/tracker | main | ✅ VALIDATED | Already on main |
| Task 2 estimator baseline | main | ✅ FROZEN | Already on main |
| Task 3 matcher (ORB-based) | main | ✅ VALIDATED | Already on main |
| Sequential adapter | main | ✅ READY | Already on main |
| Task 2 drift-crush | feature/task2-health0-drift-crush | ❌ REJECTED | **NO** |
| Task 2 thermal microlever | feature/task2-thermal-only-microlever | ❌ REJECTED | **NO** |
| Task 3 YOLOE experimental | feature/task3-score-up-yoloe | ⏸️ BLOCKED | **NO** (blocked by weight) |

## What Is Validated on Main

### Task 1: Object Detection

| File | Purpose | Status |
|------|---------|--------|
| `src/task1/detector.py` | Multi-backend detector | ✅ Validated |
| `src/task1/landing_logic.py` | Landing status assignment | ✅ Validated |
| `src/task1/motion_logic.py` | Motion status assignment | ✅ Validated |
| `src/task1/tracker.py` | Object tracking | ✅ Validated |
| `src/task1/postprocess.py` | Deduplication | ✅ Validated |

**Runtime order:** TensorRT → ONNX → Ultralytics → Synthetic fallback

### Task 2: Translation Estimation

| File | Purpose | Status |
|------|---------|--------|
| `src/task2/estimator.py` | Anchor-based estimator | ✅ FROZEN |
| `src/task2/health_logic.py` | Health status handling | ✅ FROZEN |
| `src/task2/drift_control.py` | Drift guard | ✅ FROZEN |
| `src/task2/calibration.py` | Camera calibration | ✅ FROZEN |
| `src/task2/masking.py` | Dynamic object masking | ✅ FROZEN |

**Frozen settings:**
```python
task2_confidence_floor: 0.30
task2_confidence_floor_thermal: 0.40
task2_long_drift_limit: 4.0
task2_sensor_hint_max_weight_thermal: 0.15
```

### Task 3: Reference Matching

| File | Purpose | Status |
|------|---------|--------|
| `src/task3/matcher.py` | ORB-based matcher | ✅ Validated |
| `src/task3/reference_cache.py` | Reference image cache | ✅ Validated |
| `src/task3/verifier.py` | Match verification | ✅ Validated |
| `src/task3/no_match_logic.py` | No-match handling | ✅ Validated |

**Note:** Task 3 experimental (YOLOE-based) remains blocked and will not merge.

### Infrastructure

| File | Purpose | Status |
|------|---------|--------|
| `src/server/final_sequential_adapter.py` | Competition protocol | ✅ Ready |
| `src/pipeline/mvp_processor.py` | Frame processor | ✅ Validated |
| `src/pipeline/orchestrator.py` | Protocol orchestrator | ✅ Validated |
| `tools/run_runtime_smoke.py` | Main runner | ✅ Ready |

## Blocked Experimental Branches

### `feature/task2-health0-drift-crush`

**Status:** ❌ REJECTED - DO NOT MERGE

**Reason:**
- Introduced regressions in recovery accuracy
- Did not improve aggregate health=0 drift
- Complexity not justified by results

**Files affected (branch-only):**
- Modified estimator logic (rejected)
- Modified drift control (rejected)
- Experimental quarantine windows (rejected)

### `feature/task2-thermal-only-microlever`

**Status:** ❌ REJECTED - DO NOT MERGE

**Reason:**
- `thermal_confidence_floor_045`: 0% improvement
- `thermal_sensor_hint_weight_008`: Made drift **worse** (+1.99%)
- No merge candidate identified

**Files affected (branch-only):**
- `src/evaluation/task2_thermal_microlever.py` (evaluation only)
- `tools/run_task2_thermal_microlever_eval.py` (evaluation only)
- Thermal microlever reports (documentation only)

### `feature/task3-score-up-yoloe`

**Status:** ⏸️ BLOCKED_BY_WEIGHT - DO NOT MERGE

**Reason:**
- Requires offline YOLOE-compatible weight file
- No valid local weight staged
- Cannot validate without weight

**Unblock criteria:**
1. Stage valid YOLOE weight locally
2. Run revalidation command
3. Demonstrate >10% improvement over ORB baseline
4. No regression in edge cases

## What May Enter Main

### Allowed

| Type | Example | Condition |
|------|---------|-----------|
| Documentation | `reports/*.md` | Always allowed |
| Test additions | `tests/test_*.py` | Must not alter production behavior |
| Log improvements | Structured logger changes | Must be backward compatible |
| Config additions | New optional settings | Must not change defaults |
| Bug fixes | Critical issues only | Must have evidence of bug |

### NOT Allowed

| Type | Example | Reason |
|------|---------|--------|
| Estimator changes | `src/task2/estimator.py` | Frozen |
| New control logic | Drift-crush, thermal knobs | Rejected |
| Experimental Task 3 | YOLOE pipeline | Blocked by weight |
| Default changes | Any production default | Requires hard evidence |
| Architecture redesign | Any major refactor | Phase 11 scope |

## Competition Day Rules

1. **DO NOT** modify production code on competition day
2. **DO NOT** merge experimental branches
3. **DO** use only the validated `main` branch
4. **DO** report issues but defer fixes to post-competition
5. **DO** use fallback chain if primary model fails

## Branch Hygiene

```bash
# View current state
git branch -a

# Experimental branches should NOT be merged
# main → production
# feature/task2-* → archived experiments
# feature/task3-score-up-yoloe → blocked, do not merge
```

## Post-Competition Actions

After competition:
1. Archive experimental branches
2. Document competition results
3. Decide on branch cleanup
4. Plan next iteration if needed

---

*This document defines what is allowed to change before competition. When in doubt, don't change it.*
