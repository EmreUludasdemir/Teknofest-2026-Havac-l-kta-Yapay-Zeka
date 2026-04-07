# Task 3 Weight Watchpoint

**Date:** 2026-04-07  
**Status:** BLOCKED_BY_WEIGHT

## Current State

Task 3 experimental score-up work (YOLOE-based matching) is **BLOCKED** because no valid offline YOLOE-compatible weight file is staged locally.

## Blocker Details

| Item | Status |
|------|--------|
| Branch | `feature/task3-score-up-yoloe` |
| Required weight | YOLOE-compatible `.pt` or `.onnx` |
| Expected path | Not defined (no weight staged) |
| Local weight search | **NONE FOUND** |
| Validation status | Cannot validate |

## Weight Search Results

Searched paths:
- `C:\TEKNOFEST\**/*.pt` → **No files found**
- `C:\TEKNOFEST\**/*.onnx` → **No files found**
- `C:\Users\Emre\.teknofest_models\` → Contains yolo26n.pt, yolo11n.pt (NOT YOLOE)

**Conclusion:** No YOLOE-compatible weight is present in the repository or standard model cache.

## What Is YOLOE?

YOLOE (YOLO with Embeddings) is an experimental detector variant that produces feature embeddings alongside detections. These embeddings enable:
- Learned descriptor matching (vs ORB)
- Potentially better reference object identification
- Requires specific weight format trained for embedding output

## Unblock Criteria

To unblock Task 3 experimental work:

### 1. Obtain Valid YOLOE Weight

Source options:
- Train locally with embedding head
- Download pretrained YOLOE weights
- Convert existing YOLO weights (may not work without retraining)

### 2. Stage Weight Locally

```powershell
# Place weight in standard location
mkdir -Force C:\Users\Emre\.teknofest_models\yoloe
# Copy weight file to:
# C:\Users\Emre\.teknofest_models\yoloe\yoloe_base.pt
```

### 3. Run Revalidation Command

Once weight is staged:

```powershell
# From project root
cd C:\TEKNOFEST

# Checkout experimental branch
git checkout feature/task3-score-up-yoloe

# Run weight validation (command TBD based on branch state)
python tools/run_task3_weight_staging_revalidation.py --weight-path "C:\Users\Emre\.teknofest_models\yoloe\yoloe_base.pt"
```

### 4. Demonstrate Improvement

| Metric | ORB Baseline | YOLOE Target | Minimum Threshold |
|--------|--------------|--------------|-------------------|
| Match accuracy | Current | Must improve | >10% improvement |
| False positive rate | Current | Must not increase | No regression |
| Edge case handling | Stable | Must remain stable | No regression |

## Current Production Path

Task 3 production uses ORB-based matching:

```
src/task3/
├── matcher.py         # ORB feature matching
├── reference_cache.py # Preloaded reference images
├── verifier.py        # Match verification
└── no_match_logic.py  # Fallback for unmatched objects
```

This path is **validated and frozen**. Do not modify unless YOLOE demonstrates clear improvement.

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-07 | YOLOE blocked | No weight staged |
| 2026-04-07 | ORB baseline frozen | Validated, stable |
| TBD | Re-evaluate | Only if weight appears |

## Watchpoint Actions

### If Weight Appears

1. Notify team
2. Stage in documented location
3. Run validation command
4. Compare against ORB baseline
5. Document results
6. Decision: merge if >10% improvement, otherwise archive

### If Weight Does Not Appear

1. Keep Task 3 experimental branch archived
2. Do not attempt merge
3. Use ORB baseline for competition
4. Document post-competition learnings

## Files on Experimental Branch (Do Not Merge)

These files exist only on `feature/task3-score-up-yoloe`:

| File | Purpose | Merge Status |
|------|---------|--------------|
| `src/task3/experimental/` | YOLOE pipeline | ❌ BLOCKED |
| `src/evaluation/task3_experimental.py` | Evaluation harness | ❌ BLOCKED |
| `tools/run_task3_experimental_eval.py` | Experiment runner | ❌ BLOCKED |
| `tests/test_task3_experimental_*.py` | Experimental tests | ❌ BLOCKED |

## Summary

| Question | Answer |
|----------|--------|
| Is YOLOE weight staged? | **NO** |
| Can we run YOLOE validation? | **NO** |
| Should we merge YOLOE branch? | **NO** |
| What do we use for competition? | **ORB baseline (main branch)** |
| When to revisit? | **Only if valid weight appears** |

---

*Task 3 score-up is paused until a valid YOLOE-compatible weight is staged locally.*
