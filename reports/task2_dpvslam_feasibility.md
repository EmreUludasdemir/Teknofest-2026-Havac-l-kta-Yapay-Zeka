# Task 2 DPVO/DPV-SLAM Feasibility Study

**Date:** 2026-04-07  
**Branch:** `feature/task2-dpvslam-feasibility`  
**Priority:** #3 (Only if architectural jump)  
**Status:** FEASIBILITY STUDY - Not yet started

## Executive Summary

Task 2 (translation estimation during health=0) has a frozen production baseline. Previous knob-tuning branches (drift-crush, thermal-microlever) failed to improve aggregate health=0 drift. This document explores whether an **architectural class change** (DPVO/DPV-SLAM) could succeed where threshold tuning failed.

**Critical:** This is NOT threshold tuning. Do NOT reuse rejected thermal knob branches.

## Why Previous Approaches Failed

| Branch | Approach | Result | Why Failed |
|--------|----------|--------|------------|
| drift-crush | Quarantine windows, smoothing | ❌ Rejected | Regressions in recovery |
| thermal-microlever | Confidence floor, sensor hint | ❌ Rejected | 0% improvement, some worse |

**Root Cause Analysis:**

The current estimator is fundamentally limited by:
1. **Single-frame anchor matching** - No temporal integration
2. **No explicit motion model** - Velocity is derived, not optimized
3. **No loop closure** - Drift accumulates without correction
4. **No metric scale** - Translation units are camera-relative

## Architectural Alternative: DPVO/DPV-SLAM

### What is DPVO?

Deep Patch Visual Odometry (DPVO) is a learned visual odometry system that:
- Uses patch-based feature tracking
- Jointly optimizes poses and depths
- Runs in real-time on GPU
- Handles dynamic scenes

### What is DPV-SLAM?

DPV-SLAM extends DPVO with:
- Loop closure detection
- Global bundle adjustment
- Drift correction
- Map management

### Potential Benefits for Task 2

| Current Limitation | DPVO/DPV-SLAM Solution |
|--------------------|------------------------|
| Single-frame matching | Multi-frame optimization |
| No motion model | Learned motion prior |
| No loop closure | Global consistency |
| Camera-relative scale | Metric scale recovery |
| Static scene assumption | Dynamic masking integration |

## Proposed Architecture

```
                    ┌─────────────────────────┐
                    │    Task 1 Detections    │
                    │    (Dynamic Objects)    │
                    └───────────┬─────────────┘
                                │
                                ▼
Frame Input ──────► ┌─────────────────────────┐
                    │    Dynamic Masking      │
                    │  - Exclude moving obj   │
                    │  - Use Task 1 boxes     │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │       DPVO Core         │
                    │  - Patch extraction     │
                    │  - Feature matching     │
                    │  - Bundle adjustment    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
          ┌─────────────────┐    ┌─────────────────┐
          │   DPVO-Only     │    │   DPV-SLAM      │
          │   (Fast path)   │    │   (Full path)   │
          └────────┬────────┘    └────────┬────────┘
                   │                      │
                   │     ┌────────────────┘
                   │     │
                   ▼     ▼
          ┌─────────────────────────┐
          │   Metric Scale Recovery │
          │  - Sensor hint fusion   │
          │  - Calibration lookup   │
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │   Health-Aware Output   │
          │  - health=1: Reference  │
          │  - health=0: DPVO est   │
          └───────────┬─────────────┘
                      │
                      ▼
             Translation Output
```

## Feasibility Requirements

### 1. DPVO Availability

| Package | Status | Source |
|---------|--------|--------|
| dpvo | Available | `pip install dpvo` or GitHub |
| droid-slam | Available | GitHub (princeton-vl) |
| lietorch | Required | GPU-accelerated SE3 |

### 2. Hardware Requirements

| Resource | DPVO | DPV-SLAM | Budget |
|----------|------|----------|--------|
| VRAM | ~2GB | ~4GB | 8GB max |
| Latency | ~30ms | ~50ms | <100ms |
| CPU | Low | Medium | Available |

### 3. Integration Complexity

| Component | Difficulty | Risk |
|-----------|------------|------|
| Dynamic masking | Low | Reuse Task 1 |
| DPVO core | Medium | Well-documented |
| Scale recovery | High | Research needed |
| Health logic | Low | Existing pattern |

## Implementation Plan

### Phase 1: DPVO Standalone Test

```python
# tools/test_dpvo_standalone.py
import dpvo
from dpvo import DPVO

def test_dpvo_on_task2_sequence():
    """Test DPVO on a Task 2 evaluation sequence."""
    
    model = DPVO(weights="dpvo.pth")
    
    for frame in load_sequence("thermal_long_degraded_1"):
        # Run DPVO
        pose = model.track(frame.image)
        
        # Extract translation
        translation = pose[:3, 3]  # x, y, z
        
        # Compare against ground truth
        error = compute_error(translation, frame.ground_truth)
        
    return aggregate_metrics()
```

### Phase 2: Dynamic Masking Integration

```python
# src/task2/experimental/dpvo_with_masking.py
class DPVOWithMasking:
    def __init__(self, dpvo_model, task1_detector):
        self.dpvo = dpvo_model
        self.detector = task1_detector
    
    def estimate(self, frame: FrameEnvelope, image: np.ndarray) -> Translation:
        # 1. Get Task 1 detections
        detections = self.detector.detect(image)
        
        # 2. Create dynamic mask
        mask = create_dynamic_mask(image.shape, detections)
        
        # 3. Run DPVO with masked features
        pose = self.dpvo.track(image, mask=mask)
        
        # 4. Extract and scale translation
        return self._to_canonical_translation(pose)
```

### Phase 3: Scale Recovery

```python
# src/task2/experimental/scale_recovery.py
class MetricScaleRecovery:
    def __init__(self, calibration_bundle):
        self.calibration = calibration_bundle
    
    def recover_scale(
        self,
        dpvo_translation: np.ndarray,
        sensor_hint: dict,
        confidence: float,
    ) -> np.ndarray:
        """
        Recover metric scale from:
        1. Camera calibration (known focal length, baseline)
        2. Sensor hint (if available and confident)
        3. Historical scale factor (if stable)
        """
        
        # Method 1: Calibration-based
        scale_from_calib = self._scale_from_calibration(dpvo_translation)
        
        # Method 2: Sensor hint fusion
        if confidence > 0.3 and sensor_hint:
            scale_from_hint = self._scale_from_sensor_hint(
                dpvo_translation, sensor_hint
            )
            # Weighted fusion
            alpha = min(confidence, 0.7)
            scale = alpha * scale_from_hint + (1 - alpha) * scale_from_calib
        else:
            scale = scale_from_calib
        
        return dpvo_translation * scale
```

### Phase 4: Evaluation

Compare against frozen Task 2 baseline on **exact same** evaluation set.

```python
# src/evaluation/task2_dpvslam_eval.py
def evaluate_dpvslam_vs_baseline(manifest_path: Path) -> dict:
    """
    Compare DPVO/DPV-SLAM against Task 2 baseline.
    
    CRITICAL: Use same scenarios as drift-crush evaluation
    to ensure fair comparison.
    """
    
    baseline = Task2Estimator(...)  # Frozen production
    dpvo_estimator = DPVOTask2Estimator(...)  # Experimental
    
    results = {"baseline": [], "dpvo": []}
    
    for scenario in load_manifest(manifest_path):
        baseline_result = run_scenario(baseline, scenario)
        dpvo_result = run_scenario(dpvo_estimator, scenario)
        
        results["baseline"].append(baseline_result)
        results["dpvo"].append(dpvo_result)
    
    return {
        "aggregate_health0_drift": {
            "baseline": mean([r["drift"] for r in results["baseline"]]),
            "dpvo": mean([r["drift"] for r in results["dpvo"]]),
        },
        "recovery_error": {...},
        "catastrophic_jumps": {...},
    }
```

## Success Criteria

### Minimum for Continued Development

| Metric | Baseline | DPVO Target | Required |
|--------|----------|-------------|----------|
| Aggregate health0 drift | 16.1 | <14.0 | >13% improvement |
| Recovery error | 0.0 | 0.0 | No regression |
| Catastrophic jumps | 7 | ≤7 | No regression |
| Latency | ~30ms | <100ms | Acceptable |

### Stop Criteria

**Stop immediately if:**
- Aggregate drift does not improve by >10%
- Recovery error increases
- Catastrophic jumps increase
- VRAM exceeds 6GB
- Latency exceeds 150ms

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| DPVO doesn't improve drift | Medium | High | Stop early, don't waste resources |
| Scale recovery fails | High | High | Fall back to existing anchor |
| Thermal imagery issues | High | Medium | RGB-only if needed |
| Integration complexity | Medium | Medium | Isolated branch |

## Dependencies

```txt
# requirements-task2-dpvslam.txt
# WARNING: These are experimental and may conflict with production
dpvo>=0.1.0           # If pip installable
lietorch>=0.1.0       # SE3 operations
kornia>=0.7.0         # Geometric transforms
```

## Decision Framework

| Decision | Criteria |
|----------|----------|
| **CONTINUE TO FULL INTEGRATION** | >13% drift improvement in Phase 1 |
| **PARTIAL: Scale recovery only** | Scale method useful but DPVO not |
| **STOP: Not viable** | No improvement or regression |

## Comparison with Rejected Approaches

| Aspect | Drift-Crush | Thermal-Microlever | DPVO |
|--------|-------------|--------------------|----|
| Class | Parameter tuning | Parameter tuning | Architectural |
| Changes estimator core? | Partially | No | Yes |
| New motion model? | No | No | Yes |
| Loop closure? | No | No | Optional |
| Risk of regression | High (proven) | Medium (proven) | Unknown |
| Potential upside | Low (proven) | Zero (proven) | Unknown |

## Recommendation

1. **Do NOT start DPVO work until Task 3 YOLOE is evaluated**
   - Task 3 has higher expected ROI
   - Task 2 DPVO has significant integration risk

2. **If Task 3 completes successfully**, consider DPVO as Phase 2

3. **If Task 3 blocked**, re-evaluate DPVO priority

4. **Never revisit thermal knob tuning** - proven ineffective

---

*This feasibility study defines what would be needed for a genuine architectural improvement to Task 2. It is NOT a commitment to implement.*
