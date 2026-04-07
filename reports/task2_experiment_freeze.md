# Task 2 Experiment Freeze

**Date:** 2026-04-07  
**Status:** FROZEN - No merge candidates

## Executive Summary

After extensive experimentation on two separate branches targeting Task 2 health=0 drift reduction, **no tested variant improved aggregate health=0 drift enough to justify merge**. The production Task 2 baseline remains unchanged.

## What Was Tried

### 1. Broad Drift-Crush Branch (`feature/task2-health0-drift-crush`)

Attempted comprehensive drift-reduction via:
- Extended quarantine windows during health=0 transitions
- Aggressive smoothing and hold-mode logic
- Broad estimator-family parameter overrides

**Result:** Did not clear merge review. Introduced regressions in recovery error and transition handling.

### 2. Thermal-Only Microlever Branch (`feature/task2-thermal-only-microlever`)

Focused, minimal approach testing only thermal-specific knobs:
- `thermal_confidence_floor_045`: Raised `task2_confidence_floor_thermal` from `0.40` to `0.45`
- `thermal_sensor_hint_weight_008`: Reduced `task2_sensor_hint_max_weight_thermal` from `0.15` to `0.08`

**Result:** Did not clear merge review. Neither micro-lever showed meaningful improvement:

| Variant | Aggregate Health0 Drift | Delta Ratio | Interesting? |
|---------|-------------------------|-------------|--------------|
| baseline | 16.106937 | — | — |
| thermal_confidence_floor_045 | 16.106937 | 0.0% | No |
| thermal_sensor_hint_weight_008 | 16.42817 | +1.99% | No (worse) |

## What Failed to Improve

1. **Aggregate health=0 drift** - No variant reduced overall drift accumulation
2. **Thermal-only drift** - 20.30302 baseline, variants showed 0% or worse
3. **Transition drift** - No improvement observed
4. **Catastrophic jumps** - Remained at 7 across all variants
5. **Recovery continuity** - No meaningful gains

## What Remains Production Baseline

The validated production Task 2 path includes:

### Core Estimator (`src/task2/`)
- `estimator.py` - Anchor-based translation estimator
- `health_logic.py` - Health status handling
- `drift_control.py` - Drift guard with confidence-aware clamping
- `calibration.py` - Camera calibration profiles
- `masking.py` - Dynamic object masking

### Production Settings (`src/config/settings.py`)
```python
task2_confidence_floor: float = 0.30
task2_confidence_floor_thermal: float = 0.40
task2_long_drift_limit: float = 4.0
task2_sensor_hint_max_weight_thermal: float = 0.15
task2_phase_primary_response_min_thermal: float = 0.45
task2_z_update_scale_thermal: float = 0.0
```

These settings are **validated and frozen**. Do not modify without compelling new evidence.

## Infrastructure Still Useful

The following evaluation infrastructure may be cherry-picked if needed (see `task2_reusable_infra_candidates.md`):

- Replay manifest support for scenario-based evaluation
- Report generation helpers (markdown table writers)
- Long-sequence evaluation harness (`src/evaluation/task2_long_sequence.py` - already on main)

## Decision

- **Broad drift-crush:** NOT MERGEABLE
- **Thermal-only microlever:** NOT MERGEABLE
- **Production Task 2 baseline:** STAYS AS-IS
- **Score-up work:** PAUSED unless genuinely new lever appears

## Next Steps

1. Do not attempt further Task 2 drift-reduction without new theoretical basis
2. Any future experiments must demonstrate >10% aggregate drift improvement in isolated testing before integration attempt
3. Focus resources on Task 1 and Task 3 if score optimization is needed
4. Monitor competition feedback for any Task 2 scoring clarifications

---

*This freeze document was generated after systematic evaluation of experimental branches showed no viable merge candidates.*
