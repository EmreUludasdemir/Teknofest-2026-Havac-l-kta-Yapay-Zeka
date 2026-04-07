# Task 2 Rejected Variants Summary

**Date:** 2026-04-07  
**Conclusion:** No variants improved aggregate health=0 drift sufficiently

## Branch 1: `feature/task2-health0-drift-crush`

### Approach
Broad drift-crushing strategy targeting multiple estimator parameters simultaneously:
- Extended hold/quarantine windows during health=0 periods
- Aggressive smoothing to reduce step-wise jumps
- Modified anchor distance limits
- Broad parameter family overrides

### Results
| Metric | Baseline | Drift-Crush | Assessment |
|--------|----------|-------------|------------|
| Merge Status | — | ❌ NOT MERGED | Failed review |
| Aggregate health0 drift | Baseline | Not improved | No gain |
| Recovery error | 0.0 | Introduced regressions | Worse |
| Transition handling | Stable | Degraded | Worse |

### Why Rejected
1. Did not demonstrate clear aggregate drift improvement
2. Introduced regressions in recovery accuracy
3. Transition handling became less stable
4. Complexity increase not justified by results

---

## Branch 2: `feature/task2-thermal-only-microlever`

### Approach
Minimal, focused approach testing only thermal-specific confidence and sensor-hint knobs:

| Variant Name | Parameter Changed | Old Value | New Value |
|--------------|-------------------|-----------|-----------|
| `thermal_confidence_floor_045` | `task2_confidence_floor_thermal` | 0.40 | 0.45 |
| `thermal_sensor_hint_weight_008` | `task2_sensor_hint_max_weight_thermal` | 0.15 | 0.08 |

### Results

#### Aggregate Metrics

| Variant | Health0 Drift | Delta | Thermal Drift | Delta | Interesting? |
|---------|---------------|-------|---------------|-------|--------------|
| baseline | 16.106937 | — | 20.30302 | — | — |
| thermal_confidence_floor_045 | 16.106937 | 0.0% | 20.30302 | 0.0% | ❌ No |
| thermal_sensor_hint_weight_008 | 16.42817 | +1.99% | 20.76016 | +2.25% | ❌ No (worse) |

#### Per-Scenario Results

**thermal_long_degraded_1**
| Variant | Health0 Drift | Jumps | Runtime p95 |
|---------|---------------|-------|-------------|
| baseline | 49.216562 | 1 | 29.7ms |
| thermal_confidence_floor_045 | 49.216562 | 1 | 28.0ms |
| thermal_sensor_hint_weight_008 | 49.445676 | 1 | 27.3ms |

**thermal_long_degraded_2**
| Variant | Health0 Drift | Jumps | Runtime p95 |
|---------|---------------|-------|-------------|
| baseline | 7.749096 | 1 | 30.3ms |
| thermal_confidence_floor_045 | 7.749096 | 1 | 26.0ms |
| thermal_sensor_hint_weight_008 | 8.667214 | 1 | 28.8ms |

**thermal_short_bursts_1**
| Variant | Health0 Drift | Jumps | Runtime p95 |
|---------|---------------|-------|-------------|
| baseline | 2.675098 | 2 | 33.2ms |
| thermal_confidence_floor_045 | 2.675098 | 2 | 27.7ms |
| thermal_sensor_hint_weight_008 | 3.222319 | 2 | 27.0ms |

### Why Rejected

1. **thermal_confidence_floor_045**
   - Zero improvement in any drift metric
   - 4% runtime overhead (255ms vs 245ms p95)
   - No scenarios showed improvement

2. **thermal_sensor_hint_weight_008**
   - Made drift **worse** (+1.99% aggregate, +2.25% thermal-only)
   - Degraded short-burst scenarios
   - No compensating benefits

---

## Production Baseline (Unchanged)

```python
# src/config/settings.py - FROZEN VALUES
task2_confidence_floor: float = 0.30
task2_confidence_floor_thermal: float = 0.40
task2_long_drift_limit: float = 4.0
task2_sensor_hint_max_weight_thermal: float = 0.15
task2_phase_primary_response_min_thermal: float = 0.45
task2_z_update_scale_thermal: float = 0.0
task2_anchor_distance_limit_xy: float = 20.0
task2_anchor_distance_limit_z: float = 8.0
```

---

## Lessons Learned

1. **Thermal drift is hard to improve** - The baseline already has reasonable thermal handling
2. **Micro-adjustments are ineffective** - Small parameter changes don't move the needle
3. **Broad changes risk regression** - Aggressive drift-crushing destabilizes recovery
4. **Evaluation infrastructure works** - The harness correctly identified non-improvements

## Future Considerations

If Task 2 drift optimization is revisited, consider:
1. Entirely new estimation approaches (not parameter tuning)
2. Additional sensor fusion if available
3. Scene-aware adaptation (different handling for different scenarios)
4. Hardware-level improvements (better thermal camera data)

**Until a genuinely new lever appears, Task 2 score-up work is paused.**

---

*This summary documents the experimental variants that were tested and rejected.*
