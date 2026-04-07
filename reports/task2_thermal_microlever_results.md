# Task 2 Thermal Microlever Results

- Final recommendation: `NO MERGE CANDIDATE`
- Baseline aggregate health0 drift: `16.106937`
- Baseline thermal-only drift: `20.30302`
- Baseline RGB/control drift: `6.188922`
- Baseline transition drift: `9.99736`
- Baseline catastrophic jumps: `7`
- Baseline recovery continuity: `22.21175`
- Baseline recovery error: `0.0`
- Baseline runtime p95 ms: `245.3784`

## thermal_confidence_floor_045
- Exact micro-lever tested: `{'task2_confidence_floor_thermal': 0.45, 'task2_sensor_hint_max_weight_thermal': 0.15, 'task2_phase_primary_response_min_thermal': 0.45, 'task2_z_update_scale_thermal': 0.0}`
- Improved scenarios: `[]`
- Target scenario `thermal_long_degraded_1` improvement ratio: `0.0`
- Aggregate health0 drift: `16.106937` (delta ratio `0.0`)
- Thermal-only drift: `20.30302` (delta ratio `0.0`)
- RGB/control drift: `6.188922` (delta ratio `0.0`)
- Transition drift: `9.99736` (delta ratio `0.0`)
- Catastrophic jumps: `7` (delta `0`)
- Recovery continuity / error: `22.21175` / `0.0`
- Runtime p95 ms: `255.2892` (ratio `1.04039`)
- Interesting: `False`

## thermal_sensor_hint_weight_008
- Exact micro-lever tested: `{'task2_confidence_floor_thermal': 0.4, 'task2_sensor_hint_max_weight_thermal': 0.08, 'task2_phase_primary_response_min_thermal': 0.45, 'task2_z_update_scale_thermal': 0.0}`
- Improved scenarios: `[]`
- Target scenario `thermal_long_degraded_1` improvement ratio: `-0.004655`
- Aggregate health0 drift: `16.42817` (delta ratio `0.019944`)
- Thermal-only drift: `20.76016` (delta ratio `0.022516`)
- RGB/control drift: `6.188922` (delta ratio `0.0`)
- Transition drift: `10.387083` (delta ratio `0.038983`)
- Catastrophic jumps: `7` (delta `0`)
- Recovery continuity / error: `22.6454` / `0.0`
- Runtime p95 ms: `244.4159` (ratio `0.996077`)
- Interesting: `False`

## Per-Scenario Health0 Drift

### thermal_long_degraded_1
- baseline: health0 `49.216562`, transition `44.103919`, runtime_p95 `29.7471`, jumps `1`
- thermal_confidence_floor_045: health0 `49.216562`, transition `44.103919`, runtime_p95 `28.0494`, jumps `1`
- thermal_sensor_hint_weight_008: health0 `49.445676`, transition `44.371811`, runtime_p95 `27.2806`, jumps `1`

### thermal_long_degraded_2
- baseline: health0 `7.749096`, transition `17.032155`, runtime_p95 `30.3042`, jumps `1`
- thermal_confidence_floor_045: health0 `7.749096`, transition `17.032155`, runtime_p95 `25.9913`, jumps `1`
- thermal_sensor_hint_weight_008: health0 `8.667214`, transition `17.991539`, runtime_p95 `28.7905`, jumps `1`

### thermal_short_bursts_1
- baseline: health0 `2.675098`, transition `2.675098`, runtime_p95 `33.1847`, jumps `2`
- thermal_confidence_floor_045: health0 `2.675098`, transition `2.675098`, runtime_p95 `27.6751`, jumps `2`
- thermal_sensor_hint_weight_008: health0 `3.222319`, transition `3.222319`, runtime_p95 `27.0225`, jumps `2`

### thermal_transition_noise_1
- baseline: health0 `20.434357`, transition `20.067257`, runtime_p95 `31.0189`, jumps `1`
- thermal_confidence_floor_045: health0 `20.434357`, transition `20.067257`, runtime_p95 `29.2016`, jumps `1`
- thermal_sensor_hint_weight_008: health0 `20.573197`, transition `20.208556`, runtime_p95 `29.0871`, jumps `1`

### thermal_frozen_frames_2
- baseline: health0 `0.049928`, transition `0.057072`, runtime_p95 `28.8538`, jumps `0`
- thermal_confidence_floor_045: health0 `0.049928`, transition `0.057072`, runtime_p95 `28.1508`, jumps `0`
- thermal_sensor_hint_weight_008: health0 `0.084056`, transition `0.081844`, runtime_p95 `27.8929`, jumps `0`

### rgb_long_degraded_control_1
- baseline: health0 `6.217498`, transition `12.644793`, runtime_p95 `780.0974`, jumps `1`
- thermal_confidence_floor_045: health0 `6.217498`, transition `12.644793`, runtime_p95 `810.6608`, jumps `1`
- thermal_sensor_hint_weight_008: health0 `6.217498`, transition `12.644793`, runtime_p95 `779.8929`, jumps `1`

### rgb_transition_control_2
- baseline: health0 `6.112721`, transition `10.012535`, runtime_p95 `751.1605`, jumps `1`
- thermal_confidence_floor_045: health0 `6.112721`, transition `10.012535`, runtime_p95 `814.3327`, jumps `1`
- thermal_sensor_hint_weight_008: health0 `6.112721`, transition `10.012535`, runtime_p95 `760.6926`, jumps `1`

