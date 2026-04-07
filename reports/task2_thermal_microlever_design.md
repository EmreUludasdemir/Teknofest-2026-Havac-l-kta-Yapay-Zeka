# Task 2 Thermal Microlever Design

## Approach
- Start from the safe pre-drift-crush Task 2 baseline.
- Reuse the previous Task 2 health0 slice harness shape, but test only thermal-only knob overrides.
- Keep production defaults unchanged and avoid any new hold, quarantine, smoothing, or broad estimator-family logic.

## Candidate Micro-Levers
- `thermal_confidence_floor_045`: raise `task2_confidence_floor_thermal` from `0.40` to `0.45`.
- `thermal_sensor_hint_weight_008`: reduce `task2_sensor_hint_max_weight_thermal` from `0.15` to `0.08`.

## Scenarios
- `thermal_long_degraded_1`: transform=`none`, frame_limit=`720`, frame_stride=`2`, health_segments=`120x1, 480x0, 120x1`, tags=`['thermal', 'long_degraded', 'transition']`
- `thermal_long_degraded_2`: transform=`none`, frame_limit=`720`, frame_stride=`2`, health_segments=`120x1, 480x0, 120x1`, tags=`['thermal', 'long_degraded', 'transition']`
- `thermal_short_bursts_1`: transform=`none`, frame_limit=`480`, frame_stride=`2`, health_segments=`40x1, 40x0, 40x1, 40x0, 40x1, 40x0, 40x1, 40x0, 40x1, 40x0, 40x1, 40x0`, tags=`['thermal', 'short_bursts']`
- `thermal_transition_noise_1`: transform=`low_contrast_noise`, frame_limit=`360`, frame_stride=`2`, health_segments=`90x1, 180x0, 90x1`, tags=`['thermal', 'transition_noise']`
- `thermal_frozen_frames_2`: transform=`freeze_every_6`, frame_limit=`360`, frame_stride=`2`, health_segments=`90x1, 180x0, 90x1`, tags=`['thermal', 'frozen_frames']`
- `rgb_long_degraded_control_1`: transform=`weak_sensor_hint`, frame_limit=`720`, frame_stride=`2`, health_segments=`120x1, 480x0, 120x1`, tags=`['rgb', 'control', 'weak_sensor_hint']`
- `rgb_transition_control_2`: transform=`none`, frame_limit=`360`, frame_stride=`2`, health_segments=`90x1, 180x0, 90x1`, tags=`['rgb', 'transition_control']`

## Variant Settings
- `baseline` -> `{'task2_confidence_floor_thermal': 0.4, 'task2_sensor_hint_max_weight_thermal': 0.15, 'task2_phase_primary_response_min_thermal': 0.45, 'task2_z_update_scale_thermal': 0.0}`
- `thermal_confidence_floor_045` -> `{'task2_confidence_floor_thermal': 0.45, 'task2_sensor_hint_max_weight_thermal': 0.15, 'task2_phase_primary_response_min_thermal': 0.45, 'task2_z_update_scale_thermal': 0.0}`
- `thermal_sensor_hint_weight_008` -> `{'task2_confidence_floor_thermal': 0.4, 'task2_sensor_hint_max_weight_thermal': 0.08, 'task2_phase_primary_response_min_thermal': 0.45, 'task2_z_update_scale_thermal': 0.0}`
