# Task 1 Dataset Inventory

## Repo File Inventory

- zip count: `2`
- image count: `42776`
- video count: `12`
- label/config count: `36995`

## Zip Truth

### `frames.zip`

- content type: `image-only`
- label format: `none`
- image count: `1805`
- label count: `0`
- video count: `0`
- bad member: `None`
- class counts: `{}`
- empty label files: `0`
- invalid label lines: `0`

### `labels.zip`

- content type: `mixed`
- label format: `yolo`
- image count: `0`
- label count: `1805`
- video count: `0`
- bad member: `None`
- class counts: `{'0': 3222, '1': 553, '2': 167, '3': 122}`
- empty label files: `459`
- invalid label lines: `0`

## Local Paired Truth

- image count: `1805`
- label count: `1805`
- matched pairs: `1805`
- missing image count: `0`
- missing label count: `0`
- frame index range: `0..9020`
- inferred frame stride: `5`

## Public Download Status

### `visdrone`

- status: `downloaded`
- modality: `rgb_aerial`
- similarity: `high`
- classes: `['pedestrian', 'person', 'car', 'van', 'bus', 'truck', 'motor', 'bicycle', 'tricycle', 'awning_tricycle']`
- notes: `['Public research benchmark; follow original VisDrone usage terms.']`

### `uavdt`

- status: `blocked_manual_access`
- modality: `rgb_aerial`
- similarity: `high`
- classes: `['car', 'truck', 'bus']`
- notes: `['Official dataset exists, but direct anonymous archive URL is not wired in this branch yet.']`

### `hit_uav`

- status: `downloaded`
- modality: `thermal_uav`
- similarity: `medium`
- classes: `['person', 'car', 'bicycle', 'other_vehicle']`
- notes: `['Open thermal UAV dataset; keep modality mismatch explicit in reports.']`

### `seadronessee`

- status: `blocked_manual_access`
- modality: `rgb_maritime`
- similarity: `low`
- classes: `['boat', 'jetski', 'lifesaving_appliance', 'buoy', 'swimmer']`
- notes: `['Benchmark access can require registration; skip if anonymous direct archive is unavailable.']`
