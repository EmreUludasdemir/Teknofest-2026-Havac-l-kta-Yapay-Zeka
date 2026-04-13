# Task 1 YOLO-First Experimental Comparison

**Date:** 2026-04-12
**Branch:** `feature/task1-rfdetr-sahi`
**Decision:** `EXPERIMENTAL ONLY`

## Current Truth

- Task 1 is the only active score-up branch.
- Task 2 remains frozen with no merge candidate.
- Task 3 remains blocked by missing real YOLOE weight.
- Short-term direction on this branch is YOLO-first proxy comparison, not RF-DETR-first swap.

## Runtime and Dependency State

- Selected Python: `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe`
- Version: `3.12.10`
- Repo compatible: `True`
- `teknofest-gpu` venv status: `callable`
- The teknofest-gpu venv entry point is callable from the repaired Python 3.12 environment.
- Optional dependencies: `{'cv2': True, 'ultralytics': True, 'torch': True, 'timm': True, 'transformers': False, 'sahi': False, 'onnxruntime': True, 'tensorrt': True}`

## Local Artefact Truth

- `yolo26n` count: `2`
- `yolo11n` count: `2`
- `rtdetr` count: `0`
- `rfdetr` count: `0`

## Dataset Truth

- Connected dataset roots: `0`
- No local labeled aerial dataset is connected. Real AP comparison remains blocked.

## Variant Results

### `candidate_tiled_probe`

- status: `proxy_only`
- candidate: `yolo11s_exp2_smoke`
- artifact: `C:\TEKNOFEST\_logs\task1_training\exp2_combined_smoke\weights\best.pt`
- mode: `tiled_yolo`
- sahi status: `manual_tiling_probe_only_not_sahi`
- runtime profile p50 ms: `None`
- runtime profile p95 ms: `None`
- proxy replay p50 ms: `279.8045`
- proxy replay p95 ms: `8025.4099`
- zero-detection rate: `1.0`
- duplicate ratio: `0.0`
- small box count: `0`
- class histogram: `{}`
- tile probe: `{'tile_count_total': 48, 'raw_detection_count': 0, 'merged_detection_count': 0, 'tile_merge_duplicate_ratio': 0.0}`
- issues: `['all_sample_frames_zero_detection', 'no_uap_uai_predictions_seen_in_proxy_run']`

### `candidate_probe`

- status: `proxy_only`
- candidate: `yolo11s_exp2_smoke`
- artifact: `C:\TEKNOFEST\_logs\task1_training\exp2_combined_smoke\weights\best.pt`
- mode: `single_yolo`
- sahi status: `not_applicable`
- runtime profile p50 ms: `None`
- runtime profile p95 ms: `None`
- proxy replay p50 ms: `35.216`
- proxy replay p95 ms: `457.8774`
- zero-detection rate: `1.0`
- duplicate ratio: `0.0`
- small box count: `0`
- class histogram: `{}`
- issues: `['all_sample_frames_zero_detection', 'no_uap_uai_predictions_seen_in_proxy_run']`

## Proxy Summary

- best latency variant: `candidate_probe`
- lowest zero-detection variant: `candidate_tiled_probe`
- highest small-box variant: `None`
- control outcome: `None`
- tiled probe outcome: `None`
- comparisons: `{}`

## Label-Aware State

- status: `blocked_by_labels`
- dataset root: `None`
- labels format: `None`
- No local labeled aerial dataset is connected.
- Proxy comparison is the only honest result currently available.

## Recommendation

`EXPERIMENTAL ONLY`

Reason:

- YOLO-first proxy comparison is now runnable on real local artefacts.
- Manual tiling is available as an evaluator-only small-object probe.
- Real label-aware AP comparison is still blocked by missing labeled aerial data.
- RT-DETR / RF-DETR remain gate-only because local artefacts are absent.
