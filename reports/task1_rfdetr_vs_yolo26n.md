# Task 1 RF-DETR vs YOLO26n

**Date:** 2026-04-09  
**Branch:** `feature/task1-rfdetr-sahi`  
**Decision:** `EXPERIMENTAL ONLY`

## Summary

This branch kickoff confirmed that the current `yolo26n` production path already has reusable runtime and sampled replay validation artefacts. The RF-DETR side is not yet runnable from this branch kickoff because the detector weights and higher-priority aerial datasets are not locally connected, and the current GPU venv Python entry point is broken in this shell.

The branch therefore remains `EXPERIMENTAL ONLY` at kickoff. No production swap is recommended.

## Current yolo26n baseline evidence

### Runtime profiling baseline

Source files:

- `reports/profiling/task1_real_profile_gpu.json`
- `reports/profiling/task1_real_profile_cpu.json`
- `reports/profiling/task1_real_profile_comparison.md`

Key baseline values:

| Metric | yolo26n GPU | yolo26n CPU |
| --- | --- | --- |
| Available | yes | yes |
| Cold load ms | 105.455 | 90.965 |
| Warm p50 ms | 27.115 | 105.785 |
| Warm p95 ms | 220.6496 | 176.3422 |
| Peak VRAM MB | 212.0 | 0.0 |

### Current replay-style sampled pipeline baseline

Source file:

- `reports/export/task1_yolo26n_native_vs_onnx.json`

Key values:

| Metric | Value |
| --- | --- |
| Validation video | `THYZ_2026_Ornek_Veri_1.MP4` |
| Sample count | 10 |
| Native latency p50 ms | 60.8125 |
| ONNX latency p50 ms | 52.732 |
| Native peak VRAM MB | 2589.0 |
| ONNX peak VRAM MB | 2661.0 |
| ONNX parity accepted | true |

## RF-DETR experimental status at branch kickoff

| Item | Status |
| --- | --- |
| RF-DETR local weight | not connected |
| SAHI train/infer stack | planned, not verified |
| VisDrone2019-DET | not locally connected |
| UAVDT | not locally connected |
| TEKNOFEST sample videos | connected |
| 4-class canonical Task 1 wrapper path | reusable via current tracker/motion/landing logic |

## Fair comparison path to use

The branch will compare RF-DETR against the current `yolo26n` path using:

1. Existing baseline runtime harness:
   - `src/tools/profiling_harness.py`
2. Existing sampled replay pipeline:
   - `src/evaluation/task1_onnx_validation.py`
3. Current Task 1 logic after detector output:
   - tracker
   - motion classification
   - landing suitability
4. New label-aware comparison slices once data is connected:
   - overall mAP
   - small-object AP
   - per-class AP
   - motion-status-sensitive performance
   - landing-status-sensitive performance
   - runtime
   - VRAM

## Current blockers to a real RF-DETR vs yolo26n result

- No local RF-DETR detector weight found under `C:\Users\Emre\.teknofest_models`
- No local `VisDrone2019-DET` or `UAVDT` directory found in the checked roots
- No repo-local labeled aerial Task 1 benchmark was found beyond sample videos
- Current `teknofest-gpu` venv Python entry point is broken from this shell

## Recommendation

`EXPERIMENTAL ONLY`

Reason:

- baseline path is clear and reusable
- RF-DETR path is architecturally reasonable
- but there is no honest experimental score-up result yet
- production defaults must remain untouched until a real like-for-like comparison exists
