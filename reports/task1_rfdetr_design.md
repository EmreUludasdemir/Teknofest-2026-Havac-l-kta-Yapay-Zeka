# Task 1 RF-DETR Base + SAHI Design

**Date:** 2026-04-09  
**Branch:** `feature/task1-rfdetr-sahi`  
**Status:** branch kickoff complete, evaluation path isolated, no production swap

## Goal

Evaluate whether `RF-DETR-Base` with SAHI-aware training can beat the current `yolo26n` Task 1 production path on aerial small-object detection without changing production defaults.

## Guardrails

- Production Task 1 defaults remain unchanged.
- Task 2 and Task 3 behavior remain untouched.
- Sequential runtime defaults remain untouched.
- This branch is an evaluation branch, not a production detector swap.
- SAHI must be train/infer consistent. Inference-only slicing is not acceptable if training is not tile-aware.

## Current Task 1 Baseline To Reuse

### Runtime and detector baseline

- Production primary candidate remains `yolo26n`.
- Current profiling baseline comes from:
  - `src/tools/profiling_harness.py::evaluate_candidates`
  - `reports/profiling/task1_real_profile_gpu.json`
  - `reports/profiling/task1_real_profile_cpu.json`
- Current replay-style sampled pipeline comes from:
  - `src/evaluation/task1_onnx_validation.py::run_task1_pipeline_on_samples`
  - `src/evaluation/task1_onnx_validation.py::sample_task1_validation_frames`
  - `reports/export/task1_yolo26n_native_vs_onnx.json`

### Current Task 1 logic to preserve around detector experimentation

- Tracking: `src/task1/tracker.py::Task1Tracker`
- Movement classification: `src/task1/motion_logic.py::assign_motion_status`
- Landing suitability: `src/task1/landing_logic.py::assign_landing_status`
- Deduplication: `src/task1/postprocess.py::deduplicate_detections`

This means the detector experiment can stay isolated while still being evaluated under the same Task 1 output contract:

- 4 canonical classes: vehicle, human, UAP, UAI
- motion classification through tracker displacement
- landing suitability through occupancy / overlap logic

## Experimental Path For This Branch

### Detector choice

- First model: `RF-DETR-Base`
- Large model is explicitly out of scope for the first pass
- SAHI support is planned only as part of a tile-aware train + infer path

### Branch-local experiment shape

- Detector experiment stays outside production runtime selection
- Comparison target stays `yolo26n`
- Result reports will decide only:
  - `MERGE CANDIDATE`
  - `PARTIAL MERGE CANDIDATE`
  - `EXPERIMENTAL ONLY`

### Planned branch-local experiment components

- RF-DETR-Base detector training path
- SAHI-aware slicing policy for both training and inference
- Current Task 1 tracker + motion + landing logic reused as-is around detector outputs
- Fair comparison against the same yolo26n replay/eval slices

## Data Sources

### Connected now

- TEKNOFEST sample videos in repo:
  - `data/THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1.MP4`
  - `data/THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_2_Termal.MP4`

### Not locally connected at kickoff

- `VisDrone2019-DET`
- `UAVDT`
- Any explicit local UAP/UAI detection dataset beyond current sample assets

## Required Augmentation Policy

The branch will use this priority order once training data is connected:

1. Mosaic
2. RandomCrop + Resize
3. CopyPaste for rare UAP/UAI classes
4. Weather augmentation: rain / fog / snow
5. Night degradation / brightness reduction

## Comparison Harness To Reuse

### Baseline harness already present

- Runtime / VRAM / availability:
  - `src/tools/profiling_harness.py`
- Sampled replay pipeline with current Task 1 logic:
  - `src/evaluation/task1_onnx_validation.py`

### What the RF-DETR branch must add later

- Label-aware Task 1 evaluator for:
  - overall mAP
  - small-object AP
  - per-class AP
  - motion-status-sensitive performance
  - landing-status-sensitive performance

The current repo has runtime and sampled replay coverage, but not a completed RF-DETR-vs-yolo26n label-aware AP harness.

## Kickoff Findings

- Baseline harness exists and is reusable.
- `yolo26n` baseline artefacts already exist for CPU/GPU runtime and ONNX parity.
- No local RF-DETR weight was found under `C:\Users\Emre\.teknofest_models`.
- `VisDrone2019-DET` and `UAVDT` were not found in the checked local roots.
- The current `teknofest-gpu` venv Python entry point is broken from this shell because it points at a missing base interpreter.

## Immediate Branch Scope

This kickoff branch produces:

- a corrected Task 1 RF-DETR design grounded in current repo truth
- a baseline-vs-experimental status report
- a machine-readable comparison status JSON
- a risks document

It does not claim that RF-DETR has beaten `yolo26n` yet.
