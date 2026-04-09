# Task 1 RF-DETR Risks

**Date:** 2026-04-09  
**Branch:** `feature/task1-rfdetr-sahi`

## Primary risks

### 1. No connected RF-DETR artefact yet

- No local RF-DETR or RF-DETR-Base weight was found under `C:\Users\Emre\.teknofest_models`
- Without a real detector artefact, this branch cannot yet produce a fair result against `yolo26n`

### 2. Target aerial datasets are not locally connected

- `VisDrone2019-DET` not found in checked local roots
- `UAVDT` not found in checked local roots
- Repo-local TEKNOFEST sample videos exist, but they are not a complete labeled small-object benchmark for mAP reporting

### 3. Current experimental Python entry point is unhealthy

- `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe` resolves to a missing base interpreter in this shell
- This blocks direct verification of RF-DETR / SAHI / augmentation dependencies from the current environment

### 4. No label-aware branch comparison harness exists yet

- Current repo has:
  - runtime profiling coverage
  - sampled replay coverage
  - post-detector Task 1 logic coverage
- Current repo does not yet have a completed label-aware RF-DETR-vs-yolo26n evaluator for:
  - overall mAP
  - small-object AP
  - per-class AP
  - motion-status-sensitive performance
  - landing-status-sensitive performance

### 5. SAHI misuse risk

- Inference-only SAHI without tile-aware training would create an unfair and likely misleading comparison
- This branch must keep SAHI as a train/infer-paired experiment

## Risk impact

| Risk | Impact | Current state |
| --- | --- | --- |
| Missing RF-DETR artefact | High | active |
| Missing VisDrone/UAVDT | High | active |
| Broken experiment Python runtime | High | active |
| No label-aware AP harness yet | Medium | active |
| SAHI train/infer mismatch | Medium | avoid by design |

## Immediate mitigation order

1. Connect a real local RF-DETR-Base weight or training output
2. Connect `VisDrone2019-DET`
3. Connect `UAVDT`
4. Repair the current experiment Python runtime
5. Add the label-aware Task 1 comparison harness on top of existing baseline artefacts

## Current branch posture

`EXPERIMENTAL ONLY`

This branch is worth continuing, but only after the missing detector artefact, datasets, and runnable Python entry point are restored.
