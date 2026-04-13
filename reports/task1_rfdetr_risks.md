# Task 1 Experimental Risks

**Date:** 2026-04-12
**Branch:** `feature/task1-rfdetr-sahi`

## Active Risks

### 1. Proxy metrics are not AP evidence

- Zero-detection rate, duplicate ratio and small-box count are only directional probes on unlabeled sample video.
- They cannot justify a merge decision on their own.

### 2. Labeled aerial dataset is still missing

- No local VisDrone/UAVDT or explicit labeled dataset is connected.

### 3. Tiled probe may improve recall proxy at a latency / duplicate cost

- Manual tiling is intentionally branch-local and evaluator-only.
- It must not be sold as SAHI success or production-ready behavior.

### 4. Transformer path remains blocked

- RT-DETR / RF-DETR artefacts are not locally connected.
- Any transformer-first claim would still be fabricated.

### 5. UAP/UAI coverage is still weakly evidenced

- Current local smoke path and sample replay only validate that classes 0/1 are exercised.
- UAP/UAI performance remains an open risk until labeled data is connected.

## Current Blocker List

- all_sample_frames_zero_detection
- no_uap_uai_predictions_seen_in_proxy_run
- no_supported_labeled_dataset_found
