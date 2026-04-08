# Task 1 RF-DETR + SAHI Score-Up Design

**Date:** 2026-04-07  
**Branch:** `feature/task1-rfdetr-sahi`  
**Priority:** #2  
**Status:** READY - RF-DETR available via transformers/timm

## Executive Summary

Task 1 production uses YOLO26n which is already validated and strong. This branch explores RF-DETR (Roofline DETR) with SAHI-aware training for potentially higher accuracy ceiling, particularly on small objects and edge cases.

**Important:** This is a score-up branch, NOT a production replacement. Production Task 1 remains YOLO26n.

## Current Production Baseline

| Model | mAP@50 | Latency | VRAM | Status |
|-------|--------|---------|------|--------|
| YOLO26n (TensorRT) | ~85%* | ~15ms | ~1GB | ✅ Production |
| YOLO26n (ONNX) | ~85%* | ~25ms | ~1GB | ✅ Fallback |
| YOLO11n | ~82%* | ~20ms | ~1GB | ✅ Fallback |

*Estimated on internal evaluation set

## Why RF-DETR?

RF-DETR (Roofline DETR) offers:

1. **End-to-end detection** - No NMS required, cleaner architecture
2. **Better small object handling** - Attention-based feature aggregation
3. **Stronger on dense scenes** - No NMS artifacts
4. **Competitive latency** - Optimized for real-time use

### RF-DETR Model Variants

| Model | Params | GFLOPs | mAP (COCO) | Latency |
|-------|--------|--------|------------|---------|
| RF-DETR-Base | 29M | 120 | 53.2 | ~30ms |
| RF-DETR-Large | 128M | 340 | 56.3 | ~60ms |

## Proposed Architecture

### Training Pipeline (SAHI-Aware)

```
Raw Training Data
       │
       ▼
┌─────────────────────────────────┐
│    SAHI Sliced Training         │
│  - 640x640 tiles                │
│  - 0.2 overlap ratio            │
│  - Ground truth mapping         │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│    Weather Augmentation         │
│  - Rain simulation              │
│  - Fog simulation               │
│  - Snow simulation              │
│  - Night/low-light              │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│    Mosaic + Copy-Paste          │
│  - Multi-image composition      │
│  - Rare class oversampling      │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│      RF-DETR Training           │
│  - Hybrid encoder               │
│  - Deformable attention         │
│  - Set prediction loss          │
└─────────────────────────────────┘
       │
       ▼
   Trained Weights
```

### Inference Pipeline

```
Frame Input
       │
       ▼
┌─────────────────────────────────┐
│      Optional: SAHI Tiling      │
│  - Adaptive based on resolution │
│  - Small object boost mode      │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│       RF-DETR Detector          │
│  - End-to-end (no NMS)          │
│  - Query-based detection        │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│     BoT-SORT / ByteTrack        │
│  - Object tracking              │
│  - Movement classification      │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│    Landing Suitability          │
│  - Geometric analysis           │
│  - Segmentation-assisted        │
│  - Stability scoring            │
└─────────────────────────────────┘
       │
       ▼
   Final Detections
```

## Preferred Technology Stack

| Component | Primary | Alternative | Notes |
|-----------|---------|-------------|-------|
| Detector | RF-DETR-Base | RT-DETR-L | RF-DETR is newer |
| Tiling | SAHI | None | Training + inference |
| Tracker | BoT-SORT | ByteTrack | BoT-SORT has Re-ID |
| Landing | Geometric | SAM-based | SAM adds latency |
| Training | MMDetection | Detectron2 | Better DETR support |

## Data Strategy

### Dataset Aggregation

| Dataset | Size | Classes | Purpose |
|---------|------|---------|---------|
| VisDrone | 10K+ | 12 | Aerial drone view |
| UAVDT | 80K+ | 3 | Vehicle detection |
| DOTA | 2K+ | 15 | Oriented objects |
| xView | 1M+ | 60 | Satellite imagery |
| COWC | 32K+ | 1 | Vehicle counting |
| TEKNOFEST samples | ~5K | 3 | Competition domain |

### Augmentation Strategy

```python
augmentation_config = {
    # Spatial
    "mosaic_prob": 0.5,
    "mixup_prob": 0.3,
    "random_crop": {"prob": 0.5, "scale": (0.5, 1.0)},
    "rotation": {"prob": 0.3, "degrees": 15},
    
    # Weather simulation
    "rain": {"prob": 0.2, "intensity": (0.1, 0.5)},
    "fog": {"prob": 0.2, "density": (0.1, 0.4)},
    "snow": {"prob": 0.1, "intensity": (0.1, 0.3)},
    "night": {"prob": 0.15, "gamma": (0.3, 0.7)},
    
    # Copy-Paste for rare classes
    "copy_paste": {
        "prob": 0.3,
        "target_classes": ["airplane", "person"],  # Rare in aerial
    },
}
```

### UAP/UAI Synthetic Data

UAP (Unmanned Aerial Platform) and UAI (Unmanned Aerial Infrastructure) synthetic data:

```python
synthetic_generation_config = {
    "uap_models": [
        "DJI_Phantom_4",
        "DJI_Mavic_3",
        "Generic_Quadcopter",
    ],
    "environments": [
        "urban_daylight",
        "rural_overcast",
        "coastal_sunny",
        "mountain_cloudy",
    ],
    "altitudes_m": [10, 30, 50, 100, 200],
    "viewpoints": ["nadir", "oblique_30", "oblique_45"],
    "count_per_combo": 100,
}
```

## Implementation Plan

### Phase 1: RF-DETR Baseline

```python
# tools/train_rfdetr_baseline.py
from mmdet.apis import train_detector
from mmdet.models import build_detector

config = {
    "model": {
        "type": "RFDETR",
        "backbone": "ResNet50",
        "neck": "ChannelMapper",
        "bbox_head": {
            "type": "RFDETRHead",
            "num_classes": 3,  # vehicle, person, airplane
            "num_queries": 300,
        },
    },
    "train_cfg": {
        "epochs": 100,
        "batch_size": 8,
        "lr": 0.0001,
    },
}
```

### Phase 2: SAHI Integration

```python
# src/task1/experimental/sahi_detector.py
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class SAHIRFDETRDetector:
    def __init__(self, model_path: str, slice_size: int = 640):
        self.model = AutoDetectionModel.from_pretrained(
            model_type="rfdetr",
            model_path=model_path,
            confidence_threshold=0.25,
        )
        self.slice_size = slice_size
    
    def detect(self, image: np.ndarray) -> list[Detection]:
        result = get_sliced_prediction(
            image,
            self.model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        return self._convert_to_canonical(result)
```

### Phase 3: Tracker Integration

```python
# src/task1/experimental/botsort_tracker.py
from boxmot import BoTSORT

class Task1TrackerExperimental:
    def __init__(self):
        self.tracker = BoTSORT(
            reid_weights="osnet_x0_25_market1501.pt",
            device="cuda:0",
            half=True,
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=30,
            cmc_method="sparseOptFlow",
        )
    
    def update(self, detections: list, frame: np.ndarray) -> list:
        # Convert to tracker format
        dets = self._to_tracker_format(detections)
        # Update tracker
        tracks = self.tracker.update(dets, frame)
        # Classify movement
        return self._classify_movement(tracks)
```

### Phase 4: Landing Suitability

```python
# src/task1/experimental/landing_logic.py
class LandingSuitabilityAnalyzer:
    def analyze(self, detection: Detection, frame: np.ndarray) -> str:
        """
        Determine landing suitability based on:
        1. Object stability (motion blur, tracking confidence)
        2. Geometric properties (aspect ratio, area)
        3. Context (nearby objects, terrain)
        4. Optional: Segmentation mask quality
        """
        
        stability_score = self._compute_stability(detection)
        geometry_score = self._compute_geometry(detection)
        context_score = self._compute_context(detection, frame)
        
        total_score = (
            stability_score * 0.4 +
            geometry_score * 0.3 +
            context_score * 0.3
        )
        
        if total_score > 0.7:
            return "1"  # Suitable
        return "0"  # Not suitable
```

## Evaluation Framework

### Comparison Metrics

| Metric | Description | Weight |
|--------|-------------|--------|
| mAP@50 | Primary accuracy | 40% |
| mAP@50:95 | Strict accuracy | 20% |
| Small object AP | <32x32 pixels | 15% |
| Latency p95 | 95th percentile | 15% |
| VRAM usage | Peak GPU memory | 10% |

### Test Scenarios

| Scenario | Purpose |
|----------|---------|
| Standard daylight | Baseline performance |
| Dense traffic | Multi-object handling |
| Small objects | SAHI benefit |
| Weather degraded | Robustness |
| Thermal imagery | Cross-modality |
| Fast motion | Tracking quality |

## Success Criteria

### For Merge Consideration

| Metric | YOLO26n Baseline | RF-DETR Target | Required |
|--------|------------------|----------------|----------|
| mAP@50 | ~85% | >88% | +3% min |
| Small AP | ~60% | >70% | +10% min |
| Latency | ~15ms | <50ms | <3.3x slower |
| VRAM | ~1GB | <3GB | <3x more |

### Acceptable Trade-offs

- Latency up to 50ms (3x baseline) if mAP improves >5%
- VRAM up to 3GB if small object detection improves >15%
- Training time increase acceptable (offline)

## Dependencies

### Training Dependencies

```txt
# requirements-task1-experimental.txt
mmdet>=3.0.0
mmengine>=0.10.0
sahi>=0.11.0
albumentations>=1.3.0
boxmot>=10.0.0
```

### Inference Dependencies

```txt
torch>=2.0.0
onnxruntime-gpu>=1.17.0
```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| RF-DETR underperforms | Low | High | Fall back to RT-DETR |
| Latency budget exceeded | Medium | Medium | Use smaller model |
| Training data insufficient | Medium | Medium | Augmentation + synthetic |
| Integration complexity | Low | Low | Isolated branch |

## Decision Framework

| Decision | Criteria |
|----------|----------|
| **MERGE CANDIDATE** | >3% mAP improvement, <50ms latency |
| **PARTIAL MERGE** | Useful components (tracker, landing) only |
| **EXPERIMENTAL ONLY** | Interesting but not competition-ready |
| **NOT RECOMMENDED** | No improvement or regression |

---

*This branch explores higher-ceiling detection without risking production stability.*
