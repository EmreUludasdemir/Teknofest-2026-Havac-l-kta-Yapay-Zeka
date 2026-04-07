# Task 3 Real YOLOE Score-Up Design

**Date:** 2026-04-07  
**Branch:** `feature/task3-yoloe-real`  
**Priority:** #1 (Highest upside)  
**Status:** UNBLOCKED - Weights available via Ultralytics

## Executive Summary

Task 3 currently uses an ORB/template matching baseline. YOLOE (YOLO with Embeddings) is now available in Ultralytics v8.4.0+ with open-vocabulary detection and visual prompting capabilities. This represents a **genuine architectural upgrade** rather than threshold tuning.

## Blocker Resolution

| Previous Blocker | Resolution |
|------------------|------------|
| No YOLOE weight staged | Ultralytics v8.4.0+ provides official YOLOE weights |
| YOLOE-v8-L unavailable | `yoloe-11l-seg.pt` available (YOLO11 architecture) |
| YOLOE-26-L unavailable | `yoloe-26l-seg.pt` available (YOLO26 architecture) |

### Available YOLOE Weights

| Model | Weight File | Architecture | Task |
|-------|-------------|--------------|------|
| YOLOE-11S | `yoloe-11s-seg.pt` | YOLO11 | Detection + Segmentation |
| YOLOE-11M | `yoloe-11m-seg.pt` | YOLO11 | Detection + Segmentation |
| YOLOE-11L | `yoloe-11l-seg.pt` | YOLO11 | Detection + Segmentation |
| YOLOE-26S | `yoloe-26s-seg.pt` | YOLO26 | Detection + Segmentation |
| YOLOE-26M | `yoloe-26m-seg.pt` | YOLO26 | Detection + Segmentation |
| YOLOE-26L | `yoloe-26l-seg.pt` | YOLO26 | Detection + Segmentation |

**Recommended:** `yoloe-26l-seg.pt` (YOLO26 architecture, best accuracy)

## Proposed Architecture

### Pipeline Flow

```
Reference Images (offline)
         │
         ▼
┌─────────────────────────────────────┐
│  YOLOE Visual Prompt Encoder (VPE)  │
│  - SAVPE: Semantic-Activated VPE    │
│  - Embed reference object crops     │
└─────────────────────────────────────┘
         │
         ▼
   Reference Embeddings Cache
         │
         │
Frame Input ──────────────────────────►
         │
         ▼
┌─────────────────────────────────────┐
│         YOLOE Detector              │
│  - Open-vocabulary detection        │
│  - Visual prompt conditioning       │
│  - Produces embeddings + boxes      │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       Embedding Matcher             │
│  - Cosine similarity vs refs        │
│  - Threshold: 0.05-0.15 (low)       │
│  - Multi-match disambiguation       │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        BoT-SORT Tracker             │
│  - Cross-frame association          │
│  - Camera motion compensation (CMC) │
│  - Re-ID feature integration        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│          Match Verifier             │
│  - Geometric consistency check      │
│  - Temporal stability filter        │
│  - Confidence aggregation           │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       Re-detect Policy              │
│  - Confidence decay handling        │
│  - Occlusion recovery               │
│  - SAHI for small objects           │
└─────────────────────────────────────┘
         │
         ▼
   Matched Undefined Objects
```

## Preferred Technology Stack

### Core Models

| Component | Option 1 (Recommended) | Option 2 | Notes |
|-----------|------------------------|----------|-------|
| Detector | YOLOE-26L | YOLOE-11L | 26 is newer, edge-optimized |
| Visual Encoder | SAVPE (built-in) | CLIP | SAVPE is YOLOE-native |
| Feature Matcher | LightGlue + SuperPoint | LoFTR | LightGlue is faster |
| Tracker | BoT-SORT w/ CMC | ByteTrack | BoT-SORT has Re-ID |
| Tiling | SAHI | None | For small object boost |

### Auxiliary Components

| Component | Purpose | Integration |
|-----------|---------|-------------|
| SAMURAI Tiny | Mask refinement | Optional, post-detection |
| SuperPoint | Keypoint extraction | For geometric verification |
| LightGlue | Point matching | Alternative to embedding similarity |

## Configuration Design

### Visual Prompt Settings

```python
@dataclass
class YoloeTask3Settings:
    # Model selection
    yoloe_model: str = "yoloe-26l-seg.pt"
    yoloe_device: str = "cuda:0"
    
    # Visual prompt confidence
    vpe_confidence_min: float = 0.05   # Very low - cast wide net
    vpe_confidence_max: float = 0.15   # Still low for open-vocab
    vpe_embedding_dim: int = 512
    
    # Matching thresholds
    embedding_similarity_min: float = 0.60
    embedding_ambiguity_gap: float = 0.10
    geometric_verification: bool = True
    
    # Tracker settings
    tracker_type: str = "botsort"      # or "bytetrack"
    tracker_reid_weight: float = 0.3
    tracker_cmc_enabled: bool = True
    
    # SAHI tiling
    sahi_enabled: bool = True
    sahi_slice_height: int = 640
    sahi_slice_width: int = 640
    sahi_overlap_ratio: float = 0.2
    
    # Re-detect policy
    redetect_on_confidence_drop: bool = True
    redetect_confidence_threshold: float = 0.30
    redetect_max_frames: int = 10
```

### Low Confidence Range Rationale

The 0.05-0.15 confidence range is intentionally low because:

1. **Open-vocabulary detection is harder** - Novel objects won't have COCO-trained high confidence
2. **Visual prompts are approximate** - Reference images may differ from in-frame appearance
3. **We verify downstream** - Embedding matching + tracking provides secondary validation
4. **Recall > Precision** - Better to detect-then-verify than miss entirely

## Implementation Plan

### Phase 1: Weight Staging & Validation

```python
# tools/stage_yoloe_weights.py
from ultralytics import YOLOE

def stage_yoloe_weights():
    """Download and validate YOLOE weights."""
    model_path = Path.home() / ".teknofest_models" / "yoloe"
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Download via ultralytics (auto-caches)
    model = YOLOE("yoloe-26l-seg.pt")
    
    # Validate visual prompt capability
    reference_img = load_reference_image()
    model.set_visual_prompt(reference_img)
    results = model.predict(test_frame)
    
    return model_path / "yoloe-26l-seg.pt"
```

### Phase 2: Reference Embedding Cache

```python
# src/task3/experimental/yoloe_cache.py
@dataclass
class YoloeReferenceCache:
    model: Any
    embeddings: dict[str, np.ndarray]
    
    def preload_references(self, reference_dir: Path) -> None:
        """Encode all reference images into YOLOE embeddings."""
        for ref_path in reference_dir.glob("*.png"):
            ref_id = ref_path.stem
            embedding = self._encode_reference(ref_path)
            self.embeddings[ref_id] = embedding
    
    def _encode_reference(self, image_path: Path) -> np.ndarray:
        """Extract SAVPE embedding from reference image."""
        img = cv2.imread(str(image_path))
        # Use YOLOE's visual prompt encoder
        embedding = self.model.encode_visual_prompt(img)
        return embedding / np.linalg.norm(embedding)
```

### Phase 3: Frame Processing Pipeline

```python
# src/task3/experimental/yoloe_matcher.py
class YoloeTask3Matcher:
    def match(self, frame: FrameEnvelope, decoded: DecodedFrame) -> list[CanonicalUndefinedObject]:
        # 1. Run YOLOE detection with visual prompts
        detections = self._detect_with_prompts(decoded)
        
        # 2. Extract embeddings for each detection
        det_embeddings = self._extract_embeddings(decoded, detections)
        
        # 3. Match against reference cache
        matches = self._match_embeddings(det_embeddings)
        
        # 4. Apply tracker for temporal consistency
        tracked = self.tracker.update(matches, decoded)
        
        # 5. Verify with geometric checks
        verified = self._verify_matches(tracked, decoded)
        
        # 6. Apply re-detect policy if needed
        final = self._apply_redetect_policy(verified)
        
        return final
```

### Phase 4: Evaluation Harness

```python
# src/evaluation/task3_yoloe_eval.py
def evaluate_yoloe_vs_baseline(
    manifest_path: Path,
    output_dir: Path,
) -> dict:
    """Compare YOLOE pipeline against ORB baseline."""
    
    baseline_matcher = Task3Matcher(...)  # ORB-based
    yoloe_matcher = YoloeTask3Matcher(...)  # YOLOE-based
    
    results = {
        "baseline": [],
        "yoloe": [],
    }
    
    for scenario in load_manifest(manifest_path):
        baseline_result = evaluate_scenario(baseline_matcher, scenario)
        yoloe_result = evaluate_scenario(yoloe_matcher, scenario)
        
        results["baseline"].append(baseline_result)
        results["yoloe"].append(yoloe_result)
    
    return compute_comparison_metrics(results)
```

## Success Criteria

### Minimum Threshold for Merge

| Metric | ORB Baseline | YOLOE Target | Required Delta |
|--------|--------------|--------------|----------------|
| Match accuracy | Current | Must improve | >10% |
| False positive rate | Current | Must not increase | No regression |
| Small object detection | Weak | Should improve | >15% |
| Cross-modality (thermal) | Limited | Should improve | Any improvement |
| Latency | <50ms | <100ms | Acceptable if 2x |

### Acceptable Trade-offs

- **Latency increase up to 2x** is acceptable if accuracy improves >10%
- **VRAM increase** to 4GB (from 2GB) is acceptable
- **Additional dependencies** (ultralytics update) is acceptable

### Unacceptable Outcomes

- Any regression in false positive rate
- Latency >200ms per frame
- VRAM >6GB
- Unstable results across runs

## Fallback Strategy

If YOLOE does not meet criteria:

1. **Partial Merge:** Cherry-pick only embedding cache infrastructure
2. **Hybrid Mode:** Use YOLOE for hard cases, ORB for easy cases
3. **Archive:** Document learnings, keep branch for future

## Dependencies

### Required Package Updates

```txt
# requirements-experimental.txt
ultralytics>=8.4.0    # YOLOE support
supervision>=0.20.0   # Detection utilities
lap                   # BoT-SORT dependency
filterpy              # Kalman filter for tracking
```

### Optional Packages

```txt
sahi>=0.11.0          # Sliced inference
lightglue             # Feature matching (if using)
kornia                # Geometric transforms
```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| YOLOE VPE quality insufficient | Medium | High | Fall back to embedding similarity |
| Latency exceeds budget | Medium | Medium | Use smaller model (11S) |
| Thermal modality issues | High | Medium | Fine-tune or skip thermal |
| Integration complexity | Low | Medium | Isolated experimental branch |

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Weight staging | 1 day | Validated YOLOE load |
| Reference cache | 1 day | Embedding precomputation |
| Detection pipeline | 2 days | Working YOLOE matcher |
| Tracker integration | 1 day | BoT-SORT with CMC |
| Evaluation | 1 day | Comparison report |
| **Total** | **6 days** | Merge recommendation |

## Decision Framework

After evaluation, decision must be one of:

| Decision | Criteria |
|----------|----------|
| **MERGE CANDIDATE** | >10% accuracy improvement, no regressions, <100ms latency |
| **PARTIAL MERGE** | Useful components but not full pipeline |
| **EXPERIMENTAL ONLY** | Interesting but not competition-ready |
| **BLOCKED_BY_WEIGHT** | Weight issues prevent evaluation |

---

*This design enables Task 3 score-up work to proceed with real YOLOE capabilities.*
