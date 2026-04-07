# Score-Up Priority Summary

**Date:** 2026-04-07  
**Purpose:** Final recommendations for competition score optimization

---

## 1. Which Branch Should Be Attempted First?

### **Answer: `feature/task3-yoloe-real`**

**Rationale:**
- Task 3 has the highest upside - current ORB baseline is weak
- YOLOE weights are NOW AVAILABLE (Ultralytics v8.4.0+)
- Blocker resolved - can proceed immediately
- Lower integration risk than Task 2 architectural change
- Task 1 is already strong, less room for improvement

---

## 2. What Exact Blocker Exists?

### Task 3 YOLOE: ✅ **UNBLOCKED**

| Previous Blocker | Status | Resolution |
|------------------|--------|------------|
| No YOLOE weight | ✅ RESOLVED | Ultralytics v8.4.0 provides official weights |
| Unknown YOLOE API | ✅ RESOLVED | Documented in Ultralytics docs |

**Available weights:**
- `yoloe-26l-seg.pt` (recommended)
- `yoloe-11l-seg.pt` (alternative)
- `yoloe-26s-seg.pt` (faster, less accurate)

### Task 1 RF-DETR: ✅ **UNBLOCKED**

| Blocker | Status |
|---------|--------|
| RF-DETR availability | ✅ Available via mmdet/transformers |
| Training infrastructure | ⚠️ Needs setup |
| Data aggregation | ⚠️ Needs curation |

### Task 2 DPVO: ⚠️ **CONDITIONALLY BLOCKED**

| Blocker | Status |
|---------|--------|
| DPVO package | ✅ Available |
| Scale recovery | ⚠️ Research needed |
| Thermal support | ⚠️ Unknown |
| Integration complexity | ⚠️ High |

**Condition:** Do not start until Task 3 evaluation completes.

---

## 3. What Model Component Should Be Developed Next?

### Immediate Priority Order:

| Priority | Component | Branch | Effort | Expected ROI |
|----------|-----------|--------|--------|--------------|
| **1** | YOLOE Visual Prompt Matcher | task3-yoloe-real | 6 days | HIGH |
| **2** | RF-DETR + SAHI Detector | task1-rfdetr-sahi | 10 days | MEDIUM |
| **3** | DPVO Estimator (conditional) | task2-dpvslam | 14 days | UNKNOWN |
| **4** | Shared Data Engine | data-engine | Ongoing | HIGH (long-term) |

### Next Development Steps:

```
Week 1-2: Task 3 YOLOE
├── Day 1: Stage weights, validate load
├── Day 2: Reference embedding cache
├── Day 3-4: Detection pipeline with VPE
├── Day 5: Tracker integration (BoT-SORT)
└── Day 6: Evaluation and recommendation

Week 3-4: Task 1 RF-DETR (if Task 3 unblocks time)
├── Training data aggregation
├── SAHI-aware training setup
├── RF-DETR baseline training
└── Comparative evaluation

Week 5+: Task 2 DPVO (only if Task 3 shows promise)
```

---

## 4. What Should Explicitly NOT Be Touched?

### ❌ DO NOT TOUCH

| Component | Reason |
|-----------|--------|
| `src/task2/estimator.py` | Production frozen |
| `src/task2/drift_control.py` | Production frozen |
| `src/task2/health_logic.py` | Production frozen |
| Task 2 thermal knob settings | Rejected, proven ineffective |
| `feature/task2-health0-drift-crush` | Rejected branch |
| `feature/task2-thermal-only-microlever` | Rejected branch |
| Production `main` branch core logic | Validated, stable |
| Runtime.toml production defaults | Competition-ready |

### ⚠️ TOUCH WITH EXTREME CARE

| Component | Condition |
|-----------|-----------|
| `src/task1/detector.py` | Only via isolated experimental branch |
| `src/task3/matcher.py` | Only via isolated experimental branch |
| `requirements.txt` | Experimental deps in separate file |

### ✅ SAFE TO MODIFY

| Component | Purpose |
|-----------|---------|
| `reports/*.md` | Documentation |
| `src/task*/experimental/` | Isolated experiments |
| `tools/*_experimental_*.py` | Experimental runners |
| `tests/test_*_experimental_*.py` | Experimental tests |

---

## 5. Merge Recommendation for Each Branch

### Task 3: `feature/task3-yoloe-real`

| Scenario | Recommendation |
|----------|----------------|
| >10% accuracy improvement, <100ms latency | **MERGE CANDIDATE** |
| Useful embedding infrastructure only | **PARTIAL MERGE** |
| No improvement over ORB baseline | **EXPERIMENTAL ONLY** |
| Weight loading fails | **BLOCKED_BY_WEIGHT** |

**Expected outcome:** MERGE CANDIDATE or PARTIAL MERGE (high confidence)

### Task 1: `feature/task1-rfdetr-sahi`

| Scenario | Recommendation |
|----------|----------------|
| >3% mAP improvement, <50ms latency | **MERGE CANDIDATE** |
| Tracker/landing logic useful | **PARTIAL MERGE** |
| No improvement over YOLO26n | **NOT RECOMMENDED** |

**Expected outcome:** PARTIAL MERGE (medium confidence)

### Task 2: `feature/task2-dpvslam-feasibility`

| Scenario | Recommendation |
|----------|----------------|
| >13% drift improvement, no regressions | **CONTINUE DEVELOPMENT** |
| Scale recovery method useful | **PARTIAL: Scale only** |
| No improvement | **STOP IMMEDIATELY** |

**Expected outcome:** STOP or PARTIAL (low confidence in full success)

---

## Summary Decision Matrix

| Task | Branch | Status | Priority | Expected ROI | Recommendation |
|------|--------|--------|----------|--------------|----------------|
| **Task 3** | task3-yoloe-real | UNBLOCKED | #1 | HIGH | START NOW |
| **Task 1** | task1-rfdetr-sahi | UNBLOCKED | #2 | MEDIUM | START AFTER Task 3 |
| **Task 2** | task2-dpvslam | CONDITIONAL | #3 | UNKNOWN | WAIT for Task 3 results |
| **Data** | data-engine | ONGOING | #4 | HIGH (long-term) | PARALLEL with above |

---

## Action Items

### Immediate (Today)

1. ✅ Create score-up design documents (DONE)
2. 🔲 Create `feature/task3-yoloe-real` branch
3. 🔲 Stage YOLOE weights via ultralytics
4. 🔲 Validate YOLOE visual prompt API

### This Week

1. 🔲 Implement YOLOE reference embedding cache
2. 🔲 Implement YOLOE detection pipeline
3. 🔲 Integrate BoT-SORT tracker
4. 🔲 Run evaluation vs ORB baseline
5. 🔲 Write merge recommendation

### If Task 3 Succeeds

1. 🔲 Prepare merge candidate PR
2. 🔲 Begin Task 1 RF-DETR work
3. 🔲 Re-evaluate Task 2 DPVO priority

### If Task 3 Fails

1. 🔲 Document learnings
2. 🔲 Archive branch
3. 🔲 Accelerate Task 1 RF-DETR
4. 🔲 Lower Task 2 priority further

---

## Competition Score Optimization Summary

**Optimize for competition score, not elegance.**

| Priority | Action | Score Impact |
|----------|--------|--------------|
| 1 | Task 3 YOLOE | HIGH - Baseline is weak |
| 2 | Task 1 RF-DETR | MEDIUM - Already strong |
| 3 | Task 2 DPVO | LOW - High risk, frozen baseline works |
| 4 | Data engine | LONG-TERM - Foundation for future |

**Always preserve the validated production runtime.**

---

*This summary guides score-up work in priority order while protecting production stability.*
