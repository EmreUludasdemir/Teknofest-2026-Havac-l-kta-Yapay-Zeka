# Phase 11: Operator Commands

**Date:** 2026-04-07  
**Purpose:** Minimal command surface for competition-day execution

## Quick Reference Card

### Competition Day Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `python tools/run_runtime_smoke.py --mode sequential` | **MAIN RUN** | Competition execution |
| `python tools/run_runtime_smoke.py --mode batch --frames 2` | **SMOKE TEST** | Pre-flight validation |
| `python -m pytest tests/test_sequential_interface.py -v` | **COMPAT CHECK** | Endpoint verification |

## Command #1: Main Run (Competition Execution)

```powershell
# Activate environment
C:\Users\Emre\.venvs\teknofest-gpu\Scripts\Activate.ps1

# Run sequential competition mode
python tools/run_runtime_smoke.py --mode sequential --frames 9999
```

**What it does:**
1. Loads model manifest from `final_runtime/config/model_manifest.json`
2. Initializes detector with TensorRT → ONNX → Ultralytics fallback chain
3. Connects to competition endpoint
4. Processes frames sequentially: fetch → detect → estimate → match → submit
5. Logs all activity to `final_runtime/logs/`

**Expected output:**
```
{"mode": "sequential", "diagnostics": [...]}
```

**If competition URL differs from mock:**
Edit `SequentialProtocolSettings` in the run script or create a config file.

## Command #2: Smoke Test (Pre-Flight)

```powershell
# Quick validation with mock server
python tools/run_runtime_smoke.py --mode batch --frames 2
```

**What it does:**
1. Starts local mock server
2. Processes 2 frames through full pipeline
3. Validates JSON schema compliance
4. Writes summary to `final_runtime/logs/runtime_smoke_summary.json`

**Success criteria:**
- Exit code 0
- No `ProtocolError` exceptions
- Summary JSON shows `status_code: 201` for predictions

## Command #3: Endpoint Compatibility Check

```powershell
# Verify adapter contracts
python -m pytest tests/test_sequential_interface.py tests/test_sequential_warmup_retry.py tests/test_sequential_mock_contract.py -v
```

**What it does:**
1. Tests adapter interface compliance
2. Verifies retry/timeout behavior
3. Checks warmup wait logic
4. Validates mock contract assumptions

**All tests must pass before competition.**

## Pre-Competition Checklist

```powershell
# 1. Verify Python environment
python --version  # Must be 3.10.x

# 2. Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Verify model manifest
python -c "import json; print(json.load(open('final_runtime/config/model_manifest.json'))['required'])"

# 4. Run smoke test
python tools/run_runtime_smoke.py --mode batch --frames 2

# 5. Run sequential test
python tools/run_runtime_smoke.py --mode sequential --frames 2

# 6. Check logs for errors
Get-Content final_runtime/logs/*.jsonl | Select-String "error"
```

## Troubleshooting

### Error: `ProtocolError: Sequential login basarisiz`

**Cause:** Wrong credentials or endpoint URL
**Fix:**
1. Verify `username` and `password` in settings
2. Check `base_url` points to correct endpoint
3. Test with: `curl -X POST {base_url}auth/ -d "username=X&password=Y"`

### Error: `ProtocolError: Sequential frame alinamadi: 404`

**Cause:** Wrong path for `next_frame` endpoint
**Fix:**
1. Update `path_overrides["next_frame"]` in settings
2. Verify endpoint path with competition docs

### Error: `TRT engine load failed`

**Cause:** TensorRT engine incompatible with current GPU
**Fix:** Runtime will auto-fallback to ONNX. Check `runtime_order` in `runtime.toml`.

### Error: Model not found

**Cause:** Model path in manifest doesn't exist
**Fix:**
1. Check `final_runtime/config/model_manifest.json`
2. Verify paths are absolute or relative to project root
3. Re-export models if needed

## Directory Structure

```
final_runtime/
├── config/
│   ├── runtime.toml          # Runtime configuration
│   └── model_manifest.json   # Model paths and validation status
├── artifacts/
│   ├── onnx/                 # ONNX models
│   └── trt/                  # TensorRT engines
├── logs/                     # Runtime logs (JSONL format)
└── cache/                    # Temporary cache
```

## Emergency Fallback

If everything fails, the runtime has a built-in synthetic fallback:

```python
# In runtime.toml, runtime_order ends with:
runtime_order = [
  "tensorrt:yolo26n",
  "onnxruntime:yolo26n", 
  "ultralytics:yolo26n",
  "onnxruntime:yolo11n",
  "ultralytics:yolo11n",
  "synthetic",  # <-- Last resort: returns empty detections
]
```

The synthetic backend ensures the runtime **never crashes** due to model issues—it will submit empty predictions rather than fail.

## Log Files

All logs are written to `final_runtime/logs/` in JSONL format:

| File Pattern | Content |
|--------------|---------|
| `runtime_*.jsonl` | Main runtime events |
| `error_*.jsonl` | Error events only |
| `runtime_smoke_summary.json` | Smoke test summary |

**Review logs with:**
```powershell
# All events
Get-Content final_runtime/logs/runtime_*.jsonl | ConvertFrom-Json

# Errors only
Get-Content final_runtime/logs/error_*.jsonl
```

---

*A teammate should be able to run the competition with just this document.*
