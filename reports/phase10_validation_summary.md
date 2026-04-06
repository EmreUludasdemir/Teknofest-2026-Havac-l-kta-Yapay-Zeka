# Phase 10 Validation Summary

## Files Changed
- `src/exports/trt_bridge.py`
- `src/tools/runtime_package.py`
- `src/config/settings.py`
- `src/server/final_sequential_adapter.py`
- `src/data/validators.py`
- `src/evaluation/task2_long_sequence.py`
- `src/tools/runtime_bootstrap.py`
- `tools/run_runtime_smoke.py`
- `tools/run_competition_runtime.py`
- `tools/run_phase10_gpu_validation.py`
- `tools/run_phase10_operator_drill.py`
- `tools/run_phase10_task2_dress_rehearsal.py`
- `final_runtime/README.md`
- `final_runtime/CHECKLIST.md`
- `final_runtime/RUNBOOK.md`
- `final_runtime/SEQUENTIAL_COMPATIBILITY.md`
- `final_runtime/OPERATOR_DRILL.md`
- `final_runtime/config/runtime.toml`
- `final_runtime/config/model_manifest.json`

## Machine Used
- Date: `2026-04-05`
- Python: `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe`
- Platform: `Windows-11-10.0.26200-SP0`
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`
- Total VRAM: `8151 MB`
- Torch: `2.11.0+cu128`
- ONNX Runtime: `1.24.4`
- TensorRT: `10.16.0.72`

## Final Runtime Decision Tree
1. `tensorrt:yolo26n`
2. `onnxruntime:yolo26n`
3. `ultralytics:yolo26n`
4. `ultralytics:yolo11n`
5. `synthetic`

## yolo11n Decision
- `onnxruntime:yolo11n` exported but parity-failed.
- Production status: `disabled`
- Retained fallback: `ultralytics:yolo11n`

## GPU Validation Evidence
- Active backend actually exercised: `tensorrt:yolo26n`
- Active provider: `TensorRT`
- Warm-up attempted stages: `['tensorrt:yolo26n']`
- Warm-up failed stages: `[]`
- Warm-up runs: `3`
- First real frame latency: `39.8 ms`
- Warm `P50`: `64.0355 ms`
- Warm `P95`: `73.929 ms`
- Peak VRAM: `2912.0 MB`
- TRT rebuild attempted: `False`
- TRT metadata mismatch reasons: `[]`

Fallback probes:
- `tensorrt:yolo26n` disabled -> `onnxruntime:yolo26n` on `CUDAExecutionProvider`
- `tensorrt:yolo26n` + `onnxruntime:yolo26n` disabled -> `ultralytics:yolo26n` on `cuda:0`

## Warm-up Policy
- Warm-up runs after `login -> open_session` and before the first real frame request.
- Only the active selected Task 1 backend is warmed.
- Current warm-up counts:
  - TensorRT: `3`
  - ONNX: `1`
  - Native: `1`
- If warm-up fails, runtime falls through to the next eligible stage.
- If first-frame inference fails, the same frame is retried on the next eligible stage before requesting another frame.

## Sequential Compatibility Status
- Proven:
  - Production runner is sequential-only.
  - `open_session -> warm-up -> fetch_next_frame -> send_wire_prediction -> fetch_next_frame` ordering is enforced by tests.
  - Default wire profile is `official_current`.
  - Default sequential wire omits `detected_undefined_objects`.
- Preserved but dormant:
  - `draft_with_undefined` wire profile can emit `detected_undefined_objects` if the final endpoint requires it.
- Assumed, not proven:
  - Final unpublished sequential endpoint names and exact top-level payload contract.
- Official compatibility reference snapshot checked:
  - repo commit `ce249fc8042ada788df438d2c0a1b76e28d6beaa`
  - publicly visible proven endpoints remain batch-shaped in that snapshot

## Task 2 Dress Rehearsal
- Conclusion: `stability_and_observability`
- Baseline drift: `38.015056`
- Rehearsal drift: `15.514174`
- Baseline recovery: `0.0`
- Rehearsal recovery: `0.0`
- Thermal focus drift: `20.001263`
- Thermal focus continuity: `0.14697`
- Threshold changes: `none`

Interpretation:
- Phase 9 thermal guards improved real stability, not only logging.
- No new Task 2 redesign is justified for this phase.

## Operator Drill
- Sequential operator drill completed successfully.
- Runtime return code: `0`
- Active stage during drill: `tensorrt:yolo26n`
- Wire profile during drill: `official_current`
- Expected runtime events observed:
  - `warmup_policy_started`
  - `warmup_stage_completed`
  - `sequential_prediction_sent`

## Commands Run
- `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe -m unittest discover -s tests -v`
- `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools\run_runtime_smoke.py --mode sequential --frames 2`
- `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools\run_phase10_gpu_validation.py --config final_runtime/config/runtime.toml --max-frames 10`
- `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools\run_phase10_operator_drill.py --config final_runtime/config/runtime.toml --max-frames 2`
- `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools\run_phase10_task2_dress_rehearsal.py --output-dir reports --sequence-limit 600`

## GO / NO-GO
- Decision: `GO`

Reason:
- A real GPU-backed production stage was actually exercised.
- Warm-up and first-frame behavior were measured and are within the acceptance rule.
- Sequential ordering and wire safety are covered by tests and drill.
- Task 2 degraded replay shows no regression and improved stability.
- Operator-facing runtime package is usable from one clear entrypoint.

## Remaining Risks
- Final unpublished sequential endpoint is still not fully proven; current adapter stays aligned to the latest accessible reference plus mock assumptions.
- TensorRT currently warns about default stream synchronization overhead; correctness is fine, but there may still be a small performance headroom.
- `yolo11n` remains native-only fallback; ONNX parity is still unresolved and intentionally disabled in production.
