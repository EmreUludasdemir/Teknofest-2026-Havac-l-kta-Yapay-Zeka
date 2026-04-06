# Phase 10 GPU Validation

- Python: `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe`
- Platform: `Windows-11-10.0.26200-SP0`
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`
- CUDA Visible: `True`
- ONNX Providers: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- TensorRT Version: `10.16.0.72`

## Production Run
- Active backend: `tensorrt:yolo26n`
- Active provider: `TensorRT`
- Warm-up attempted stages: `['tensorrt:yolo26n']`
- Warm-up failed stages: `[]`
- First frame latency ms: `39.8`
- Warm P50 ms: `64.0355`
- Warm P95 ms: `73.929`
- Peak VRAM MB: `2912.0`

## TRT Compatibility
- Rebuild attempted: `False`
- Mismatch reasons: `[]`

## Fallback Probes
- `trt_unavailable_probe` -> `onnxruntime:yolo26n` (CUDAExecutionProvider)
- `trt_onnx_unavailable_probe` -> `ultralytics:yolo26n` (cuda:0)

## Decision
- GPU path exercised: `True`
- First-frame regression: `False`
- Accepted: `True`
- No-go reason: `None`
