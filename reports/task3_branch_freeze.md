# Task 3 Experimental Branch Freeze Note

- Production Task 1 chain remains locked:
  `tensorrt:yolo26n -> onnxruntime:yolo26n -> ultralytics:yolo26n -> ultralytics:yolo11n -> synthetic`
- `onnxruntime:yolo11n` remains production-disabled.
- Production sequential runner and default wire profile stay unchanged.
- `draft_with_undefined` remains dormant and default-off.
- Experimental Task 3 code is config-gated and default-off.
- This branch must not alter the validated production path unless explicitly enabled for local evaluation.
