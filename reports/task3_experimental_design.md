# Task 3 Experimental Design

## Freeze Boundary
- Production Task 1 chain korunuyor.
- `onnxruntime:yolo11n` production-disabled kaliyor.
- Default sequential wire profile `official_current` olarak kaliyor.
- Task 3 experimental path config-gated ve default-off.

## Implemented Experimental Pipeline
- Detector path: `prompt_approx`
- Requested mode: `yoloe_prompted`
- Active tracker: `light`
- Active verifier: `orb_homography`
- Tiled inference enabled: `False`
- Multiscale enabled: `False`

## Outcome
- True YOLOE achieved: `False`
- Recommendation: `EXPERIMENTAL ONLY`
- Meaningful gain: `False`
