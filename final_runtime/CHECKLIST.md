# Competition Checklist

## Environment Preflight
- Aktif interpreter `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe` mi?
- Python 3.10 aktif mi?
- GPU driver ve CUDA gorunuyor mu?
- `onnxruntime-gpu` ve TensorRT ortamda kurulu mu?
- `final_runtime/config/runtime.toml` duzgun mu?

## Dosya Kontrolu
- [x] yolo26n.pt (ultralytics, production_enabled=True)
- [x] yolo26n.onnx (onnxruntime, production_enabled=True)
- [x] yolo26n.engine (tensorrt, production_enabled=True)
- [x] yolo11n.pt (ultralytics, production_enabled=True)
- [x] yolo11n.onnx (onnxruntime, production_enabled=False)

## Warm-up Kontrolu
- Session acildiktan sonra warm-up loglari olusuyor mu?
- `warmup_stage_completed` eventi goruluyor mu?
- Ilk frame warm-up bitmeden istenmiyor mu?

## Zorunlu Son Smoke
- `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools/run_runtime_smoke.py --mode sequential --frames 2`
- `competition_runtime_summary.json` icinde aktif stage production zincirinden biri mi?

## Sequential Wire Profili
- Varsayilan: `official_current`
- Draft undefined-object profili yalniz final endpoint bunu zorunlu kilarsa acilsin.
- Draft profile acilirsa payload icine `detected_undefined_objects` girdigi validator ile dogrulansin.

## Failure Mode Kontrolu
- TRT fail ederse ONNX yolo26n devreye giriyor mu?
- ONNX fail ederse native yolo26n devreye giriyor mu?
- yolo26n tamamen fail ederse native yolo11n devreye giriyor mu?
- Son fallback olarak synthetic aktif mi?
