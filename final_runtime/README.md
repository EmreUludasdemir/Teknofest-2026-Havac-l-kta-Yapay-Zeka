# Final Runtime

Bu klasor yarismada kullanilacak production runtime paketinin yerel iskeletidir.

## Klasor Yapisi
- `config/runtime.toml`: tek production config kaynagi
- `config/model_manifest.json`: model/artifact ve production eligibility kararlari
- `artifacts/onnx/`: ONNX dosyalari
- `artifacts/trt/`: TensorRT engine ve metadata
- `logs/`: runtime loglari
- `cache/`: gecici yerel cache

## Production Task 1 Karar Agaci
- `tensorrt:yolo26n`
- `onnxruntime:yolo26n`
- `ultralytics:yolo26n`
- `ultralytics:yolo11n`
- `synthetic`

## Onemli Notlar
- `onnxruntime:yolo11n` export edilmis olsa da parity basarisiz oldugu icin production disabled durumundadir.
- Varsayilan sequential wire profili `official_current` olup Task 3 undefined object sonuclari canonical modelde tutulur, production wire'a gonderilmez.
- Teknik sartname taslagi undefined object isterse `final_runtime/config/runtime.toml` icinde `sequential.wire_profile = "draft_with_undefined"` acilabilir.
- Ilk gercek frame alinmadan once warm-up tamamlanmis olmalidir.

## Model Beklentileri
### Manifest
- required:
  - yolo26n.pt | runtime=ultralytics | candidate=yolo26n | validated=True | production_enabled=True
  - yolo26n.onnx | runtime=onnxruntime | candidate=yolo26n | validated=True | production_enabled=True
- optional:
  - yolo26n.engine | runtime=tensorrt | candidate=yolo26n | validated=True | production_enabled=True
  - yolo11n.pt | runtime=ultralytics | candidate=yolo11n | validated=True | production_enabled=True
  - yolo11n.onnx | runtime=onnxruntime | candidate=yolo11n | validated=False | production_enabled=False

## Baslatma
- Production: `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools/run_competition_runtime.py --config final_runtime/config/runtime.toml`
- Batch smoke: `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools/run_runtime_smoke.py --mode batch --frames 2`
- Sequential smoke: `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools/run_runtime_smoke.py --mode sequential --frames 2`
