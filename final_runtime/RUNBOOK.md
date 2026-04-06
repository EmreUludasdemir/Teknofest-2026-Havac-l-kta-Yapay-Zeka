# Competition Day Runbook

1. `requirements-runtime.txt` ile ortam hazirligini tamamla.
2. `final_runtime/config/runtime.toml` icinde base_url, kullanici ve sifreyi doldur.
3. Model dosyalarinin `model_manifest.json` ile uyumlu oldugunu kontrol et.
4. `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools/run_runtime_smoke.py --mode sequential --frames 2` ile son yerel smoke'u kos.
5. Production komutu: `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools/run_competition_runtime.py --config final_runtime/config/runtime.toml`
6. Loglarda su eventleri gor: `warmup_policy_started`, `warmup_stage_completed`, `sequential_prediction_sent`.
7. Ilk gercek frame fail ederse ayni frame icin bir sonraki eligible backend otomatik denenir.
8. Prediction retry limiti dolarsa bir sonraki frame istenmez; oturum kontrollu kapatilir.

## Sequential Wire Notu
- Varsayilan wire profili `official_current` olarak kalir.
- Final endpoint `detected_undefined_objects` isterse `draft_with_undefined` profilini acip validator ve mock ile tekrar smoke alin.

## Native-only Fallback Karari
- `yolo11n` ONNX parity kanitlanmadigi icin production kullanimi kapatildi.
- `yolo11n` yalniz native fallback olarak kullanilir.

## Required / Optional Ozet
### Manifest
- required:
  - yolo26n.pt | runtime=ultralytics | candidate=yolo26n | validated=True | production_enabled=True
  - yolo26n.onnx | runtime=onnxruntime | candidate=yolo26n | validated=True | production_enabled=True
- optional:
  - yolo26n.engine | runtime=tensorrt | candidate=yolo26n | validated=True | production_enabled=True
  - yolo11n.pt | runtime=ultralytics | candidate=yolo11n | validated=True | production_enabled=True
  - yolo11n.onnx | runtime=onnxruntime | candidate=yolo11n | validated=False | production_enabled=False
