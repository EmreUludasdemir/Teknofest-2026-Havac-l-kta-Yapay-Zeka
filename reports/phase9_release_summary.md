# Phase 9 Release Summary

## Files Changed
- `src/config/settings.py`
- `src/task1/detector.py`
- `src/task2/estimator.py`
- `src/task3/reference_cache.py`
- `src/data/validators.py`
- `src/server/final_sequential_adapter.py`
- `src/tools/runtime_package.py`
- `src/tools/runtime_bootstrap.py`
- `tools/run_runtime_smoke.py`
- `tools/run_competition_runtime.py`
- `requirements-runtime.txt`
- `final_runtime/README.md`
- `final_runtime/CHECKLIST.md`
- `final_runtime/RUNBOOK.md`
- `final_runtime/config/runtime.toml`
- `final_runtime/config/model_manifest.json`
- `reports/export/task1_fallback_tree.json`
- `reports/export/task1_fallback_tree.md`
- `reports/export_readiness.md`

## Final Runtime Decision Tree
1. `tensorrt:yolo26n`
2. `onnxruntime:yolo26n`
3. `ultralytics:yolo26n`
4. `ultralytics:yolo11n`
5. `synthetic`

## yolo11n Karari
- `yolo11n` ONNX export dosyasi mevcut.
- `reports/export/task1_yolo11n_onnx_export_summary.json` icinde `validation_passed=false`.
- `reports/export/task1_yolo11n_native_vs_onnx.md` parity ve latency tarafinda kabul disi kaliyor.
- Son karar: `yolo11n` production ortaminda native-only fallback. `onnxruntime:yolo11n` production disabled.

## Warm-up Policy
- Warm-up `login -> open_session` sonrasinda ve ilk `fetch_next_frame` oncesinde yapilir.
- Warm edilen backend yalniz aktif secilen Task 1 stage'dir.
- Warm-up run sayilari:
  - TRT: `3`
  - ONNX: `1`
  - Native: `1`
- Warm-up stage fail ederse bir sonraki eligible stage denenir.
- Ilk gercek frame sirasinda backend fail ederse ayni frame icinde sonraki eligible stage denenir.
- Tum Task 1 stage'leri fail ederse Task 1 bos/guvenli fallback ile devam eder; Task 2 ve Task 3 durmaz.

## Sequential Contract Enforcement
- Production entrypoint: `tools/run_competition_runtime.py`
- Production path yalniz `FinalSequentialAdapter` kullanir.
- Sira sabitlenmistir:
  - `login`
  - `open_session`
  - warm-up / preload
  - `fetch_next_frame`
  - process
  - canonical validate
  - sequential wire validate
  - `send_wire_prediction`
  - sonra sonraki frame
- Sequential validator zorunlu alanlari ayri profilde kontrol eder:
  - `session_id`
  - `frame_id`
  - `frame`
  - `detected_objects`
  - `detected_translations`
- `detected_undefined_objects` canonical-only kalir; wire'a gonderilmez.

## Smoke / Test Commands
- `python -m unittest discover -s tests -v`
- `python tools/run_runtime_smoke.py --mode batch --frames 2`
- `python tools/run_runtime_smoke.py --mode sequential --frames 2`
- `python tools/run_competition_runtime.py --config final_runtime/config/runtime.toml --max-frames 2 --base-url <mock-url>`

## Measured Evidence
- Full test suite: `67 tests, OK`
- `reports/export/task1_native_vs_onnx_vs_trt.md`
  - Native P50: `60.8125 ms`
  - ONNX P50: `52.732 ms`
  - TRT P50: `45.524 ms`
  - TRT peak VRAM: `2821 MB`
  - TRT smoke accepted: `True`
- `reports/export/task1_yolo11n_native_vs_onnx.md`
  - Accepted: `False`
- `final_runtime/config/model_manifest.json`
  - `yolo11n.onnx`: `production_enabled=false`

## Remaining Risks
- Current local Python 3.12 smoke env CPU-only oldugu icin production smoke burada `synthetic` stage'e dustu; gercek competition laptopunda GPU runtime ile dogrulama tekrar alinmali.
- Sequential final server contract halen mock + official repo bilgisiyle dogrulaniyor; gercek final endpoint davranisi geldigi an son bir compatibility turu lazim.
- Task 2 thermal `health_status=0` kararliligi iyilesti ama halen competition-day replay uzerinde son uzun kosu tavsiye edilir.
- Cold-start maliyeti halen yuksek; operator runbook warm-up'i zorunlu tutar.
