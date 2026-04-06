# Operator Drill

## Hazirlik
1. GPU venv aktif: `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe`
2. `requirements-runtime.txt` yuklu.
3. `final_runtime/config/model_manifest.json` icindeki required artifact'lar mevcut.

## Drill Adimlari
1. `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools/run_runtime_smoke.py --mode sequential --frames 2`
2. `C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe tools/run_competition_runtime.py --config final_runtime/config/runtime.toml --max-frames 2 --base-url <mock-url>`
3. Loglarda `warmup_policy_started`, `warmup_stage_completed`, `sequential_prediction_sent` eventlerini dogrula.
4. `competition_runtime_summary.json` icinde `active_task1_stage` alanini kontrol et.
5. Prediction fail olursa bir sonraki frame istenmedigini logdan teyit et.
6. Aktif stage `tensorrt:yolo26n`, `onnxruntime:yolo26n` veya `ultralytics:yolo26n` degilse production oturumunu baslatma.

## Durdur / Yeniden Baslat Kurali
- Aktif stage `synthetic` veya `ultralytics:yolo11n` ise production oturumu durdur.
- Warm-up tamamlanmadiysa oturumu yeniden baslat.
- Prediction retry limiti biterse logu sakla ve yeni oturum ac.
