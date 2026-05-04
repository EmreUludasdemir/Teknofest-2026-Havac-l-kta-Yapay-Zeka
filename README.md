# TEKNOFEST Task 3 Repo

Bu repo TEKNOFEST 2026 Havacilikta Yapay Zeka Yarismasi `Gorev 3 - Goruntu Esleme` icin gelistirme platformudur.

## Kapsam

- Task 3 production code: `src/task3/`
- Batch ve sequential adapterlar: `src/server/`
- Mock server ve replay altyapisi: `src/tools/mock_server.py`, `src/pipeline/`
- 12-ref mixed bank ve manifestler: `data/references/2026_baseline/`, `data/task3_eval_manifest.json`
- D5-D10 diagnostic archive ve probe scriptleri

Task 1 ve Task 2 bu repodan cikarilmistir. O gorevlerin gelistirmesi ekipteki diger repo'larda surdurulur ve yarismadan once ayrica entegre edilir.

## Hizli Baslangic

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## GPU Runtime

Task 3 YOLOE + LightGlue GPU olcumleri `.venv` icinde `torch 2.9.1+cu128` ile alinmistir.

```powershell
.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Ana Komutlar

```powershell
py -3.12 -m unittest discover -s tests -v
py -3.12 tools\diagnostic\d10_wire_audit.py
.venv\Scripts\python.exe tools\diagnostic\d10_post_cleanup_pass_a.py
```

## Notlar

- Batch adapter resmi `2025` wire formatina hizalanmistir: `frame`, `detected_objects`, `detected_translations`
- Canonical model Task 3 `detected_undefined_objects` alanini icerir; batch wire bu alani gondermez
- `ref_07` routing override ve `ref_12` manifest promotion aktif durumdadir
