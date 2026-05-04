# Setup

Bu repo artik yalnizca `Task 3 - Goruntu Esleme` icin kullanilir. Hedef gelistirme runtime'i `Python 3.12` ve GPU olcum runtime'i `.venv + torch 2.9.1+cu128` kombinasyonudur.

## Windows 11 hizli kurulum

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## GPU runtime

```powershell
.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Calistirma notlari

- Batch adapter resmi 2025 wire formatina gore gonderim yapar.
- Task 3 canonical payload `detected_undefined_objects` alanini tutar; batch wire bu alani omit eder.
- `ultralytics` Task 3 YOLOE backend icin gerekir.
- `timm` Task 3 learned descriptor baseline'i icin korunmustur; Task 1 bagimliligi degildir.
- Resmi sample videolari ve `data/references/2026_baseline/` bank'i cleanup sonrasi korunmustur.
