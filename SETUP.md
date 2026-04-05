# Setup Notu

Bu iskelet Faz 2.5 itibariyla `Python 3.10.x` hedeflenerek tutulmaktadir.

## Windows 11 hizli kurulum

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Notlar

- Bu fazda egitim baslatilmaz.
- Bu fazda ONNX/TRT export eklenmez.
- Faz 5 itibariyla hafif CV baseline icin `opencv-python-headless` ve Task 1 gercek local-model profiling icin `ultralytics` eklenmistir.
- Task 1 gercek baseline agirliklari repo icine konmaz; `settings` veya CLI ile repo disi local path verilir.
- RT-DETR, ONNXRuntime ve benzeri alternatif runtime'lar zorunlu runtime bagimliligi degildir.
- Mevcut iskelet batch protokol, structured logging, Faz 3 MVP ve Faz 4 baseline hazirligi icindir.
