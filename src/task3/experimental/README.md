# Task 3 Experimental Backend

Bu klasor `yoloe_vp_lightglue` backend'ini tutar.

## Manual Smoke Gate

- `tools/task3_yoloe_smoke.py` CI tarafinda calistirilmaz.
- CI ortaminda GPU yoktur ve staged weight bulunmaz.
- PR 2 benzeri deneysel backend degisiklikleri merge edilmeden once gelistirici tarafindan elle smoke calistirilmalidir.

## Offline Weight Staging

- Beklenen agirlik yolu: `data/weights/task3/yoloe-11m-seg.pt`
- Runtime ilk cagride internetten agirlik indirmeye guvenmez.
- Agirlik bulunamazsa matcher kontrollu sekilde ORB fallback'e duser.
