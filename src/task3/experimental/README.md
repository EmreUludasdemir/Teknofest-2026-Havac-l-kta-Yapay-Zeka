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

## Known Limitation

Frozen validation on 2026-04-19 with `task3_yoloe_min_score=0.4520`, `task3_yoloe_match_normalization_scale=50`, and YOLOE score weights `0.30/0.70` produced 6 accepted matches on `rgb_reference_session`, 0 on `thermal_cross_sensor_proxy`, 0 on `rgb_absent_target_proxy_2025`, and 3 accepted false positives on `thermal_absent_target_proxy_2025`.

This trade-off is intentional for the experimental phase. The frozen threshold recovers RGB-present recall to the v1 target point, but thermal-present recall remains weak and thermal-absent false positives are still present. Future work: add a homography-consistency signal to the scoring formula, which is expected to separate present-target true positives from absent-target false positives without sacrificing recall.

YOLOE scoring constants frozen as of 2026-04-19. No further tuning planned before TEKNOFEST 2026 On Tasarim Raporu submission (April 22, 2026). Post-competition iterations should revisit this calibration with larger probe data and a homography-consistency signal.
