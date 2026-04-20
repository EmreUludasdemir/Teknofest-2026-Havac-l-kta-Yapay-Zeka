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

Validation on 2026-04-20 with `task3_yoloe_min_score=0.4520`, `task3_yoloe_thermal_min_score=0.3240`, `task3_yoloe_match_normalization_scale=50`, `task3_yoloe_homography_ransac_reproj_threshold=5.0`, and 3-term YOLOE score weights `0.25/0.61/0.14` (confidence/matches/inlier_ratio) produced 5 accepted matches on `rgb_reference_session`, 4 on `thermal_cross_sensor_proxy`, 0 on `rgb_absent_target_proxy_2025`, and 3 accepted false positives on `thermal_absent_target_proxy_2025`.

This trade-off is intentional for the experimental phase. The YOLOE path now uses 3-term scoring with homography consistency, but thermal-absent false positives are not fully eliminated at the current recall point. Runtime and evaluator scoring dispatch are aligned, so online pipeline behavior matches manifest measurements. Future work should revisit this calibration with larger probe data and additional geometric signals after the competition window.

YOLOE scoring constants are frozen as of 2026-04-20. No further tuning is planned before TEKNOFEST 2026 competition-report deadlines; post-competition iterations should revisit this calibration with larger probe data and richer homography-consistency features.

One-shot v2 re-calibration was attempted on 2026-04-20 with proposed weights `0.20/0.50/0.30`. The static gate failed before manifest execution: thermal absent candidate count above threshold increased relative to v1, critical false positives remained above threshold, and RGB top-5 candidates dropped below the current accepted level. Because the attempt failed the pre-run decision gate, v1 (`0.25/0.61/0.14`) remains the final frozen configuration.
