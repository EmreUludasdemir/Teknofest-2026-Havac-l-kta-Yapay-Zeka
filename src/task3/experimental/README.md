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

## Frozen Configuration (2026-04-21)

Thermal threshold raised to 0.50 (2026-04-21) to prioritize FP minimization. Thermal TPs reduced (4→1) but all thermal absent FPs eliminated (7→0). Trade-off justified by mAP penalty economics and evidence that thermal TPs were semantically weak in the YOLOE/LightGlue RGB-biased pipeline.

| Setting | Value |
|---|---|
| `task3_yoloe_min_score` | 0.4520 |
| `task3_yoloe_thermal_min_score` | **0.50** |
| `task3_yoloe_match_normalization_scale` | 50 |
| `task3_yoloe_homography_ransac_reproj_threshold` | 5.0 |
| YOLOE score weights (confidence/matches/inlier_ratio) | 0.25 / 0.61 / 0.14 |

Manifest validation on 2026-04-21 (`2026-04-21_thermal_threshold_050_v1`):

| Scenario | v2 (thermal=0.3240) | v3 (thermal=0.50) |
|---|---|---|
| rgb_reference_session | 6 | 6 |
| thermal_cross_sensor_proxy | 4 | 1 |
| rgb_absent_target_proxy_2025 | 1 | 1 |
| thermal_absent_target_proxy_2025 (FP proxy) | 7 | **0** |

No further threshold tuning is planned before TEKNOFEST 2026 competition deadlines. Post-competition iterations should revisit thermal recall with larger probe data and cross-sensor-aware reference banks.
