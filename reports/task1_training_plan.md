# Task 1 Training Plan

## Experiment 1

- status: `smoke_completed_full_run_pending`
- scope: public-only baseline
- model: `yolo11s`
- imgsz: `960`
- epochs: `40`
- batch: `8` then fallback `4` or `2`
- augment: mosaic `0.7`, scale `0.5`, translate `0.1`, fliplr `0.5`, copy_paste `0.15`, offline weather/night overlay on 15% of train images
- executed now: `1 epoch smoke`, `batch=2`, `imgsz=960`

## Experiment 2

- status: `smoke_completed_full_run_pending`
- scope: local labeled + public combined
- model: `yolo11s`
- imgsz: `960`
- epochs: `60`
- local train oversample: `2x`
- executed now: `1 epoch smoke`, `batch=2`, `imgsz=960`

## Experiment 3

- status: `proxy_only_smoke_completed`
- scope: tiled inference probe on Experiment 2 best checkpoint
- tile size: `960`
- overlap: `0.25`
- current truth: proxy-only tiled replay produced zero detections and severe latency regression; no label-aware tiled gain is claimed
