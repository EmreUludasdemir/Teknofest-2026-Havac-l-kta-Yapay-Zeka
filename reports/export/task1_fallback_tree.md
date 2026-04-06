| Order | Stage | Runtime | Candidate | Validation Passed | Production Enabled | Required Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | tensorrt:yolo26n | tensorrt | yolo26n | True | True | yolo26n.engine |
| 2 | onnxruntime:yolo26n | onnxruntime | yolo26n | True | True | yolo26n.onnx |
| 3 | ultralytics:yolo26n | ultralytics | yolo26n | None | True | yolo26n.pt |
| 4 | ultralytics:yolo11n | ultralytics | yolo11n | None | True | yolo11n.pt |
| 5 | synthetic | synthetic | None | None | True | None |
