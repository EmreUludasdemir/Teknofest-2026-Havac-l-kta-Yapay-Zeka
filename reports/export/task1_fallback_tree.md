| Order | Stage | Runtime | Candidate | Validation Passed | Required Artifact |
| --- | --- | --- | --- | --- | --- |
| 1 | tensorrt:yolo26n | tensorrt | yolo26n | True | yolo26n.engine |
| 2 | onnxruntime:yolo26n | onnxruntime | yolo26n | True | yolo26n.onnx |
| 3 | ultralytics:yolo26n | ultralytics | yolo26n | None | yolo26n.pt |
| 4 | onnxruntime:yolo11n | onnxruntime | yolo11n | False | yolo11n.onnx |
| 5 | ultralytics:yolo11n | ultralytics | yolo11n | None | yolo11n.pt |
| 6 | synthetic | synthetic | None | None | None |
