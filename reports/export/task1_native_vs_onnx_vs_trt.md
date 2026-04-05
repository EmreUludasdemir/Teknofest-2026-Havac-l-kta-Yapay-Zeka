Primary Candidate: yolo26n
Fallback Candidate: yolo11n
Validation Video: data\THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001\THYZ_2026_Ornek_Veri_Seti\THYZ_2026_Ornek_Veri_1.MP4
Sample Count: 10

| Metric | Native | ONNX | TRT |
| --- | --- | --- | --- |
| Runtime | cuda:0 | CUDAExecutionProvider | CUDAExecutionProvider |
| Provider | - | CUDAExecutionProvider | CUDAExecutionProvider |
| P50 Latency ms | 60.8125 | 52.732 | 45.524 |
| P95 Latency ms | 536.6668 | 2439.0837 | 1647.59315 |
| Peak VRAM MB | 2589.0 | 2661.0 | 2821.0 |
| Aggregate Classes | {"0": 5, "1": 1} | {"0": 5, "1": 1} | {"0": 5, "1": 1} |

Accepted: True
TRT Smoke Passed: True
TRT Latency OK: True
TRT VRAM OK: True

| Frame | Native Count | TRT Count | Allowed Delta | Count OK | Class OK |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 1 | True | True |
| 1002 | 1 | 1 | 1 | True | True |
| 2005 | 0 | 0 | 0 | True | True |
| 3007 | 2 | 2 | 1 | True | True |
| 4009 | 0 | 0 | 0 | True | True |
| 5012 | 1 | 1 | 1 | True | True |
| 6014 | 0 | 0 | 0 | True | True |
| 7016 | 0 | 0 | 0 | True | True |
| 8019 | 1 | 1 | 1 | True | True |
| 9021 | 0 | 0 | 0 | True | True |
