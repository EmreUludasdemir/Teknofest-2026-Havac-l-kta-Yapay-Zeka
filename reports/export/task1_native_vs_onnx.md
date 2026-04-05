Candidate: yolo26n
Validation Video: data\THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001\THYZ_2026_Ornek_Veri_Seti\THYZ_2026_Ornek_Veri_1.MP4
Sample Count: 10

| Metric | Native | ONNX |
| --- | --- | --- |
| Runtime | cuda:0 | CUDAExecutionProvider |
| Provider | - | CUDAExecutionProvider |
| P50 Latency ms | 60.8125 | 52.732 |
| P95 Latency ms | 536.6668 | 2439.0837 |
| Aggregate Classes | {"0": 5, "1": 1} | {"0": 5, "1": 1} |

Accepted: True
ONNX Smoke Passed: True

| Frame | Native Count | ONNX Count | Allowed Delta | Count OK | Class OK |
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
