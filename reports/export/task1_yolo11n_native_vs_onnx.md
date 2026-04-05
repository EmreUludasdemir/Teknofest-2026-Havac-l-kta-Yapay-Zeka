Candidate: yolo11n
Validation Video: data\THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001\THYZ_2026_Ornek_Veri_Seti\THYZ_2026_Ornek_Veri_1.MP4
Sample Count: 10

| Metric | Native | ONNX |
| --- | --- | --- |
| Runtime | unknown | unknown |
| Provider | - | None |
| P50 Latency ms | 58.618 | 225.9575 |
| P95 Latency ms | 271.5485 | 2205.5985 |
| Aggregate Classes | {"0": 10, "1": 1} | {"0": 4} |

Accepted: False
ONNX Smoke Passed: True

| Frame | Native Count | ONNX Count | Allowed Delta | Count OK | Class OK |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | True | True |
| 1002 | 0 | 0 | 0 | True | True |
| 2005 | 3 | 1 | 1 | False | False |
| 3007 | 2 | 1 | 1 | True | True |
| 4009 | 0 | 0 | 0 | True | True |
| 5012 | 2 | 1 | 1 | True | True |
| 6014 | 2 | 1 | 1 | True | True |
| 7016 | 1 | 0 | 1 | True | True |
| 8019 | 0 | 0 | 0 | True | True |
| 9021 | 1 | 0 | 1 | True | True |
