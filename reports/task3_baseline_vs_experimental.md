# Task 3 Baseline vs Experimental

- Phase status: `BLOCKED_BY_WEIGHT`
- Revalidation with real YOLOE weight: `False`
- Weight staging status: `WEIGHT_MISSING`
- Note: The metrics below are the last prompt_approx comparison already measured on this branch.

| Metric | Baseline | Experimental | Delta |
| --- | --- | --- | --- |
| Accepted Match Count | 40 | 3 | -37 |
| Present Detection Frames | 16 | 3 | -13 |
| False Positive Proxy | 24 | 0 | -24 |
| Absent-target No-match Rate | 0.0 | 1.0 | 1.0 |
| Re-detection Success Count | 2 | 0 | -2 |
| Runtime P50 ms | 189.0036 | 457.1844 | 268.1808 |
| Runtime P95 ms | 330.8254 | 572.72 | 241.8946 |
| Peak VRAM MB | 3079.0 | 2934.0 | - |
