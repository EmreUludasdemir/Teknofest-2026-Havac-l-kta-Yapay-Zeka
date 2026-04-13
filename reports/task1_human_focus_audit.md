# Task 1 Human Focus Audit

## Diagnosis

- train_local_human_share_low:0.0081
- test_humans_are_local_only
- all_holdout_humans_are_tiny
- occlusion_not_directly_measurable_from_bbox_labels

## Train/Test Human Source Truth

- train local human boxes: `1090`
- train public human boxes: `132691`
- test local human boxes: `73`

## Human Size Summary

- train local insan: `{"count": 1090, "p10": 0.00028191621000000003, "p25": 0.000315266696, "p50": 0.000524346651, "p75": 0.0010568877499999999, "p90": 0.001423155608, "tiny_ratio_le_0_001": 0.7}`
- train public insan: `{"count": 132691, "p10": 4.6665499999999996e-05, "p25": 9.795510199999999e-05, "p50": 0.000229559452, "p75": 0.000522875, "p90": 0.0011115396600000001, "tiny_ratio_le_0_001": 0.8833153718036642}`
- test local insan: `{"count": 73, "p10": 0.000567371673, "p25": 0.000600764156, "p50": 0.000600928371, "p75": 0.000700345804, "p90": 0.0007999274519999999, "tiny_ratio_le_0_001": 1.0}`