# Fold-0 low-CNR seed replication

All 21 validation cases, seeds 0/1/2; fixed 4-mm removal; IoU >0.1 Hungarian matching, 26-connectivity. Unfiltered detections and FP counts reproduce eval_val.

| Run | Low-CNR recall (%) | FP/scan | Tumor Dice (%) |
|---|---:|---:|---:|
| A1_s0 | 66.667 | 1.524 | 66.506 |
| A1_s1 | 69.231 | 1.667 | 70.238 |
| A1_s2 | 61.538 | 1.476 | 67.521 |
| A11_s0 | 74.359 | 1.762 | 69.017 |
| A11_s1 | 76.923 | 1.667 | 75.568 |
| A11_s2 | 74.359 | 1.524 | 69.913 |
| A13a_s0 | 79.487 | 1.667 | 73.618 |
| A13a_s1 | 74.359 | 1.714 | 72.571 |
| A13a_s2 | 76.923 | 1.286 | 74.539 |
| A13b_s0 | 74.359 | 1.857 | 71.970 |
| A13b_s1 | 71.795 | 1.762 | 72.787 |
| A13b_s2 | 74.359 | 1.667 | 69.779 |

Frozen section 6c: mean recall gain >=5 pp; mean FP/scan increase <=0.3; at least two individual seeds pass both conditions against the A1 three-seed mean.

| Group | Mean recall gain (pp) | Mean FP/scan change | Passing seeds | Eligible |
|---|---:|---:|---|---|
| A13b | +7.692 | +0.206 | [1, 2] | PASS |
| A13a | +11.111 | +0.000 | [0, 1, 2] | PASS |
| A11 | +9.402 | +0.095 | [0, 1, 2] | PASS |

First eligible group in registered order A13b -> A13a -> A11: **A13b**.

All groups are reported. Validation was used for development; these results do not establish independent efficacy. No test-set inference or protocol-freezing action is performed by this script.
