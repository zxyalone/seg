# Fixed 4-mm pilot metrics

All 21 validation cases; 26-connectivity; IoU > 0.1 Hungarian matching. No inference or threshold tuning.

| Run | Low-CNR recall (%) | FP/scan | Tumor Dice (%) | F1 (%) |
|---|---:|---:|---:|---:|
| A1_s0 | 66.667 | 1.524 | 66.506 | 72.165 |
| A9_s0 | 66.667 | 1.429 | 63.891 | 73.575 |
| A11_s0 | 74.359 | 1.762 | 69.017 | 73.529 |
| A12_s0 | 76.923 | 2.857 | 70.534 | 66.667 |

Pre-registered gate: low-CNR recall >= A1_s0 + 5 pp AND FP/scan <= A1_s0 + 0.3.

Every pilot is reported. These results are gates, not manuscript evidence. A0 + threshold adjustment and recall at matched FP/scan remain required comparisons.

- A9_s0: STOP; gain +0.000 pp; FP/scan delta -0.095
- A11_s0: PASS; gain +7.692 pp; FP/scan delta +0.238
- A12_s0: STOP; gain +10.256 pp; FP/scan delta +1.333
