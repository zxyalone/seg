# E11 detection metrics

Three A6 seeds; 21 validation cases (18 tumor, 3 negative). Counts are averaged over
seeds per case, then pooled before computing precision, recall and F1. These are
ratios of pooled counts, not averages of seed F1 values or a prediction ensemble.

`e11_case_metrics.csv` retains the original integer TP/FP/FN counts for every
case, seed, condition and both IoU thresholds. `e11_detection_per_seed.csv`
gives each seed separately. All CSV metrics are fractions; tables below use percent.

Delta = intervention minus none. Paired confidence intervals for the original
Tumor / IoU > 0.1 comparisons are in `e11_paired_vs_none.csv`; blank p-values
for pooled lesion metrics mean no Wilcoxon test was performed, not p=0.

The joint PC/prior-zero follow-up is separate and is not included in the original
nine-comparison Holm family. Similar point estimates alone do not establish equivalence.

## Tumor: IoU > 0.1

| Condition | FP/scan | Δ FP/scan | Precision (%) | Δ precision (pp) | F1 (%) | Δ F1 (pp) | Recall (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate_zero | 2.981 | +0.037 | 56.72 | -0.31 | 65.12 | -0.20 | 76.45 |
| none | 2.944 | +0.000 | 57.03 | +0.00 | 65.33 | +0.00 | 76.45 |
| pc_zero@0 | 3.000 | +0.056 | 56.57 | -0.46 | 65.02 | -0.30 | 76.45 |
| pc_zero@1 | 2.981 | +0.037 | 56.72 | -0.31 | 65.12 | -0.20 | 76.45 |
| pc_zero@2 | 2.870 | -0.074 | 57.65 | +0.62 | 65.73 | +0.41 | 76.45 |
| pc_zero@3 | 2.722 | -0.222 | 58.94 | +1.91 | 66.56 | +1.24 | 76.45 |
| pc_zero@4 | 2.815 | -0.130 | 58.13 | +1.10 | 66.04 | +0.72 | 76.45 |
| pc_zero@all | 2.778 | -0.167 | 58.45 | +1.42 | 66.25 | +0.92 | 76.45 |
| prior_zero | 2.926 | -0.019 | 57.18 | +0.15 | 65.43 | +0.10 | 76.45 |
| route_uniform | 2.519 | -0.426 | 60.81 | +3.78 | 67.74 | +2.41 | 76.45 |

## All: IoU > 0.1

| Condition | FP/scan | Δ FP/scan | Precision (%) | Δ precision (pp) | F1 (%) | Δ F1 (pp) | Recall (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate_zero | 2.984 | +0.032 | 52.88 | -0.27 | 62.52 | -0.19 | 76.45 |
| none | 2.952 | +0.000 | 53.15 | +0.00 | 62.70 | +0.00 | 76.45 |
| pc_zero@0 | 2.968 | +0.016 | 53.02 | -0.13 | 62.61 | -0.09 | 76.45 |
| pc_zero@1 | 2.968 | +0.016 | 53.02 | -0.13 | 62.61 | -0.09 | 76.45 |
| pc_zero@2 | 2.889 | -0.063 | 53.69 | +0.54 | 63.08 | +0.37 | 76.45 |
| pc_zero@3 | 2.730 | -0.222 | 55.09 | +1.94 | 64.04 | +1.33 | 76.45 |
| pc_zero@4 | 2.794 | -0.159 | 54.52 | +1.37 | 63.65 | +0.95 | 76.45 |
| pc_zero@all | 2.810 | -0.143 | 54.38 | +1.23 | 63.55 | +0.85 | 76.45 |
| prior_zero | 2.905 | -0.048 | 53.55 | +0.40 | 62.99 | +0.28 | 76.45 |
| route_uniform | 2.540 | -0.413 | 56.87 | +3.72 | 65.22 | +2.52 | 76.45 |

