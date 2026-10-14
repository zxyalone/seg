# 边界与体积指标（external_msd08；描述性）

肿瘤为 4 mm 后处理后的预测；Dice/VOE/RVD 为 %，ASSD/HD95/MSSD 为 mm；每个运行取病例均值，组内种子取均值。

region group  n_seeds  n_cases  pred_empty_cases  dice   voe   rvd  assd  hd95  mssd
tumour    A0        3      303                13 58.40 53.67 70.12 21.51 55.30 82.35
tumour    A1        3      303                 6 58.41 53.76 63.05 18.19 53.18 81.66
tumour  A13b        3      303                 0 62.25 50.52 66.57 12.84 55.62 94.48

## A13b 与对照的差（配对病例 bootstrap 95% CI，只描述）

region comparison metric  diff  ci95_low  ci95_high
tumour  A13b - A1   dice  3.84      2.59       5.17
tumour  A13b - A1    voe -3.24     -4.40      -2.14
tumour  A13b - A1    rvd  3.52    -24.06      20.33
tumour  A13b - A1   assd -5.35    -10.47      -1.25
tumour  A13b - A1   hd95  2.44     -3.31       7.68
tumour  A13b - A1   mssd 12.82      6.91      18.20
tumour  A13b - A0   dice  3.85      2.62       5.17
tumour  A13b - A0    voe -3.15     -4.32      -2.04
tumour  A13b - A0    rvd -3.55    -36.65      16.88
tumour  A13b - A0   assd -8.68    -16.61      -2.28
tumour  A13b - A0   hd95  0.31     -7.52       7.32
tumour  A13b - A0   mssd 12.13      3.99      19.50
