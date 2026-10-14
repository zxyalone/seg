# 测试集漏检病灶的覆盖率（计划 §11，2026-10-14 计算前登记；描述性）

起因：图 5 按规则选出的代价例 LiTS_117 病灶 3 在 A13b 三个种子中都被预测覆盖 98% 以上，但预测与相邻大病灶连成一个连通域，一对一匹配 IoU 约 0.04，按协议计为漏检。

- 脚本：`experiments/analysis/miss_coverage.py`。对 A0、A1、A13b 各 3 个种子，每个按协议漏检的病灶计算被 4 mm 后处理后肿瘤预测覆盖的比例；≥ 50% 记为 merged（覆盖它的预测连通域还与其他金标准病灶重叠）或 oversized（不重叠），< 50% 记为 uncovered。检出状态与 `results/test_protocol/tables/lesions_pp4mm.csv` 逐个核对一致。
- 结果（3 种子均值）：高 CNR < 10 mm 层（92 个）中，A1 漏检 29（uncovered 25.3、merged 3.3、oversized 0.3），A13b 漏检 36（uncovered 31.7、merged 4.3）。代价主要是真正的漏检。全部病灶中覆盖 ≥ 50% 的漏检：A0 7.0、A1 6.3、A13b 8.0。
- 图 5 代价行据此修订为 LiTS_109 病灶 26（A13b 三个种子覆盖率均为 0），见计划 §11。
