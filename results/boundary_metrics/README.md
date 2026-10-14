# 边界与体积指标（计划 §11，2026-10-14 计算前登记；描述性，冻结协议之外）

- 脚本：`experiments/analysis/boundary_metrics.py`（`cases`、`summary` 两步），日志 `run.log`。
- `test/`：LiTS 测试集 27 例，A0、A1、A11、A13a、A13b 各 3 个种子和 A12_s0。肿瘤取 4 mm 后处理后的预测，只在 24 例含肿瘤病例上计算；肝脏比较预测的肝+肿瘤与金标准的肝+肿瘤，不做后处理。
- `external_msd08/`：MSD Task08 303 例，A0、A1、A13b，只算肿瘤（肝参照来自 A0 预测，不评肝脏）。
- 逐例指标：Dice、VOE、RVD（%），ASSD、HD95、MSSD（mm）。表面为 6 邻域腐蚀边界。含肿瘤病例若没有肿瘤预测，表面距离取图像对角线长度；`pred_empty_cases` 为该组所有种子合计的例次。
- 汇总：每个运行取病例均值，组内种子取均值；`boundary_comparisons.csv` 为 A13b − A1、A13b − A0 的配对病例 bootstrap 95% CI（10 000 次，种子 20261012），只描述。
- 核对：测试集肿瘤 Dice 与冻结协议的次要终点一致（例如 A13b 73.97、A0 74.58）；单例肝脏 Dice 与 evaluate.py 一致。

要点：测试集上 A13b 与 A1 的肿瘤 Dice、ASSD、HD95 差的区间都包含 0；肿瘤 RVD 从 −15.0% 变为 −3.4%（差 +11.6，[4.7, 22.1]），体积低估减少。MSD Task08 上 A13b 的肿瘤 Dice +3.8（[2.6, 5.2]），ASSD −5.4 mm（部分来自对照模型无肿瘤预测的病例），MSSD +12.8 mm（[6.9, 18.2]），与假阳性增加一致。
