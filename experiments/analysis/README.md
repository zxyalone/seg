# experiments/analysis/ — E11 / E12 / E13

对应论文 4.8 节（E13）和 4.9 节（E11、E12）。三个脚本都只用 **21 例 fold-0 验证集**，读取各 run 的 `checkpoint_final.pth`，推理设置与训练结束时的验证完全相同：滑窗步长 0.5、Gaussian 加权、无 TTA。测试集不碰。

共同做法：所有模型一次性加载；每个病例只读取、预处理一次，之后所有模型和所有条件都在同一份缓存数组上推理。E11/E12 支持 `--shard i/n` 按病例分给多张 GPU，中断后直接重跑同一命令即可，已完成的病例会自动跳过。

| 脚本 | 内容 | 默认 run | 预测次数 | 粗估耗时（3 × 3080 Ti） |
|---|---|---|---|---|
| `e11_interventions.py` | 推理期干预（表 6） | A6 三个 seed | 21 例 × 3 × 10 种条件 = 630 | 约 2 小时 |
| `e12_degradation.py` | 病灶对比度受控退化（图 6） | A0/A1/A6 各 3 个 seed + A7 | 21 例 × 10 个模型 × 4 个 α = 840 | 约 2–3 小时 |
| `e13_mechanism.py` | 学到的滤波器、特征上的 PC、路由图、残差强度（图 3 训练后面板、图 5） | A6 三个 seed + A7 | 每个病灶一个 128³ patch | 约 0.5 小时，单卡 |
| `missed_lesion_probability.py` | A0 漏检病灶内的肿瘤概率与判定阈值扫描（计划 §4b：决策层还是表征层） | A0 三个 seed | 21 例 × 3 = 63 | 约 1 小时，两卡分片 |

耗时按 A6 单个 patch 前向 77 ms 估算。以第一个病例日志打印的秒数为准再修正。

## 当前服务器执行记录（2026-10-10）

已拉取 main 上的 `6dd11f1`。附带 CPU 合成数据端到端自检全部通过，日志归档为 `results/server/analysis_selftest.log`；这些合成数据指标不属于论文实验结果。
北京时间 21:47 由 `experiments/server_analysis.py` 启动：GPU0 E13，GPU1/2 E11 的前两个病例分片。
共享队列随后派发第三个 E11 分片和三个 E12 分片；推理齐备后，在 CPU 上自动做评测、统计和基线核验。
实际 GPU 核验发现：LiTS_006 的 A6_s0 none 与原验证相差 102 个体素（占全卷约 7.51e-7），其中肿瘤标签只变 3 个体素，Tumor Dice 差 −1.698e-5；原生 FP16 导出复推仍相差 38 个体素，FP32 导出为 102 个体素。CPU 自检的逐体素一致不能直接当作 GPU 历史验证的精确复现要求。
保留用户原分析脚本，仅将执行核查中的 A0/A1/A6 宏平均 Tumor Dice 绝对差容限设为 1e-4（0.01 pp），完整核查保存实际差值，不宣称逐体素一致。A7 的 PC 空间置换每次 forward 使用随机排列，历史验证不能精确复现，其差值只记录为随机对照参考。E12 realised CNR 中位比值与 α 的容差沿用附带测试的 0.02。
核验失败会记录 failed，不把输出当作已完成分析。小结果归档到 `results/E11`、`results/E12`、`results/E13`，预测体数据留在服务器。

实时进度：`$DATA/nnUNet_results_v2/_orchestration/analysis_state.json`；调度日志为 `analysis_scheduler.log`，各任务为 `analysis_<任务名>.log`。
上述分析只读取 21 例验证集；用户已明确暂不冻结阶段 7 终点，等待 E11–E13 结果，外部集尚未准备。阶段 7 的一次性测试须在协议前置条件完成后另行执行。

## 运行

```bash
cd /root/autodl-tmp/Downstream_2 && git pull
PY=/root/autodl-tmp/envs/mamba/bin/python
export PYTHONPATH=$PWD/nnUNet

# E11：三卡分片，全部完成后汇总
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i $PY experiments/analysis/e11_interventions.py run --shard $i/3 > e11_$i.log 2>&1 & done; wait
$PY experiments/analysis/e11_interventions.py summarize

# E12：同上
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i $PY experiments/analysis/e12_degradation.py run --shard $i/3 > e12_$i.log 2>&1 & done; wait
$PY experiments/analysis/e12_degradation.py summarize

# E13：单卡
CUDA_VISIBLE_DEVICES=0 $PY experiments/analysis/e13_mechanism.py | tee e13.log
```

默认输出目录为 `$DATA/nnUNet_results_v2/_analysis/{E11,E12,E13}`。路径可以用环境变量 `DATA`、`RESULTS_ROOT`、`RAW_DATASET` 覆盖，也可以用 `--results-root`、`--image-dir`、`--label-dir` 参数指定。

## 各脚本的定义

**E11**（论文中的层号 = 代码层号 + 1，level 1 是 128³ 那一层）

- 条件：`none`；`pc_zero@0..4`（逐层把 PC 置零）；`pc_zero@all`；`route_uniform`（路由固定为 1/3）；`prior_zero`；`gate_zero`（关闭 PFCR，out = x）。
- 自检：汇总时把 `none` 和训练结束时的 `eval_val` 比较，结果写入 `check_none_vs_eval_val.csv`，差值应接近 0。
- 每个病灶的效用 U_j = L_j(干预) − L_j(none)。L_j 是病灶外接框外扩 10 体素范围内、肿瘤概率的 soft-Dice loss，在预处理空间计算。U_j 为正表示被去掉的信号对这个病灶有帮助。CNR 分层沿用冻结的三分位阈值。
- 统计：各条件与 `none` 做配对比较。先对 seed 取平均，再做病例级 bootstrap（10,000 次）、Wilcoxon 检验和 Holm 校正。
- 输出：`table6_interventions.md`、`e11_paired_vs_none.csv`、`e11_lesion_utility.csv`。

**E12**

- 退化方式：每个病灶（26 连通，与 `evaluate.py` 的病灶编号一致）在 HU 上做均值平移，使病灶均值向其肝实质环的均值靠拢：x' = x − (1−α)(μ_病灶 − μ_环)·w。
- w：病灶内为 1，病灶外按 σ = 1 体素的高斯平滑衰减。这样边界没有台阶，病灶内部的纹理和环的统计量都不变，病灶 CNR 恰好变为 α·CNR（`e12_cnr_check.json` 会核验这一点）。这对应论文 4.9 节写的"均值平移"版本，不是乘性缩放。
- 标签不变。肝实质环少于 10 个体素的病灶不做退化，也不计入统计。
- 输出：
  - `e12_recall_curve.csv`：各组在各 α 下的召回率；
  - `e12_recall_by_stratum.csv`：按 CNR 分层的召回率；
  - `e12_tumor_dice.csv`；
  - `e12_interaction.csv`：交互项 [B 的召回下降 − A 的召回下降]，附病例级 bootstrap CI，正值表示 B 随对比度下降得更慢；
  - `fig6_degradation.png`。

**E13**

1. **滤波器**：每层每个尺度的中心波长 1/ρ、带宽 σ、噪声倍数 k、tanh(γ)，与初始值（波长 3.0 / 6.3 / 13.2 体素）比较。
2. **特征上的 PC**：每个病灶（预处理空间 ≥ 50 体素）取一个以它为中心的 128³ patch 前向，在每层比较边界壳层（3×3×3 膨胀减腐蚀）和肝实质环（距离 2–6）上的 PC，报告比值、AUC 和壳层非零比例，按 CNR 三分位分组。
3. **路由**：肿瘤、肝、背景三个区域内的平均路由；逐频带的选择性 |Δ|（每个频带分别取绝对值）；变异系数；与预算的 L1 距离。运行前会先用已知答案的合成路由做自检。
4. **先验头**：肿瘤内和背景中的平均先验概率。

输出：`summary.md`、`filters.csv`、`pc_by_level_stratum.csv`、`route_by_level.csv`、`lesion_level_stats.csv`、`fig5_mechanism.png`。

## 需要提交回仓库的文件

### E11 补充核验（2026-10-11）

原 E11 已于 08:21 完成；`results/E11` 保存原始评测 CSV/JSON、utility、配对比较和基线核查。
`e11_detection_metrics.py` 从已有 `cases.csv` 导出 FP/scan、精确率、F1、召回率，包含 Tumor（18 例）
和 All（21 例，含 3 例阴性）队列，以及 IoU > 0.1 / 0.5 两个阈值。先按病例平均 seed 计数，再汇总
计算比例；同时提供每个 seed 的指标和逐病例整数 TP/FP/FN，便于复算。

新增 `e11_joint_zero.py` 将路由器的两个输入通道（PC、prior）同时置零，保留所有模型参数。
路由器为 3×3×3 conv + ReLU + 1×1×1 conv，其输入仅为 PC/prior。零输入后的频带比例为
`softmax(W2 * ReLU(b1) + b2)`，每个 seed、每层分别固定；残差仍处理该病例的特征。
脚本在真实推理中核查 15 个路由器的空间恒定性，并保存比例与 tanh(γ)。
CPU 自检验证联合置零与显式 PC=0/prior=0 一致、清除钩子后恢复正常推理、权重不变。

联合干预只增加 63 次验证集推理；基线预测/评测通过符号链接复用，utility 单独复制，
输出放在服务器 `_analysis/E11_followup`，完成后归档到 `results/E11_followup`。
`server_e11_followup.py` 等待 GPU 1 或 2 空闲，不中断 E12，不使用预留的 GPU 0；
自动汇总、导出检测指标并以普通 push 推送 main，远端冲突时记录失败，不强制覆盖。

2026-10-11 11:47，用户明确停止 E12。已终止 E12 推理和原调度器，服务器保留 687/840 个
预测文件（未完成完整评测，不作为完整实验结果）。取消标记阻止自动恢复 E12 推理或汇总，
详见 `results/server/e12_cancellation.json`；PC/prior 联合置零补充队列独立继续。

此检查为看过原结果后新增的验证分析，单独报告，不加入原九组比较的 Holm 校正。
相近的点估计本身不能证明统计等价；固定路由也不等于整个 PFCR 输出固定，滤波器、残差门控
与输入特征仍保留。FP/scan 未上升只能按病例级不确定性判断该验证集对机制的支持程度。

```bash
PYTHONPATH=nnUNet python experiments/analysis/test_e11_joint_zero.py
python experiments/analysis/e11_detection_metrics.py --source "$RESULTS_ROOT/_analysis/E11" --out results/E11
python experiments/server_e11_followup.py
```

各脚本输出目录里的 CSV、MD、JSON 和 PNG 都很小，复制到 `results/E11`、`results/E12`、`results/E13` 后提交。`pred/` 下的预测体数据留在服务器上。

## 测试

```bash
PYTHONPATH=nnUNet python experiments/analysis/test_analysis.py
```

这个测试在 CPU 上运行，约 10–20 分钟。它用合成数据让 nnU-Net 实际做一次计划，生成真实格式的 checkpoint，然后跑通三个脚本，并核对以下几点：

- E11 的 `none` 与 `nnUNetv2_predict`（无 TTA）的结果逐体素一致；
- 干预确实改变了输出；
- E12 在 α = 1 时与 E11 的 `none` 一致，且退化后实际 CNR = α·CNR；
- E13 能从未训练的 checkpoint 中读回初始滤波器参数。
