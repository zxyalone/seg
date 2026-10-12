# 验证集阈值曲线（原 dtype 导出，15 个模型）

- 模型：A0、A1、A11、A13a、A13b，各 seed 0/1/2，fold 0 的 checkpoint_final；21 例验证集；无 TTA；滑窗步长 0.5、Gaussian 加权。
- 导出：`missed_lesion_probability.py run --workers 4 --offload-models`（提交 28ff943），logits 保持 FP16（与训练末验证相同），315 个 JSON 全部记录 `export_logits_dtype = torch.float16`。`--workers` 与 `--offload-models` 在 LiTS_042 上逐字节复核与串行结果相同。
- 汇总：`threshold_sweep_per_run.csv`（每个 run × 判定规则的召回、FP/scan、Dice、F1、分层召回）、`lesions.csv`、`diag_missed_probability.md`/`decision.json`（§4b 诊断按组的附带输出）。
- 运行记录：`shard*.log`、`last_*.log`（OOM 与内存中断后的续跑，已写结果自动跳过）、`summarize.log`。

## 一致性检查（计划 §11，2026-10-12）：未通过（2/315）

`parity/`（`check_curve_parity.py --jobs 15`）。三项判据：

| 判据 | 结果 |
|---|---|
| 未过滤病灶检出与 eval_val 逐病灶一致 | 15/15 个 run 全部一致 |
| 4 mm 过滤后每例 FP 数与逐病灶检出与保存的验证分割完全一致 | 检出全部一致；FP 数有 2 处不一致 |
| 未过滤 FP 数与 eval_val 每例相差 ≤ 1 | 通过；A11_s0、A11_s1 在 LiTS_094 各差 1 |

不一致的两处都是 4 mm 阈值边界上的一个 FP 连通域：

- A0_s0 / LiTS_094：保存分割中该连通域 3.996 mm（68 体素）被去掉，重新推理时多一个体素、≥ 4 mm 被保留，FP +1；
- A1_s1 / LiTS_053：保存分割中该连通域 4.046 mm（12 体素）被保留，重新推理时少一个体素、< 4 mm 被去掉，FP −1。

原因与未过滤差异相同：训练进程 `cudnn.benchmark = True` 按计时选卷积算法，接近打平的体素可翻转，逐比特一致不可得。按用户决定：**判据不事后放宽，记为未通过**；曲线按原样使用，只用于"同等 FP/scan 比较"这一次要分析，A0_s0 与 A1_s1 在 argmax_pp 点上与保存分割相差 ±1 个 FP 连通域（FP/scan ±0.048），并在论文中注明。不用保存分割替换重新推理结果。§6c 的判定基于保存分割，不受影响。
