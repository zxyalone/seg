# experiments/ — V2 训练、评测与统计工具

对应 `manuscript/实验清单.md` 的 E0（代码修复 + 梯度检查）、E1（评测脚本）、E15（一次性测试）、E16（成本）。
所有新实验都用 **V2** 代码；旧 checkpoint（Copy2 等）只能用 `PHASER_LEGACY=1` 推理。

## 1. V2 代码改动（相对 Copy2 时期）

| # | 问题（来源） | V2 处理 | 位置 |
|---|---|---|---|
| 1 | prior 双重 sigmoid（审计 I3） | `softmax(logits)[:,1]`，只算一次；`PHASER_LEGACY=1` 恢复旧行为 | `phaser/unet.py`、`modules.py` PFCR |
| 2 | prior 损失 epoch ≥ 60 才启用，Copy2（ep48）的 prior 未训练（I4） | 从 epoch 0 启用，ψ(e)=min(1,(e+1)/5) | trainer `train_step` |
| 3 | budget 损失同样被 Phase B 延迟 | 从 epoch 0 启用，同一 ψ(e) | 同上 |
| 4 | 三个辅助损失 ramp 公式不一致 | 统一 `_v2_ramp()`，`PHASER_AUX_WARMUP`（默认 5）；删除死开关 `PHASER_PC_BOUNDARY_WARMUP` | 同上 |
| 5 | 主损失恒带 Focal Tversky（且 docstring 写错权重） | `PHASER_FT_WEIGHT` 默认 0，=0 时只用 DiceCE | `_build_loss` |
| 6 | weight decay 作用于标量/重参数化参数 | `rho_raw, sigma_raw, noise_k, residual_gate, alpha, beta, gamma` 单独成组，wd = 0 | `configure_optimizers` |
| 7 | 无肿瘤样本的 budget 病灶项是常数 1/3 | 乘指示函数（论文 Eq. 25） | `losses.py` |
| 8 | 最浅层 PC 被 2²⁰ 上限降到 64³（I2） | 默认上限 2²²，128³ 不再降采样 | trainer `PC_MAX_VOXELS_DEFAULT` |
| 9 | 噪声尺度固定取 scale 0，学习后不一定最细 | 取当前 ρ_s 最大的尺度（已验证与尺度顺序无关） | `PhaseCongruency.forward` |
| 10 | 训练不可复现（I8） | `PHASER_SEED`：python/numpy/torch + 增强 worker 种子 | trainer `__init__`、`get_dataloaders` |
| 11 | 无运行记录 | `run_manifest.json`：git commit、全部 `PHASER_*/nnUNet_*` 变量、seed、GPU、起止时间、epoch、final checkpoint SHA-256 | trainer `on_train_start/end` |
| 12 | nnU-Net 训练末验证默认 TTA | 默认无 TTA（`PHASER_VAL_TTA=1` 恢复） | `perform_actual_validation` |
| 13 | 新对照 | A7：`PHASER_PFCR_SHUFFLE_PC=1`（router 看到逐样本空间置换的 PC；排序损失仍用原 PC） | PFCR |
| 14 | 推理期干预（E11） | `PHASER_INTERVENE=pc_zero/route_uniform/prior_zero/gate_zero`，`PHASER_INTERVENE_LEVELS=0,1` | PFCR |
| 15 | torch ≥ 2.2 的 `LRScheduler` 不再接受 `verbose` 位置参数 | 去掉该参数（torch 2.1 行为不变） | `GroupPolyLRScheduler` |

未改动、在论文中作为性质或局限陈述：硬阈值 `[E−T]+`；噪声增益指数 1（3D 下是启发式）；频率网格以体素为单位（z 与 xy 物理尺度差 1.28×）；prior 不 detach；softmax 路由无法逐体素关闭。

## 2. E0 梯度检查结果（`check_grads.py`）

CPU、64³ 合成体数据（病灶 CNR 0.3–1.7，接近 LiTS 训练病灶中位 1.6），每组 20 步，真实 `train_step` 与 `configure_optimizers`：

- **9 组全部 PASS**：没有任何 PhASER 参数梯度恒为 0（Copy2 那类"滤波器没被训练"的问题不会再出现）。
- **A6 / A7 / A8**：5 层 log-Gabor 参数（ρ、σ、k）与 proxy 全部拿到正常量级梯度（~10⁻⁴–10⁻³）。弱梯度只有：router 第一层卷积（输出层零初始化，正常）；`small_head.head.0.bias`（紧接 InstanceNorm 的卷积 bias 被 IN 抵消，结构上无效，无害）。
- **A3 / A4 / A5（无 PC 排序损失）**：滤波器参数梯度 10⁻⁸–10⁻²⁶，训练初期实际冻结，只能等 tanh(γ) 变大后经分割损失更新。**解读 A5 vs A6 时必须考虑这一点**：A5 的"去掉排序损失"同时意味着"滤波器几乎不学"。
- 合成数据过易时（CNR > 3），浅层的排序 hinge 在初始化就已满足（gap ≈ 0.32 > margin 0.2），该层滤波器同样只剩 tanh(γ) 路径。这符合 Eq. (24) 的设计，但说明真实数据上 margin 0.2 是否起作用要看实际 gap（E13 记录）。

服务器上请再跑一次 128³、50 步的版本并保存日志：

```bash
PYTHONPATH=nnUNet python experiments/check_grads.py --patch 128 --steps 50 --device cuda | tee experiments/check_grads_gpu.log
```

## 3. 运行

```bash
experiments/queue.sh smoke     # E0 冒烟：每组 2 epoch（结果写到 nnUNet_results_smoke）
experiments/queue.sh stage1    # E2–E4：A6 / A0 / A1，seed 0 → 决策门
experiments/queue.sh stage2    # E5–E8：A7、A6/A0 seed 1–2、A5、A8
```

`run.sh GROUP SEED` 训练一个组：读 `configs/GROUP.env`，结果目录 `nnUNet_results_v2/GROUP_sSEED/`，训练后自动做无 TTA 验证并用 `eval/evaluate.py` 评测 21 例验证集（`eval_val/`）。已有 `checkpoint_final.pth` 的 run 自动跳过，可随时重启队列。

**测试集**：`predict_test.sh GROUP SEED` 只有在 `experiments/PROTOCOL_FROZEN` 存在时才会运行（内容为冻结时的 commit）。冻结前需确定：主要终点、后处理规则（E10，验证集上定）、外部数据集。

## 4. 评测与统计

```bash
python experiments/eval/evaluate.py --pred-dir P --gt-dir G --image-dir I --cases-file experiments/eval/val_cases.txt --out-dir OUT
python experiments/eval/stats.py --cohort Tumor \
   --method A0=r/A0_s0/eval_val,r/A0_s1/eval_val --method A6=r/A6_s0/eval_val,r/A6_s1/eval_val \
   --compare A6:A0 --out stats.csv
python experiments/eval/test_eval.py   # 已知答案的自检
```

`evaluate.py` 的口径（写入 `summary.json`）：

- Dice/IoU：肝实质（label 1）、整肝（label > 0）、肿瘤（label 2），按病例平均；同时累计 TP/FP/FN 以算全局 Dice。
- 空参考 + 空预测 → 肿瘤 Dice = 1；空参考 + 有预测 → 0。
- HD95 = 两个有向 95 分位的最大值（mm）；NSD 容差 τ = 2 mm；含肿瘤病例无预测时 HD95 = 图像对角线、NSD = 0，并报告病例数。
- 病灶级：26 连通分量，IoU > 0.1 为候选，匈牙利一对一匹配（另报 IoU > 0.5）；一个病灶被预测成两块记 1 TP + 1 FP。
- 分层：等体积球直径 <10 / 10–20 / >20 mm；CNR 三分位 1.2785 / 1.9208，CNR 计算与冻结阈值时的 `s42_real_pc.py` 完全一致（计划间距重采样、6 连通、肝实质环 2–6 体素）。

`stats.py`：多 seed 先按病例平均；配对 bootstrap 95% CI（默认 10,000 次，按病例重采样，病灶级指标同样按病例重采样后重算合并比例）；双侧 Wilcoxon 符号秩检验；配对秩二列相关作为效应量；同一指标的多个预设比较做 Holm 校正。

## 5. 其它

- `cost.py`：每组参数量、128³ patch 前向耗时与显存峰值（表 7）。实测 A6 比骨干多 141,641 个参数（与论文 3.4 节手算一致），骨干 31,196,399，增量 **0.454%**（按 6 stage、3×3×3 卷积、步长 [1,2,2,2,2,2] 的计划拓扑；请在服务器上用真实 plans 再确认）。
- `nnUNetTrainer_200epochs`：E9 ResEnc 基线用的 200 epoch 标准训练器，命令见 `queue.sh` 末尾。
- `eval/make_case_lists.py`：从 `splits_final.json` / `test_set.json` 生成病例清单并检查三集合不相交（`run.sh` 首次运行时自动调用）。
