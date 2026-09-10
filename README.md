# PhASER-3D 肝脏/肿瘤分割 — 最小复现包

本仓库是 PhASER-3D（VoComni 预训练 3D nnU-Net + 相位-频率一致性路由 PFCR）的**最小复现集**，
目标只有一个：在 `Dataset102_LiTS` fold 0 的 27 例测试集上，用指定 checkpoint 精确复现 24 例
含肿瘤病例的全体积 Dice / IoU。

仓库只保留复现所需的推理代码（本项目 fork 的 nnU-Net），所有旧实验代码、模块、文档、PPT 和
实验产物已删除；完整旧版本见文末备份。

---

## 1. 复现目标

测试口径：

- 27 例测试病例中剔除 `LiTS_032`、`LiTS_106`、`LiTS_115`（这 3 例 GT 无肿瘤体素）
- 其余 24 例逐病例计算全体积 Dice / IoU，再取**逐病例均值 ± 总体标准差（ddof=0）**
- 推理关闭 TTA（`--disable_tta`）

| 类别 | Dice | IoU |
|---|---:|---:|
| Liver | `0.951908 ± 0.027051` | `0.909411 ± 0.045781` |
| Tumor | `0.702935 ± 0.152399` | `0.560872 ± 0.162564` |

## 2. 快速复现

环境（已存在，勿删）：

- Python：`/root/autodl-tmp/envs/mamba/bin/python`（torch 2.1.0+cu118，SimpleITK 2.5.6）
- 本项目代码通过 `PYTHONPATH=<repo>/nnUNet` 生效，无需重装

```bash
# 方式一：直接从已保存的预测重新计算指标（秒级，推荐先跑这个）
/root/autodl-tmp/envs/mamba/bin/python reproduce_metrics.py

# 方式二：先重新生成 27 例预测（无 TTA，约 40–60 分钟），再计算指标
/root/autodl-tmp/envs/mamba/bin/python reproduce_metrics.py --predict
```

脚本会把 24 例逐病例结果打印出来，并与预期值逐项比对（容差 `2e-4`），全 PASS 即精确复现。
当前已保存的预测位于：

```
/root/autodl-tmp/nnUNet_data/nnUNet_results/Dataset102_LiTS/
  nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/test_predictions_100ep_no_tta_clean/
```

## 3. 模型与权重

| 项目 | 值 |
|---|---|
| Checkpoint | `.../fold_0/_old_arch_backup/checkpoint_best-Copy2.pth` |
| 文件 md5 | `553225e565b62a1d9f996fbc1d6b3a45` |
| 保存 epoch | `49`（即第 48 epoch 验证后保存） |
| 内部指标 | `_best_ema = 0.8213` |
| 对应日志 | `training_log_2026_9_5_15_36_00.txt`，`2026-09-05 17:21:40` |
| 架构 | VoComni PlainConvUNet（6 stage，32/64/128/256/320/320，patch 128³）+ PFCR（monogenic PC + Haar 8 子带路由 + tumor prior）+ Small-Tumor Head |
| 预训练骨干 | `VoComni_nnunet.pt`（md5 `ef44f2fb457ea3652c3566848b40b338`）：加载 282/292 keys，encoder 96/96；10 个 `decoder.seg_layers.*` 因 VoComni 为 21 类、本模型 3 类而保持初始化 |
| 激活模块 | PFCR(prior=True, budget=True, w=0.05)、PC boundary ranking(w=0.2)、spectral(w=0.05)、boundary(w=0.05)、lesion weight(w=0.2)、Small-Tumor Head、Focal Tversky(w=0.15) |
| 关闭模块 | PCSF、LWFD、ACG、LA-ACG、T-RAF、CFF |

注意：run 目录里的 `checkpoint_best.pth` 不是这个模型。该次训练后续 EMA 继续升高
（0.8286 → … → 0.8610），原始 `checkpoint_best.pth` 被覆盖；epoch 48/49 的权重只在
`_old_arch_backup/checkpoint_best-Copy2.pth` 中保留。

## 4. 该 checkpoint 对应的训练配置

来自 `training_log_2026_9_5_15_36_00.txt` 的配置头：

```
数据集/配置: Dataset102_LiTS, 3d_fullres, fold 0, nnUNetPlans
训练器: nnUNetTrainer_phaser
patch 128³, batch 2, oversample_foreground_percent = 0.66

PCSF: False          PFCR: True (prior=True, budget=True, w=0.05)
PC boundary loss: True (w=0.2, margin=0.2, warmup=5ep, active from epoch 0)
LWFD: False          T-RAF: none          ACG: False       LA-ACG: False
Small-Tumor Head: True                    CFF Fusion: none
Spectral loss: True (w=0.05)              Boundary loss: True (w=0.05)
Lesion weight: True (w=0.2)               Focal Tversky: w=0.15
Aux loss warmup: 50 epochs
Three-phase: 0-59 (A), 60-79 (B), 80-200 (C)   # 配置 200 epoch
Initial LR: 0.005 (差分 LR encoder 0.3x / decoder 0.5x / phaser 1.0x)
SGD momentum 0.99 + nesterov, weight decay 3e-5, grad clip 12, AMP, PolyLR 0.9
```

实际执行：该次运行在 epoch 100 左右被中断，从未跑满 200。**Copy2 是第 48 epoch（Phase A）
的权重**，此时只有主分割损失 + PC ranking 生效，boundary/spectral/lesion/small-head 等
Phase B 损失（epoch ≥ 60）尚未启用。

### 4.1 训练命令（产生该 checkpoint 的运行）

以下命令逐项对应上表配置，即产生 Copy2 权重的那次运行的启动命令。

> ⚠️ 必须在**同一个 shell** 里先 `export` 上面所有变量，只运行最后一行 `nnUNetv2_train`
> 不会带上任何 PhASER 配置（会退化成 `budget=False`，且默认 VoComni 路径不存在 → 不加载预训练）。
> 命令已把 `nnUNet_results` 指向新目录 `nnUNet_results_retrain`，避免覆盖现有
> `nnUNet_results/.../fold_0/` 的旧产物（`_old_arch_backup/` 与带时间戳的旧日志不受影响）。

```bash
# --- 路径 ---
export nnUNet_raw=/root/autodl-tmp/nnUNet_data/nnUNet_raw
export nnUNet_preprocessed=/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed
export nnUNet_results=/root/autodl-tmp/nnUNet_data/nnUNet_results_retrain   # 新目录，不动旧产物
export PYTHONPATH=/root/autodl-tmp/Downstream_2/nnUNet
export CUDA_VISIBLE_DEVICES=0

# --- 模型模块开关（缺一不可）---
export PHASER_PFCR=1              # 相位-频率一致性路由
export PHASER_PFCR_USE_PRIOR=1    # PFCR 以 tumor prior 为条件
export PHASER_L_PFCR_BUDGET=1     # 频带预算损失
export PHASER_SMALL_HEAD=1        # 小肿瘤辅助头（tumor prior 来源）
export PHASER_PC_SCALES=3
export PHASER_PC_MAX_VOXELS=1048576
export PHASER_PCSF=0
export PHASER_LWFD=0
export PHASER_ACG=0
export PHASER_T_RAF=none
export PHASER_LA_ACG=0

# --- 辅助损失（与日志 header 一致）---
export PHASER_L_PC_BOUNDARY=1
export PHASER_PC_BOUNDARY_WEIGHT=0.2
export PHASER_PC_BOUNDARY_MARGIN=0.2
export PHASER_PC_BOUNDARY_WARMUP=5
export PHASER_L_SPECTRAL=1
export PHASER_L_BOUNDARY=1
export PHASER_ENABLE_LESION_WEIGHT=1
export PHASER_PFCR_BUDGET_WEIGHT=0.05
export PHASER_BOUNDARY_WEIGHT=0.05
export PHASER_SPECTRAL_WEIGHT=0.05
export PHASER_LESION_WEIGHT=0.2
export PHASER_FT_WEIGHT=0.15

# --- 训练超参（与日志 header 一致）---
export PHASER_OVERSAMPLE=0.66
export PHASER_WARMUP_EPOCHS=50
export PHASER_PHASE2_START=60
export PHASER_PHASE3_START=80
export PHASER_NUM_EPOCHS=200
export PHASER_AMP=1

# --- VoComni 预训练骨干（产生 Copy2 的运行加载了它；从头训练消融时置空）---
export VOCOMNI_NNUNET_CKPT=/root/autodl-tmp/Downstream_2/VoComni_nnunet.pt

/root/autodl-tmp/envs/mamba/bin/nnUNetv2_train Dataset102_LiTS 3d_fullres 0 \
  -tr nnUNetTrainer_phaser -p nnUNetPlans
```

### 4.2 启动后自检

1. 终端/日志 header 必须出现：

   ```
   PFCR: True (prior=True, budget=True, w=0.05)
   ```

   若显示 `budget=False`，说明漏了 `PHASER_L_PFCR_BUDGET=1`。

2. 确认运行进程实际继承的环境（两个最容易漏的变量）：

   ```bash
   PID=$(pgrep -f "nnUNetv2_train Dataset102_LiTS" | head -1)
   tr '\0' '\n' < /proc/$PID/environ | grep -E \
     'PYTHONPATH|nnUNet_results|PHASER_L_PFCR_BUDGET|VOCOMNI_NNUNET_CKPT'
   ```

   期望包含 `PYTHONPATH=/root/autodl-tmp/Downstream_2/nnUNet`、
   `PHASER_L_PFCR_BUDGET=1`、
   `VOCOMNI_NNUNET_CKPT=/root/autodl-tmp/Downstream_2/VoComni_nnunet.pt`。
   另外日志里的 `No module named 'hiddenlayer'` 只影响网络结构图绘制，不影响训练。

### 4.3 训练与"指标复现"的关系（重要）

- 这组指标**确定性地**来自 `checkpoint_best-Copy2.pth`。重训命令用于复现产生该权重的运行
  配置，但 nnU-Net 此版本未固定全局随机种子，重新训练会得到另一组权重，全卷指标只会在
  正常波动范围内接近，不能保证逐位一致。
- 训练日志每次出现 `Yayy! New best EMA pseudo Dice: X` 都会覆盖 `checkpoint_best.pth`。
  要复刻 Copy2，需要在日志打印 `0.8213`（约 epoch 48）时保存一份快照：

  ```bash
  cp .../fold_0/checkpoint_best.pth .../fold_0/checkpoint_best-Copy2.pth
  ```

  Copy2 就是 2026-09-05 17:21:40 在 epoch 48 验证后保存的 `checkpoint_best.pth` 快照。
- 与本文档完全一致的 4 个指标数值，请直接评估 Copy2 checkpoint（第 2 节）；
  若只是想验证配置，可跑完训练后执行 `reproduce_metrics.py --predict`。

预训练权重说明：2026-09-05 训练时 `/root/autodl-tmp/Downstream/VoComni_nnunet.pt` 确实存在
（9/5 会话记录里有该路径的 `EXISTS` 检查输出），因此 **Copy2 是 VoComni 预训练 + 微调的结果**。
加载方式为按 key/shape 选择性加载，实测匹配 **282/292**（encoder 96/96；10 个
`decoder.seg_layers.*` 因 VoComni 为 21 类、本模型 3 类而保持初始化）。

项目内已恢复该权重：

```
/root/autodl-tmp/Downstream_2/VoComni_nnunet.pt   # md5 ef44f2fb457ea3652c3566848b40b338
```

指标复现不依赖该权重（checkpoint 已包含全部网络参数），它只在重训时使用；做从头训练
消融时设置 `VOCOMNI_NNUNET_CKPT=""` 即可（等价于旧的 `phaser_scratch`）。

## 5. 预测测试集（完整命令）

预测对象：`Dataset102_LiTS/imagesTs` 的 27 例；协议与第 1 节一致：**无 TTA**。
架构环境变量必须与训练一致，否则 checkpoint 里的 `pfcr_modules.*` / `small_head.*`
会被当作 unexpected keys 跳过，预测结果就不是训练模型。

```bash
# --- 路径与环境 ---
export nnUNet_raw=/root/autodl-tmp/nnUNet_data/nnUNet_raw
export nnUNet_preprocessed=/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed
export nnUNet_results=/root/autodl-tmp/nnUNet_data/nnUNet_results_retrain
export PYTHONPATH=/root/autodl-tmp/Downstream_2/nnUNet
export CUDA_VISIBLE_DEVICES=0

# --- 与训练一致的架构开关 ---
export PHASER_PFCR=1
export PHASER_PFCR_USE_PRIOR=1
export PHASER_SMALL_HEAD=1
export PHASER_PC_SCALES=3
export PHASER_PC_MAX_VOXELS=1048576
export PHASER_PCSF=0
export PHASER_LWFD=0
export PHASER_ACG=0
export PHASER_T_RAF=none
export PHASER_LA_ACG=0
export VOCOMNI_NNUNET_CKPT=""        # 推理只用 checkpoint，跳过预训练加载

/root/autodl-tmp/envs/mamba/bin/nnUNetv2_predict \
  -i /root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/imagesTs \
  -o /root/autodl-tmp/nnUNet_data/nnUNet_results_retrain/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/test_predictions_100ep_no_tta_clean \
  -d Dataset102_LiTS \
  -c 3d_fullres \
  -tr nnUNetTrainer_phaser \
  -p nnUNetPlans \
  -f 0 \
  -chk checkpoint_best.pth \
  --disable_tta \
  -device cuda -npp 1 -nps 1
```

`-chk` 可选值（重训运行的 `fold_0/` 目录）：

| checkpoint | 含义 |
|---|---|
| `checkpoint_best.pth` | EMA pseudo Dice 最佳（与 Copy2 的选择口径一致） |
| `checkpoint_best_tumor_dice.pth` | tumor pseudo Dice 最佳 |
| `checkpoint_latest.pth` | 最后保存的 epoch |

预测 27 例约 40–60 分钟。完成后算 24 例指标（`--no-check` 只输出数值，不与 Copy2 预期比对）：

```bash
/root/autodl-tmp/envs/mamba/bin/python /root/autodl-tmp/Downstream_2/reproduce_metrics.py \
  --pred-dir /root/autodl-tmp/nnUNet_data/nnUNet_results_retrain/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/test_predictions_100ep_no_tta_clean \
  --no-check
```

若要复现第 1 节的 4 个指标（Copy2），去掉 `--pred-dir` 和 `--no-check` 即可（默认读取
`test_predictions_100ep_no_tta_clean` 并逐项 PASS/FAIL 比对）：

```bash
/root/autodl-tmp/envs/mamba/bin/python /root/autodl-tmp/Downstream_2/reproduce_metrics.py
```

## 6. 结果（本次重训 vs Copy2）

三组权重在同一协议下的 24 例全卷指标（逐病例均值 ± 总体标准差，ddof=0，无 TTA）：

| 权重 | 选择依据 | Liver Dice | Liver IoU | Tumor Dice | Tumor IoU |
|---|---|---:|---:|---:|---:|
| Copy2（9/5 运行，epoch 48） | best EMA 0.8213 | 0.951908 ± 0.027051 | 0.909411 ± 0.045781 | 0.702935 ± 0.152399 | 0.560872 ± 0.162564 |
| 重训 `checkpoint_best.pth` | best EMA 0.8563 | 0.951957 ± 0.029525 | 0.909749 ± 0.050822 | **0.746671 ± 0.110168** | **0.607231 ± 0.131280** |
| 重训 `checkpoint_best_tumor_dice.pth` | best tumor pseudo Dice 0.8864 | **0.953924 ± 0.026661** | **0.913091 ± 0.046497** | 0.657817 ± 0.178278 | 0.514364 ± 0.183784 |

预测目录：

| 权重 | 预测目录 |
|---|---|
| Copy2 | `nnUNet_results/Dataset102_LiTS/.../fold_0/test_predictions_100ep_no_tta_clean/` |
| 重训 EMA-best | `nnUNet_results_retrain/Dataset102_LiTS/.../fold_0/test_predictions_100ep_no_tta_clean/` |
| 重训 best_tumor | `nnUNet_results_retrain/Dataset102_LiTS/.../fold_0/test_predictions_100ep_tta_clean/` |

（目录名 `..._tta_clean` 只是命名沿用；该次推理约 65 s/例，与 no-TTA 那次一致，实际未开 TTA。）

### 6.1 结论

1. **重训 EMA-best 优于 Copy2**：Tumor Dice +0.0437（0.7029 → 0.7467），配对 t-test
   p=0.0112，改善 17/24 例；肿瘤 Dice 标准差从 0.152 降到 0.110；肝脏指标基本持平。
2. **best_tumor_dice 明显更差**：Tumor Dice 0.6578，低于 Copy2 和 EMA-best（vs EMA-best
   p=0.0058，仅 8/24 例改善）。它的 patch 级 tumor pseudo Dice 最高（0.8864），全卷 case
   Dice 却最低——再次说明 **patch 级 pseudo dice 不能用来选 checkpoint**。
3. 模型选择应以全卷滑窗 case Dice 为准；本次三组中应使用重训的 `checkpoint_best.pth`。

### 6.2 best_tumor 的典型崩塌病例（全卷 Tumor Dice）

| 病例 | Copy2 | EMA-best | best_tumor |
|---|---:|---:|---:|
| LiTS_021 | 0.711 | 0.782 | **0.271** |
| LiTS_075 | 0.753 | 0.839 | **0.466** |
| LiTS_011 | 0.769 | 0.713 | **0.443** |
| LiTS_118 | 0.855 | 0.894 | **0.686** |

逐病例 CSV：

```
nnUNet_results_retrain/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/
├── test_predictions_100ep_no_tta_clean/metrics_24cases_dice_iou.csv   # EMA-best
└── test_predictions_100ep_tta_clean/metrics_24cases_dice_iou.csv      # best_tumor_dice
```

注意：以上为单 fold、单 seed、24 例内部测试集结果；重训到 ~epoch 101（Copy2 为 epoch 48），
可作为内部对比，尚不足以作为论文级证据。

## 7. 目录结构

```
Downstream_2/
├── README.md                 # 本文档
├── reproduce_metrics.py      # 预测 + 指标复现脚本
├── VoComni_nnunet.pt         # 预训练骨干（仅重训需要，被 .gitignore 忽略）
├── .gitignore
└── nnUNet/                   # 本项目 fork 的 nnU-Net（最小化）
    └── nnunetv2/
        ├── inference/        # 滑窗推理 / 导出（复现必需）
        ├── preprocessing/    # CT 归一化 / 重采样（复现必需）
        ├── utilities/
        ├── evaluation/
        └── training/
            ├── loss/         # 含 focal_tversky（phaser 使用）
            └── nnUNetTrainer/
                ├── nnUNetTrainer.py          # 基类（仅保留必要改动）
                ├── nnUNetTrainer_phaser.py   # Phaser 训练器
                └── phaser/                   # PhaserUNet / PFCR / 损失
                    ├── unet.py
                    ├── modules.py
                    ├── losses.py
                    └── dim_utils.py
```

复现用的 checkpoint、数据集与预测结果位于 `nnUNet_data`（外部路径，不随仓库移动）；
VoComni 预训练骨干随仓库放在根目录（被 `.gitignore` 忽略，仅重训需要）。

## 8. 已知坑（复现失败基本都出自这里）

1. **checkpoint 选错**：必须用 `_old_arch_backup/checkpoint_best-Copy2.pth`；run 目录里的
   `checkpoint_best.pth` 已被后续 epoch 覆盖（0.86 EMA，但全卷指标不同）。
2. **必须 `--disable_tta`**：开启镜像 TTA 结果会变差，无法复现该组数值。
3. **代码版本**：旧的 `/root/autodl-tmp/Downstream` 已加入 uncertainty / scale_aligned 等
   新模块，与该 checkpoint 不兼容；本仓库的 phaser 代码与 Copy2 结构完全一致
   （0 missing / 0 mismatched keys，已逐 shape 验证）。
4. **不要用 `PHASER_OVERSAMPLE=1.0` + tumor oversampling**：9/9 的运行用
   `oversample=1.0` + `tumor_oversample=0.8` 训练，模型在纯背景 tile 上大面积幻觉前景
   （空气 patch 也能预测 60% 前景），全卷 Dice 崩到 0.24/0.11。Copy2 用的是 `oversample=0.66`、
   无 tumor oversampling。
5. **Pseudo dice ≠ 全卷 Dice**：训练日志的 `Pseudo dice` 是前景过采样的 patch 级指标，
   不能作为病例级结果；本复现用的是完整 CT 的滑窗推理。第 6.2 节 `best_tumor_dice`
   的崩塌（pseudo 最高、全卷最低）是直接证据。
6. **3 例剔除是 GT 决定的**：`LiTS_032/106/115` 的 label2（肿瘤）体素为 0，剔除后可避免
   分母为 0；其余 24 例全部保留，不做任何挑选。
7. **GT 来源**：Dataset102 raw 里没有 `labelsTs`，测试集 GT 取自
   `nnUNet_raw/Dataset101_LiTS/labelsTr`（与 LiTS 官方标签一致）。

## 9. 资产路径

| 资产 | 路径 |
|---|---|
| 预测 checkpoint | `/root/autodl-tmp/nnUNet_data/nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/_old_arch_backup/checkpoint_best-Copy2.pth` |
| 预训练骨干 | `/root/autodl-tmp/Downstream_2/VoComni_nnunet.pt` |
| 测试图像 | `/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/imagesTs` |
| 测试 GT | `/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset101_LiTS/labelsTr` |
| 训练/验证数据 | `/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset102_LiTS` |
| 训练日志（配置出处） | `.../fold_0/training_log_2026_9_5_15_36_00.txt` |
| 环境 | `/root/autodl-tmp/envs/mamba` |

## 10. 清理与备份记录

- 清理日期：2026-09-10
- 本次删除：`monai/`、`U-TTT-main/`、`paper/`、`submission/`、`predictions_phaser_liver/`、
  `verification/`、`docs/`、`output/`、`scripts/`、`.codex-build/`、`.ipynb_checkpoints/`、
  全部根目录旧文档/脚本/PPT/SVG、
  nnUNet 内 PACT/freqmamba/swin/vit/nets/variants 等实验代码
- 恢复：`VoComni_nnunet.pt` 于同日恢复到项目根目录（确认它是训练管线的一部分；
  与 `/root/autodl-tmp/PhaSER/VoComni_nnunet.pt`、备份包内副本 md5 一致）
- 安全备份：`/root/autodl-tmp/Downstream_2_backup_20260910.tar.gz`（408 MB，含清理前全部文件）
- `opencode.json` 因内含明文 API key 已删除，该 key 建议作废重发
