# PhASER-3D 核查报告与改进方案（PhASER-H）

- 核查日期：2026-09-10
- 核查对象：`/root/autodl-tmp/Downstream_2`（最小复现仓库）
- 核查权重：`checkpoint_best-Copy2.pth`（epoch 49，`_best_ema = 0.8213`）
- 核查数据：Dataset102_LiTS fold 0 测试集 24 例（剔除 3 例无肿瘤 GT），预测目录 `test_predictions_100ep_no_tta_clean`
- 核查方法：源码审查 + checkpoint 参数统计 + 真实病例前向探针 + 逐病例/逐病灶/逐体素误差分解

> 结论先行：当前 PhASER-3D 的结果（Tumor Dice 0.7029）不是"完整 PhASER-3D"的结果，而是只跑了 1/4 训练计划、只开了主损失 + PC ranking 的半成品；其核心卖点 PFCR 在实测中已退化为静态全图带通滤波，"相位条件路由"并未发生。改进方向应从"通用边缘检测"升级为"病灶假设检验"，并把优化目标从体素搬到病灶。

---

## 一、当前模型实际是什么

### 1.1 配置事实

| 项目 | 事实 |
|---|---|
| 被测权重 | `_old_arch_backup/checkpoint_best-Copy2.pth`，epoch 49，EMA pseudo Dice 0.8213 |
| 架构 | VoComni PlainConvUNet（6 stage，32/64/128/256/320/320，patch 128³）+ PFCR×5 + Small-Tumor Head |
| 预训练加载 | 282/292 keys；encoder 96/96；10 个 `decoder.seg_layers.*` 因 VoComni 21 类 vs 本模型 3 类保持初始化 |
| 实际生效模块 | 主损失（DiceCE + FocalTversky w=0.15）+ PC boundary ranking（w=0.2） |
| 未生效模块 | boundary / spectral / lesion / small-head / PFCR budget 全部在 Phase B（epoch ≥ 60）才启动，epoch 49 时**从未启用** |
| 指标 | Liver Dice 0.951908±0.027051，Tumor Dice 0.702935±0.152399，Tumor IoU 0.560872±0.162564 |

训练计划为 200 epoch，Copy2 只到第 49 epoch；`training_log_2026_9_5_15_36_00.txt` 显示该 run 在约 epoch 100 被中断，从未跑满。

### 1.2 关键含义

README 中"激活模块"一栏写的 spectral / boundary / lesion / small-head，在该权重上全部未生效。所有基于该 checkpoint 的结论，都只能说明"VoComni 骨干 + 主损失 + PC ranking"的能力，不能代表完整架构。

---

## 二、逐模块"是否发挥到极致"——用实测证据说话

用 checkpoint 权重 + 真实病例前向探针（LiTS_001 预处理数据，肿瘤中心 128³ patch，含 12,970 肿瘤体素）实测。

### 2.1 PFCR：核心卖点事实上没有实现

- 5 个尺度的 route 图**空间标准差只有 1e-4 ~ 1e-3，空间极差 ≤ 1%**（见下表）。`softmax(router(...))` 输出几乎是一个常数，"逐体素相位条件路由"退化成**每个尺度一个固定的频带混合系数**。
- checkpoint 中 router 最后一层权重 `mean|W| ≈ 2e-3 ~ 9e-3`，而 bias 达 0.33（scale 2），路由完全由 bias 决定，PC 输入被淹没。
- **5 个尺度的可学习 log-Gabor 中心频率完全相同**：rho = [0.3216, 0.1714, 0.0937]，初始值为 [0.3333, 0.1587, 0.0756]。"每尺度可学习频率选择"没有发生。
- PFCR 残差门 `tanh(residual_gate)` 非零（-0.2989 / +0.1072 / +0.4929 / +0.3720 / -0.3721），说明它确实在调制特征，但只是**静态、全图共享的多尺度带通滤波**。

实测 route 统计（LiTS_001）：

| scale | 空间尺寸 | mean route [low, mid, high] | 空间 std | 空间极差 |
|---|---|---|---|---|
| 0 | 128³ | [0.3447, 0.3268, 0.3286] | [0.00055, 0.00030, 0.00025] | [0.0051, 0.0028, 0.0023] |
| 1 | 64³ | [0.3338, 0.3275, 0.3387] | [0.00011, 0.00024, 0.00018] | [0.0013, 0.0026, 0.0018] |
| 2 | 32³ | [0.2290, 0.4107, 0.3602] | [0.00099, 0.00084, 0.00015] | [0.0086, 0.0071, 0.0015] |
| 3 | 16³ | [0.2794, 0.3399, 0.3807] | [0.00166, 0.00022, 0.00144] | [0.0104, 0.0012, 0.0093] |
| 4 | 8³ | [0.3111, 0.3570, 0.3319] | [0.00092, 0.00087, 0.00008] | [0.0041, 0.0038, 0.0004] |

### 2.2 PhaseCongruency：最该发力的浅层被降了分辨率

- `pc_max_voxels = 2^20`，skip0 在 128³ = 2.1M > cap → PC 实际在 **64³** 上计算再上采样回 128³。小病灶/边界恰恰只存在于最浅层。
- 深层 skip（8³）PC 均值仅 0.02–0.03，>0.3 的体素占 2–3%，基本是死的（log-Gabor 波长 3/5.8/10.7 体素 > 8³ 感受野）。
- 频率网格用 `d=1`（数字频率），未使用真实 spacing（1.0/0.779/0.779 mm），不同轴、不同层代表的物理尺度不一致。

实测 PC 图统计（LiTS_001）：

| scale | 形状 | mean | std | frac>0.3 | max |
|---|---|---|---|---|---|
| 0 | 128³ | 0.0973 | 0.1582 | 0.1407 | 0.8335 |
| 1 | 64³ | 0.0620 | 0.1384 | 0.0885 | 0.8550 |
| 2 | 32³ | 0.0474 | 0.1119 | 0.0587 | 0.7378 |
| 3 | 16³ | 0.0221 | 0.0772 | 0.0227 | 0.7231 |
| 4 | 8³ | 0.0314 | 0.0824 | 0.0273 | 0.5049 |

### 2.3 Small-Tumor Head / tumor prior：完全未训练 + 概率 bug

- 实测 prior 在 GT 肿瘤内均值 **0.518**，在肝内均值 0.5067，在背景均值 **0.5358** —— 背景比肿瘤还高，**毫无判别力**。
- 原因一：该头直接监督在 Phase B（epoch ≥ 60）才启用，Copy2 是 epoch 49，从未学过。
- 原因二：**双重 sigmoid**。`unet.py:258` 已做 `sigmoid(small_logits[:,1:2])`，`modules.py:440` 又做 `prior.sigmoid()`；而 small head 用 2 类 CE 训练，正确概率应是 `softmax(small_logits, dim=1)[:,1]`。先错概率语义、再压缩到 (0.5, 0.73)，PFCR 的条件输入是噪声。

### 2.4 主损失与训练协议

- Focal Tversky α=0.3 / β=0.7（FN 权重高）+ 无对称 boundary 项 + 训练不足 → 边界双向漂移。
- **模型选择指标错了**：用 patch 级 EMA 前景伪 Dice 选 checkpoint。实测 EMA 0.8213 → 全卷肿瘤 Dice 0.7029；后续 EMA 0.86 的 checkpoint → 全卷肿瘤 Dice 仅 0.6593。**选模型的方向与肿瘤指标相反**。
- `pc_boundary_ranking_loss` 的负样本只有"肿瘤内部 + 外周肝实质环"，**没有血管这类硬负样本**，算子学不会区分血管与病灶。

---

## 三、预测误差画像（方案设计依据）

24 例全体积逐体素/逐病灶统计（预测目录 `test_predictions_100ep_no_tta_clean`，GT 为 Dataset101_LiTS labelsTr）。

### 3.1 总体指标

| 指标 | 数值 |
|---|---|
| Tumor Dice | 0.7029（中位 0.7202，std 0.1524） |
| Precision / Recall（病例宏平均） | 0.7151 / 0.7364 |
| Precision / Recall（体素池化） | 0.8662 / 0.8050 |
| HD95 | 均值 48.43 mm，中位 34.14 mm |
| ASSD | 8.59 mm |
| 边界 Dice（2mm 容差） | 0.3499 |
| FN / FP 体素总量 | 1,085,956 / 692,502 |
| GT / 预测肿瘤体素总量 | 4,482,376 / 4,088,922 |

### 3.2 病灶级

| 分层 | 数量 | 漏检 | 平均 Dice | 中位 Dice |
|---|---|---|---|---|
| 全部 GT 病灶（≥10 体素） | 251 | 31（12.4%） | 0.5715 | 0.6945 |
| tiny [0,100) | 20 | 7 | 0.2082 | 0.0832 |
| small [100,1000) | 122 | 22 | 0.4906 | 0.6134 |
| medium [1000,10000) | 74 | 2 | 0.6967 | 0.7855 |
| large [10000,∞) | 35 | 0 | 0.7961 | 0.8435 |

### 3.3 边界偏移（有符号距离，单位体素）

以 GT 肿瘤表面体素到预测表面的有符号平均距离衡量（负=预测外扩/过分割，正=预测内缩/欠分割）：

- 24 例平均 **+2.06**，中位 **+1.06**
- **18/24 例为欠分割**（正），6/24 为过分割（负）
- 最严重：LiTS_092 +19.54、LiTS_112 +7.67、LiTS_130 +5.13、LiTS_099 +3.63、LiTS_122 +3.00
- 过分割代表：LiTS_016 −1.48、LiTS_003 −1.17、LiTS_097 −1.13

边界偏移是**双向、病例相关**的，因此全局腐蚀/膨胀无效：全体平均 erode1 −0.0832、dilate1 −0.0380（均为 Dice 下降）。但个别病例可获较大局部收益，例如 LiTS_003 erode2 +0.114、LiTS_130 dilate3 +0.088、LiTS_036 dilate2 +0.072。

### 3.4 低对比度 × 小尺寸崩溃区

以病灶内部 vs 5 体素肝内环的 CNR = |μ_in − μ_out| / σ_out 分层：

| 条件 | 数量 | 检出率 | 平均 Dice |
|---|---|---|---|
| CNR < 1.0 且 < 500 体素 | 15 | **0.467** | 0.202 |
| CNR < 1.0 且 ≥ 500 体素 | 31 | 0.968 | 0.604 |
| CNR ≥ 1.0 且 < 500 体素 | 94 | 0.819 | 0.449 |
| CNR ≥ 1.0 且 ≥ 500 体素 | 111 | 0.955 | 0.716 |

结论：**低对比度 × 小尺寸是真正的崩溃区**，直接支撑"相位/对比度不变证据"的研究动机。

### 3.5 假阳性结构

排除与 GT 肿瘤重叠的连通域后，≥10 体素的 FP 连通域共 **90 个**：

- 65/90 完全位于肝内（血管/囊肿/脂肪等混淆结构），典型 HU 38–78；
- 18/90 完全位于肝外；
- FP 体素总量 74,910；其中肝内约 24%（肝外大团块主导）
- 最大肝外 FP：**LiTS_028 37,875 体素（HU≈41）**；LiTS_122 6,549 体素（HU≈74）
- 典型肝内 FP：LiTS_036 4,697 体素（HU≈38）、LiTS_072 1,612（HU≈78）、LiTS_109 1,314（HU≈61）、LiTS_130 1,283（HU≈76）

### 3.6 大肿瘤整体漏（LiTS_130）

- GT 肿瘤 1,383,692 体素；预测 690,874；漏 728,765（53%）
- 漏掉部分平均 HU **92.1**，已预测部分平均 HU **73.2**，GT 整体 83.2
- 漏掉体素到预测肿瘤的平均距离 5.8 体素（最大 44.3），即"只分了低密度那一半"

**误差本质一句话：当前模型是"外观/幅度驱动"的检测器——它把典型低密度瘤体分出来，但（a）分不清血管与病灶、（b）在低对比度小病灶上失效、（c）边界无校准、（d）会跑到肝外。而本应补这些短板的"相位证据链"，从分辨率、概率语义、监督时机到路由机制，没有一环真正工作。**

---

## 四、改进方案：PhASER-H —— 把"相位一致性"改造成"病灶假设检验"

> 定位：不做模块堆叠。核心是**一个机制**（相位证据的判别化与标定），加**一套决策/训练协议**（病灶级优化），二者互为因果。
> 目标 venue：MICCAI / IEEE TMI / Medical Image Analysis（单数据集 + 3D 医学任务的现实顶会顶刊区间）。

### 4.1 核心科学问题

现有频域分割（FDConv CVPR'25、FADC CVPR'24、FFR CVPR'25、WaveFormer、HNOSeg、FreqU-FNet）**全部只用幅度谱**；现有 phase congruency 用法都把它当**通用边缘算子**。本文要回答：

> 在低对比度、噪声、血管/包膜干扰下，相位证据要成为有用的分割信号，必须先回答"这条边是不是病灶的边"。这本质上是一个**假设检验**，而不是一个边缘响应。

### 4.2 机制（一个东西的四个面，不是四个模块）

#### A. 判别式单演相位分解（Discriminative Monogenic Decomposition）

把现在被压成 1 个标量的 PC 还原成**完整单演向量**：

```
y_s(x) = (e_s, o_s,z, o_s,y, o_s,x)     # 1 个偶部 + 3 个 Riesz 奇部
```

- 用真实 spacing + encoder stride 构造频率网格（物理尺度一致，消除当前 d=1 的错误）；
- 学习两组相位原子字典：**病灶边界字典 U_L** 与 **混淆结构字典 U_C**（血管/包膜/噪声）；
- 证据定义为竞争比值 `E_L = ||U_L^T y|| / (||U_L^T y|| + ||U_C^T y|| + ε)`，天然对比度不变；
- 监督用 **hard-negative ranking**：正样本 = GT 肿瘤边界壳层；负样本 = ① 肝内 Frangi 血管响应高的区域 ② 模型在线 FP 连通域 ③ 肿瘤内部/外周肝实质。现有 `pc_boundary_ranking_loss` 只有"边界 vs 自身内部+外环"，**没有血管这类硬负样本**，所以算子永远学不会区分血管与病灶。

这一面直接解决"90 个 FP 中 65 个肝内混淆结构"和"低 CNR 小病灶"两个问题。

#### B. 噪声标定的可靠性场（Reliability field）

- 用局部高频残差/一个轻量噪声头估计 σ(x)（CT 局部噪声近似平稳）；
- 把 E_L 标准化为显著性 `R = E_L / (σ·√N_eff)`（t 统计量形式），取代 Kovesi 的启发式 Rayleigh 阈值；
- 低可靠区不输出证据（而不是靠固定阈值），使门控**真正空间变化**。这是可解释、可校准的量，可直接用于病灶级阈值决策。

#### C. 方向场驱动的方向子带调制（Orientation-aware subband routing）

- Riesz 奇部给出**局部方向 u(x)**；3D Haar 的方向性子带（LLH/LHL/HLL/LHH/…）本身编码方向；
- 用 u(x) 与子带方向的对齐度 + 证据 E_L 共同调制高频子带，而不是像现在把 7 个高频子带按"高通轴个数"分成 3 组、再乘一个近似常数的 route；
- 同时**取消浅层 FFT 的 2^20 cap**（或改用空间域可分离递归滤波），让 e0 的相位证据保持 128³ 全分辨率。
- 这一面替代 PFCR 的"softmax 三路路由"，从根上消除"路由退化为常数"的问题（审计已证明它必然退化）。

#### D. 病灶级决策与训练协议（把指标从体素搬到病灶）

针对误差画像逐条对症，且都是"协议"不是"模块"：

1. **病灶实例均衡采样/加权**：预计算全卷连通域与 mm³ 体积，按实例采样（现在是 patch 级前景过采样，且 `compute_lesion_importance_map` 的 100/1000 阈值是体素、且从未用于评估）；
2. **对称边界目标**：把 Focal Tversky 的 α/β 改为对称或加 symmetric boundary loss，消除当前边界双向漂移（18/24 欠、6/24 过）；
3. **肝域约束**：启用/重写 inside-liver penalty，并让 liver prior 从 epoch 0 就有监督 —— 直接消掉肝外 37.9k 体素级假瘤；
4. **大肿瘤成分完整性**：对 LiTS_130 这类"只分会分的那一半"，用证据场加权的区域补全（在粗分割基础上做证据约束的区域生长/图割，可微展开），目标是把同质成分（HU 92 部分）纳入；
5. **模型选择改为全卷肿瘤/病灶级指标**（当前 EMA 伪 Dice 与肿瘤 Dice 反相关）；
6. **修正所有已确认 bug**：
   - 双重 sigmoid → `softmax(small_logits, 1)[1]`；
   - small head 从 epoch 0 监督；
   - PC 频率加单调性/分离度约束并在日志里报告学到的 λ；
   - 跑满 Phase B/C。

### 4.3 为什么这个创新性够

| 与最近工作的区别 | 本方案的新意 |
|---|---|
| FDConv/FADC/FFR/WaveFormer/HNOSeg/FreqU-FNet | 它们全在**幅度谱**上做文章；本方案以**相位/单演向量**为证据，且不是复刻 PC |
| 经典 phase congruency（Kovesi'99）与医学 PC 分割 | 它们是**通用边缘检测 + 固定滤波器 + 启发式阈值**；本方案把它变成**病灶 vs 混淆结构的假设检验 + 噪声标定 + 方向保持** |
| PFCR（本仓库） | 它把多尺度/方向压成一个 PC 标量再学路由（已实测退化为常数）；本方案保留尺度/方向向量，且用方向场直接驱动方向子带 |
| 小病灶三件套（small head / lesion weight / focal tversky） | 本方案用**实例均衡 + 病灶级指标**统一替代，且给出可证伪的分层证据（CNR×尺寸） |

可写成的主贡献一句话：

> 我们证明相位分割的瓶颈不是相位算子本身，而是缺少判别性（病灶 vs 血管）、可靠性标定（噪声）与病灶级决策；我们给出统一的相位假设检验框架，并在低对比度小病灶检测、肝外假瘤抑制和边界校准上同时取得提升。

### 4.4 预期收益（基于本次误差分解的保守估计）

| 误差来源 | 现状 | 机制对应 | 目标 |
|---|---|---|---|
| 肝外假瘤 | 90 个 FP 连通域，最大 37.9k 体素 | D-3 肝域约束 | FP/scan 3.75 → <2 |
| 小病灶漏检 | 31/251，低 CNR 小病灶检出 46.7% | A+B+D-1 | 病灶召回 0.876 → ≥0.92 |
| 边界双向漂移 | HD95 中位 34mm，边界 Dice 0.35 | A+C+D-2 | HD95 中位 <15mm |
| 大肿瘤欠分割 | LiTS_130 漏 53% | C+D-4 | 该例 Dice +0.10 以上 |
| 模型选择 | EMA↑ 而肿瘤 Dice↓ | D-5 | 不再负相关 |

综合目标：**同一 24 例肿瘤 Dice 0.70 → 0.74–0.78，且病灶级指标显著改善**。

参考：仅把 PFCR 换成 scale-aligned 的 `sa_pfcr_real` 预测（`nnUNet_results/sa_pfcr_real/test_predictions_best_tumor_dice`）在同一 24 例上是 Tumor Dice 0.722445±0.130640，说明"结构微调"只有 +2 点，必须靠上面的机制才能突破。

---

## 五、落地路线（按优先级）

### P0 · 1 周内（不训练也能改，止血）

1. `modules.py:440` 去掉第二重 sigmoid；`unet.py:258` 改 `softmax(small_logits, 1)[:,1:2]`；
2. small head 监督提到 epoch 0（或至少与 PC ranking 同步）；
3. 模型选择改为全卷 tumor Dice / 病灶 F1（`on_validation_epoch_end`）；
4. `pc_max_voxels` 在 e0/e1 取消或改空间域滤波；
5. 修 Focal Tversky 的 α/β 或加对称边界项；
6. 跑满 200 epoch（当前所有结论建立在 epoch 49 上）。

### P1 · 2–4 周（核心机制 pilot）

7. 实现单演向量 + 病灶/混淆双字典 + hard-negative ranking（替换 `pc_boundary_ranking_loss` 的负样本集合）；
8. 噪声标定可靠性场；把 PFCR 的三路 softmax 路由替换为 E_L 直接门控 + 方向子带调制；
9. 在 fold0 上做 B 系列对照：
   - B0：VoComni PlainConvUNet baseline
   - B1：B0 + gradient boundary gate（幅度域对照）
   - B2：B0 + fixed phase congruency gate（固定 PC）
   - B3：B0 + learnable phase congruency（可学习尺度）
   - B4：B3 + 判别式双字典
   - B5：B4 + 噪声标定可靠性
   - B6：完整 PhASER-H
   - 所有配置参数量、训练预算、推理设置可比。

### P2 · 1–2 月（证据链）

10. 病灶实例采样与病灶级损失；
11. 干预实验：推理时移除某个尺度的证据，测病灶级损失变化 `U_{j,s} = L_j(p_{−s},Y) − L_j(p_full,Y)`；
12. 报告 lesion-FROC（0.5/1/2 FP/scan）、CNR×尺寸分层、HD95/NSD、FP/scan；
13. 外部验证：TriALS 非增强 CT（低对比度场景，本地有仓库代码）或受控重建核/噪声退化实验。

### 止损条件（可证伪）

若 B4/B5 在同参数同预算下不能在低 CNR 小病灶召回或 FP/scan 上稳定优于 B2/B3，则"判别式相位"假设不成立，应缩回到"可学习 PC + 病灶级协议"的应用型论文（JBHI/CIBM 档次）。

---

## 六、已确认的 bug 清单（供修复对照）

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| 1 | `phaser/unet.py:258` + `phaser/modules.py:440` | 双重 sigmoid；small head 用 2 类 CE，正确概率应为 softmax | tumor prior 语义错误，范围被压到 (0.5,0.73) |
| 2 | `nnUNetTrainer_phaser.py:593` | small head 监督仅在 Phase B（epoch ≥ 60） | Copy2（epoch 49）该头完全未训练 |
| 3 | `phaser/modules.py:253-258` | `pc_max_voxels=2^20`，skip0 在 64³ 算 PC 再上采样 | 最浅层（小病灶关键层）相位证据分辨率减半 |
| 4 | `phaser/modules.py:417-423` | router 最后一层零初始化 + 小权重 | 路由被 bias 主导，空间上近似常数 |
| 5 | `phaser/modules.py:181-189` | 各尺度 rho/sigma 独立初始化但无分离约束 | 实测 5 个尺度频率完全相同 |
| 6 | `nnUNetTrainer_phaser.py:818-833` | 用 patch 级 EMA 前景伪 Dice 选 checkpoint | 与全卷肿瘤 Dice 反相关（0.8213→0.7029 vs 0.86→0.659） |
| 7 | `phaser/losses.py:325-410` | PC ranking 负样本无血管等硬负样本 | 算子无法区分血管与病灶 |
| 8 | `nnUNetTrainer_phaser.py:384-391` | Focal Tversky α=0.3/β=0.7 | 边界偏移（结合训练不足，实测欠分割为主） |
| 9 | `phaser/losses.py:417-479` | `compute_lesion_importance_map` 用体素阈值 + CPU scipy | 与 mm³ 物理体积脱节、训练慢、且仅用于损失不用于评估 |
| 10 | `phaser/modules.py:202-222` | 频率网格 d=1，未用 spacing | 不同轴/层的物理尺度不一致 |

---

## 七、核查方法可复现性附录

### 7.1 关键路径

| 资产 | 路径 |
|---|---|
| 被测 checkpoint | `/root/autodl-tmp/nnUNet_data/nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/_old_arch_backup/checkpoint_best-Copy2.pth` |
| 预测目录 | 同目录 `test_predictions_100ep_no_tta_clean/` |
| 测试 GT | `/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset101_LiTS/labelsTr` |
| 测试图像 | `/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/imagesTs` |
| 预处理训练数据 | `/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset102_LiTS/nnUNetPlans_3d_fullres` |
| 训练日志（配置出处） | `.../fold_0/training_log_2026_9_5_15_36_00.txt` |
| scale-aligned 参考预测 | `/root/autodl-tmp/nnUNet_data/nnUNet_results/sa_pfcr_real/test_predictions_best_tumor_dice` |

### 7.2 本次核查执行的统计

1. 逐病例 Dice / IoU / Precision / Recall / FN / FP；
2. 逐病灶连通域匹配（≥10 体素）：检出率、per-lesion Dice、尺寸分层；
3. 表面距离：HD95、ASSD、2mm 边界 Dice、有符号边界偏移；
4. CNR 分层：病灶内部 vs 5 体素肝内环，按 CNR × 尺寸交叉分层；
5. 肝内外假阳性连通域分解（体积、HU、肝内占比）；
6. checkpoint 参数统计：PFCR residual_gate / rho / sigma / noise_k / router 权重 / proxy 权重 / small head 权重；
7. 真实病例前向探针：route 空间统计、PC 图统计、prior 判别力、patch 级 Dice。

### 7.3 环境

```
/root/autodl-tmp/envs/mamba/bin/python   # torch 2.1.0+cu118, SimpleITK 2.5.6, scipy 1.15.3, scikit-image 0.25.2
GPU: NVIDIA GeForce RTX 3090 (24 GB)
```

---

## 八、一句话总结

当前 PhASER-3D 的问题不是"模块不够多"，而是**唯一的核心机制（PFCR）在实测中已退化为静态带通滤波，而它本应补足的三个短板（低对比度小病灶、血管/肝外假瘤、边界校准）一个都没有被真正优化**；把相位从"通用边缘算子"升级为"病灶 vs 混淆结构的假设检验"，并把目标从体素 Dice 搬到病灶级指标，是同时兼顾性能提升与顶刊/顶会创新性的最短路径。
