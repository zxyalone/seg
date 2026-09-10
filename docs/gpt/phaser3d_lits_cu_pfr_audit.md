# PhASER-3D LiTS 审计与 CU-PFR 改进方案

## 结论摘要

当前 PhASER-3D 在 LiTS 肝脏肿瘤分割中的问题不是单纯“小肿瘤漏检”，而是同时存在低对比边界的欠分割，以及血管、肝实质纹理等强结构区域诱发的肿瘤假阳性。因此继续叠加小目标头、边界头、频谱损失或注意力模块不是合理方向。

建议以 **CU-PFR (Counterfactual Uncertainty-Calibrated Phase-Frequency Routing，反事实不确定性校准相位频率路由)** 替换现有 PFCR。方法保留相位一致性和 Haar 小波，但用肿瘤-肝脏决策不确定性控制频率干预，并以反事实损失验证高频路径是否真的提升真实肿瘤边界、且不制造高结构难阴性假阳性。

这不是模块堆叠：CU-PFR 将当前 PFCR 的三频带路由器改为同一控制器输出“频带比例 + 绝对干预门”，并用一个反事实风险损失对该控制器进行可证伪监督。

## 1. 审计范围与证据限制

### 1.1 已核查内容

| 组件 | 文件/数据 | 核查结论 |
|---|---|---|
| 网络注入位置 | `nnUNet/nnunetv2/training/nnUNetTrainer/phaser/unet.py` | PFCR 处理每个非 bottleneck encoder skip，随后进入标准解码器。 |
| 相位与路由 | `nnUNet/nnunetv2/training/nnUNetTrainer/phaser/modules.py` | PFCR 由 Phase Congruency 与可选肿瘤先验决定低/中/高频相对比例。 |
| 辅助损失 | `nnUNet/nnunetv2/training/nnUNetTrainer/phaser/losses.py` | 存在边界、PC 排序、频谱、病灶加权等重叠训练目标。 |
| 训练配置 | `nnUNetTrainer_phaser.py` 与 2026-09-10 日志 | 当前训练启用 PFCR prior、小肿瘤头、PC 边界、频谱、边界、病灶加权和 Focal-Tversky。 |
| 预测误差 | Dataset102 当前预测 NIfTI 对 Dataset101 LiTS 标签 | 已按病例重算 precision、recall、Dice、体积比及连通域错误。 |

### 1.2 版本一致性风险

当前评测材料不能直接混用：

- `checkpoint_best-Copy2.pth` 的时间为 2026-09-05；当前训练日志为 2026-09-10。
- 预测目录的 `metrics_24cases_dice_iou.csv` 与当前 NIfTI 的逐例肿瘤体素数和 Dice 不一致。
- 两者总体 tumor Dice 都约为 `0.7029`，但这不能证明它们来自同一次推理。

因此本文涉及病例级机制的结论只以磁盘中的实际 NIfTI 为准；历史 CSV 仅作为待核对记录。论文实验必须为每次运行保存不可变 `manifest.json`：代码 commit、checkpoint SHA256、环境变量、标签版本、推理命令、TTA 状态、预测和概率文件 SHA256。

## 2. 当前预测的错误证据

### 2.1 24 个有肿瘤病例的可复算统计

| 指标 | 均值 | 中位数 | 标准差 |
|---|---:|---:|---:|
| Tumor precision | 0.7151 | 0.7486 | 0.1701 |
| Tumor recall | 0.7363 | 0.7843 | 0.1862 |
| Tumor Dice | 0.7029 | 0.7202 | 0.1557 |
| 预测/GT 肿瘤体积比 | 1.1819 | 1.0350 | 0.9016 |

平均体积比接近 1 不代表稳定，因为漏检和过分割会相互抵消。

| 病例 | Precision | Recall | Dice | 预测/GT 体积 | 主要错误 |
|---|---:|---:|---:|---:|---|
| LiTS_005 | 0.160 | 0.833 | 0.268 | 5.21x | 严重过分割；6 个预测连通域中 5 个几乎无 GT 重叠。 |
| LiTS_054 | 0.744 | 0.413 | 0.531 | 0.56x | 典型保守内收/欠分割。 |
| LiTS_077 | 0.587 | 0.410 | 0.483 | 0.70x | 漏检与碎片假阳性同时存在。 |
| LiTS_092 | 0.393 | 0.339 | 0.364 | 0.86x | 错位与碎片化；14 个预测连通域中 9 个为假阳性。 |
| LiTS_130 | 0.948 | 0.473 | 0.631 | 0.50x | 高 precision 但大范围肿瘤漏检。 |
| LiTS_097 | 0.800 | 0.961 | 0.873 | 1.20x | 成功病例，说明主干具备大病灶定位能力。 |

三个无肿瘤 GT 病例依然有肿瘤预测：`LiTS_032=151`、`LiTS_106=1758`、`LiTS_115=3189` 个体素。这是必须独立报告的临床安全风险。

### 2.2 由数据得出的约束

1. 不能只增强高频或边界召回，`LiTS_005` 和无肿瘤病例说明非病灶结构也有强边缘证据。
2. 不能仅做假阳性后处理，`LiTS_054/077/092/130` 的 recall 已明显不足。
3. 优化目标应是肿瘤相对于肝脏的局部判别，不是泛化的边缘增强。
4. 必须同时评价病灶级检出和 high-PC 非肿瘤区域的假阳性。

## 3. 现有模块的作用与不足

### 3.1 PFCR 的核心缺口

当前 `PhaseFrequencyCoherenceRouting` 的过程为：

```text
PC = PhaseCongruency(proxy(skip))
route = softmax(router([PC, tumor_prior]))
routed = IWT(route_low * LLL,
             route_mid * {LLH, LHL, HLL},
             route_high * {LHH, HLH, HHL, HHH})
skip_out = skip + tanh(residual_gate) * routed
```

该实现具有相位一致性、子带分解和残差零初始化的优点，但无法判断：

1. 高 PC 是否真是肿瘤-肝脏边界，而非血管、肝包膜、噪声或实质纹理；
2. 主分割器是否已经有稳定正确的判断，是否真正需要额外频率细节；
3. 高频干预是否提高 tumor-vs-liver 判别，还是仅提升肿瘤 logit。

三个 `softmax` 权重只能表示频带的相对比例，总和恒为 1。当前只有一个全局 `residual_gate`，没有逐体素“关闭所有频率增强”的绝对控制，因此强结构难阴性仍会被迫接受某种频率组合。

### 3.2 小肿瘤头不是可靠的路由先验

当前实现为：

```text
small_logits = small_head(skips)
tumor_prior = sigmoid(small_logits[:, 1:2])
```

小肿瘤头以二分类交叉熵训练，但第 1 类 logit 直接经 sigmoid 被用作肿瘤概率。互斥二分类的相对类别概率应为 `softmax(small_logits, dim=1)[:, 1:2]`，而非单个类别 logit 的 sigmoid。这个未校准先验随后直接控制 PFCR。

即使修正为 softmax，它也只是浅层“是否肿瘤”的粗提议，不是“肿瘤和肝脏是否难以区分”的决策不确定性。它与病灶加权及 3x 肿瘤像素权重共同偏向提高 recall，却没有对应的难阴性 FP 约束。

### 3.3 相位计算与小灶定位

当前 `pc_max_voxels=2^20`。对于 `128^3` patch 的浅层特征，PC 在约 `64^3` 代理分辨率上计算再上采样，可能模糊细小病灶边界。相位机制应保留，但不应对全部 encoder 层无差别路由；更合适的位置是高分辨率 decoder 末两级的语义-细节融合。

### 3.4 现有辅助损失难以归因

当前同时启用 Dice+CE、Focal-Tversky、肿瘤边界 Dice、spectral consistency、lesion-aware weighting、小肿瘤头和 PC boundary ranking。

- Focal-Tversky 的 `alpha=0.3, beta=0.7` 对 FN 惩罚强于 FP。
- 病灶加权对小病灶肿瘤体素加 2-5 倍权重。
- 小肿瘤头再对肿瘤体素加 3 倍权重。

这些目标共同偏向召回，却没有直接抑制 high-PC 难阴性造成的肿瘤 FP。频谱一致性也不能解决该问题：encoder skip 以 `detach()` 保存，因而 encoder 是固定频谱 target；该损失主要作用于 decoder，且没有病灶语义。

结论：当前缺少的不是另一个模块或损失，而是把频率干预与肿瘤决策风险绑定的机制。

## 4. CU-PFR：反事实不确定性校准频率路由

### 4.1 设计边界

CU-PFR 是对 PFCR 的替换，不是累加：

- 保留相位一致性、Haar DWT/IWT、残差初始化和 VoCo 主干；
- 删除独立小肿瘤先验作为路由输入；
- 仅在 decoder 最后两级和对应高分辨率 skip 融合处执行；
- 用一个 controller 同时输出频带比例和绝对干预门；
- 不再对所有 encoder skip 进行频率路由。

### 4.2 由主分割器产生肿瘤-肝脏不确定性

令前一 decoder 级别的粗 logits 为 `z_c`：

```text
p = stopgrad(softmax(z_c / T))
u = H(p) / log(C)
m = p_tumor - max(p_liver, p_background)
q = (p_tumor + p_liver) * 4 p_tumor p_liver / (p_tumor + p_liver + eps)^2
b = PC(skip) * u * q
```

- `u`：归一化语义熵；
- `m`：肿瘤相对于竞争类别的 margin；
- `q`：只在肿瘤和肝脏同时有显著概率时高；
- `b`：高相位一致性与肿瘤-肝脏不确定性的交互证据；
- `stopgrad`：避免路由器操纵粗分割置信度来规避监督。

### 4.3 一个控制器、两类输出

```text
(r_low, r_mid, r_high, a) = Controller([PC, b, m, p_tumor])
r = softmax(r_low, r_mid, r_high)
a = sigmoid(a)
F_on = F + eta * a * IWT(r_low * B_low,
                          r_mid * B_mid,
                          r_high * B_high)
```

`r` 选择低/中/高频比例，`a` 为逐体素绝对干预门，允许模型完全关闭频率增强；二者由同一个 controller 生成。`eta` 零初始化或小初始化，以维持预训练主干稳定性。

### 4.4 反事实风险损失

训练时以共享 decoder 参数计算频率开启和关闭的预测：

```text
z_on = Decoder(F_on)
z_off = Decoder(F)
Delta_m = (z_tumor - z_liver)_on - (z_tumor - z_liver)_off
```

令 `B_gt` 为 GT 肿瘤窄边界，`N_hp` 为 GT 非肿瘤 high-PC 难阴性区域，`S` 为稳定低不确定区域：

```text
L_risk = mean(B_gt) softplus(delta_pos - Delta_m)
       + mean(N_hp) softplus(Delta_m + delta_neg)
       + lambda_a * mean(S union N_hp) a

L_total = L_DiceCE + lambda_risk * L_risk
```

该目标要求：真实肿瘤边界上频率路径必须增加 tumor-vs-liver margin；high-PC 非肿瘤区域不得增加该 margin；稳定区域和高风险难阴性区域应关闭 `a`。因此该路由不再依赖“频率特征与肿瘤标签相关”，而依赖可验证的局部干预收益。

`z_off` 仅在训练期通过共享 decoder 的 checkpointed 分支计算；推理仍为单分支。为控制训练成本，CU-PFR 只在末两级使用，概率 `p` 停止梯度，且可对反事实分支使用 activation checkpoint。

## 5. 错误到机制的映射

| 错误模式 | 当前 PFCR 缺口 | CU-PFR 对策 |
|---|---|---|
| LiTS_054/077/130 欠分割 | PC 不知道哪里是肿瘤-肝脏犹豫区 | `u*q` 选择肿瘤-肝脏竞争位置；边界项要求频率开启提高 margin。 |
| LiTS_005/092 过分割 | 高 PC 血管/纹理也会被频带增强，且不能局部关闭 | `a` 可关闭干预；`N_hp` 禁止难阴性上升 tumor margin。 |
| 无肿瘤病例 FP | 浅层肿瘤先验可能放大结构响应 | 移除独立 prior；缺少肿瘤-肝脏竞争或正向收益时门收缩。 |
| 小病灶细节不足 | 浅层 PC 降采样且全层无差别路由 | 仅在高分辨率 decoder 融合处按语义不确定性激活。 |

## 6. 建议的代码改动范围

1. 在 `modules.py` 中以 `CUPFRouting` 替换 PFCR 的 `PhaseFrequencyCoherenceRouting`。
2. 在 `unet.py` 中将路由位置从全部 encoder skip 改为 decoder 最后两级的 skip 融合点。
3. 删除 PFCR 的 `tumor_prior` 输入和 `SmallTumorDetailHead` 训练/推理路径。
4. 在 trainer 中实现 `counterfactual_frequency_risk_loss`，并缓存 `a`、`r`、`PC`、`Delta_m` 用于审计。
5. 训练损失收敛为 `DiceCE + L_risk`，不要同时开启旧 PC ranking、spectral、lesion weight、Focal-Tversky、小肿瘤头和额外 boundary loss。
6. 推理时保存 softmax、PC、绝对门 `a`、频带比例和 `Delta_m`，支持机制可视化。

删除旧辅助项是方法可归因的必要条件。若 CU-PFR 单独无效，应否定核心假设，而非立即把旧模块重新叠加。

## 7. 顶会/顶刊级验证协议

### 7.1 对照与消融

1. PlainConvUNet + VoCo；
2. 当前 PFCR；
3. PFCR + 修正的二分类 prior 概率；
4. CU-PFR，仅用语义不确定性输入；
5. CU-PFR + 绝对门；
6. 完整 CU-PFR + 反事实风险损失；
7. 完整方案去掉 `N_hp` 项；
8. 完整方案去掉 GT 边界项。

第 7、8 项分别验证假阳性抑制和低对比边界改善是否确实来自所声称的机制。

### 7.2 数据与统计

- LiTS 五折交叉验证：病例级均值、标准差、95% paired bootstrap CI、Wilcoxon signed-rank 检验；
- 至少一个独立外部测试集，如 MSD Task03 Liver 或 3D-IRCADb，并明确标签映射；
- 所有实验固定 TTA、后处理、概率阈值；
- 不共享病人或同一扫描的重建版本；
- 发布逐病例指标和 manifest，而非只报总 Dice。

### 7.3 必须报告的结果

| 维度 | 指标 |
|---|---|
| 体素级 | Tumor Dice、IoU、HD95、ASSD、precision、recall。 |
| 体积稳定性 | 病例预测/GT 体积比及分位数。 |
| 病灶级 | `<1 mL`、`1-10 mL`、`>=10 mL` 的 lesion-wise recall 与每例 FP 病灶数。 |
| 安全性 | 无肿瘤病例的 FP volume、FP lesion count。 |
| 高频难阴性 | GT 肿瘤外 high-PC 区域的 tumor FP rate。 |
| 机制证据 | `a` 对真实边界和 non-tumor high-PC 的 AUPRC；`Delta_m` 在两类区域的分布与显著性。 |

### 7.4 可视化病例

对同一病例并列显示 CT、GT、当前 PFCR、CU-PFR、FP/FN、PC、语义熵、绝对门 `a`、三频带比例和 `Delta_m`。至少覆盖：`LiTS_005`、`LiTS_054`、`LiTS_092`、`LiTS_130` 及无肿瘤病例 `LiTS_032/106/115`。

## 8. 预注册式拒绝标准

出现以下任何情况，不能将 CU-PFR 视为成立：

1. Dice 上升但无肿瘤 FP 或 high-PC non-tumor FP rate 不下降；
2. `a` 对真实边界与难阴性没有可分性；
3. `Delta_m` 在 GT 边界不为正，或在难阴性仍显著为正；
4. 改进仅出现在单一 fold，跨 fold bootstrap CI 跨过零；
5. LiTS 改进不能迁移至外部数据；
6. 只有加回旧辅助损失才有效，说明 CU-PFR 没有独立可归因收益。

## 9. 创新性定位

CU-PFR 的创新不应表述为“使用 wavelet、phase congruency、不确定性或 attention”，这些组件都已有大量工作。可检验的创新主张是：

> 将频率增强建模为肿瘤-肝脏决策 margin 的局部可干预变量；通过同一控制器的反事实风险约束，使高频细节仅在提升真实肿瘤边界判别、且不提高高结构难阴性肿瘤分数时被激活。

这将频率模块从相关性特征增强转化为具有可验证决策收益的频带干预。该方向有形成 MICCAI/TMI/MedIA 级完整论文的潜力，但不能仅凭架构名称或单 fold LiTS 结果宣称顶会/顶刊水平。是否成立取决于上述消融、反事实证据、统计显著性与跨域泛化是否全部通过。
