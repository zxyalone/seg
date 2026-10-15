# 文献索引

共 134 篇，全部已下载：01–04 是 2026-10-09 整理的 33 篇；05–13 是 2026-10-15 补充的顶刊顶会论文，共 101 篇。各论文的划分、统计口径和结果对比见 [文献对比表.md](文献对比表.md)。命名规则：`年份_第一作者_简称_期刊.pdf`。"BSPC-arXiv"表示已在 BSPC 发表、此处存的是 arXiv 预印本。

优先级：**P0** 必读，会直接进对比表或正文讨论；**P1** 相关工作 / 方法依据；**P2** 补充引用。

## 01_BSPC肝肿瘤分割（目标期刊同类工作，13 篇）

| 优先级 | 文件 | 论文 | 出处 | 链接 | 与本文的关系 |
|---|---|---|---|---|---|
| P0 | `2025_Ma_MSFFE-Net_BSPC.pdf` | MSFFE-Net: Multi-scale Spatial-Frequency Feature Enhancement for accurate liver tumor segmentation | BSPC 113 (2025) 108963 | [doi](https://doi.org/10.1016/j.bspc.2025.108963) | **最直接的竞争工作**（空间-频率增强 + 肝肿瘤），必须讲清区别 |
| P0 | `2026_Bian_MambaClinix_BSPC-arXiv.pdf` | MambaClinix: Hierarchical Gated Convolution and Mamba-Based U-Net for 3D Medical Image Segmentation | BSPC 124 (2026) 110657 | [doi](https://doi.org/10.1016/j.bspc.2026.110657) · [arXiv](https://arxiv.org/abs/2409.12533) | **写作模板**：基于 nnU-Net 的 3D 方法，所有基线在同一框架下重训；LiverT 肿瘤 DSC：nnU-Net 66.57 / U-Mamba 70.67 / 本文 71.37 |
| P0 | `2024_Cheng_EG-UNETR_BSPC.pdf` | EG-UNETR: An edge-guided liver tumor segmentation network based on cross-level interactive transformer | BSPC 97 (2024) 106739 | [doi](https://doi.org/10.1016/j.bspc.2024.106739) | 边缘引导，对应我们的相位边界证据 |
| P0 | `2025_Liu_ObjectKnowledge-MIL-SmallTumor_BSPC.pdf` | Object knowledge-aware multiple instance learning for small tumor segmentation | BSPC 115 (2025) 109400 | [doi](https://doi.org/10.1016/j.bspc.2025.109400) | 小肿瘤分割，对应病灶大小分层 |
| P0 | `2025_Raghaw_T-MPEDNet_BSPC-arXiv.pdf` | T-MPEDNet: Transformer-aware Multiscale Progressive Encoder-Decoder Network ... for Tumor and Liver Segmentation | BSPC (2025) 108225 | [doi](https://doi.org/10.1016/j.bspc.2025.108225) · [arXiv](https://arxiv.org/abs/2507.19590) | 2D 切片、随机 80/10/10 划分，tumor Dice 89.1，**口径与我们不可比**；引言五段式结构可借鉴 |
| P1 | `2025_Wang_ODConv-FeatureFusion_BSPC.pdf` | A novel omni-dimensional dynamic convolution-based feature fusion framework for liver and tumor segmentation from CT | BSPC 112 (2025) 108572 | [doi](https://doi.org/10.1016/j.bspc.2025.108572) | 同期肝肿瘤方法 |
| P1 | `2025_Wu_TCA-Net_BSPC.pdf` | TCA-Net: Liver tumor segmentation based on a new contextual feature encoding module and attention mechanism | BSPC 112 (2025) 108518 | [doi](https://doi.org/10.1016/j.bspc.2025.108518) | LiTS 83.90 / 3DIRCADb 84.64，需核对统计口径 |
| P1 | `2024_Qi_AD-DUNet_BSPC.pdf` | AD-DUNet: dual-branch encoder combining axial Transformer with cascaded dilated convolutions for liver and hepatic tumor segmentation | BSPC 95 (2024) 106397 | [doi](https://doi.org/10.1016/j.bspc.2024.106397) | 4.24M 参数、无预训练，LiTS + 3DIRCAD |
| P2 | `2025_Zhu_BPP-Net_BSPC.pdf` | BPP-net: bio-inspired parallel pathway network for liver and tumor segmentation | BSPC 112 (2025) 108497 | [doi](https://doi.org/10.1016/j.bspc.2025.108497) | 轻量模型（0.14M 参数） |
| P2 | `2025_Zheng_ABeDice-Loss_BSPC.pdf` | Adaptive boundary-enhanced Dice loss for image segmentation | BSPC 106 (2025) 107741 | [doi](https://doi.org/10.1016/j.bspc.2025.107741) | 边界损失，放相关工作 |
| P2 | `2024_Hu_MSML-AttUNet_BSPC.pdf` | MSML-AttUNet: hierarchical attention network with multi-scale and multi-task for precision liver tumor segmentation | BSPC 99 (2024) 106861 | [doi](https://doi.org/10.1016/j.bspc.2024.106861) | 多尺度 + 多任务 |
| P2 | `2024_Peng_Spider-Net_BSPC.pdf` | Spider-Net: High-resolution multi-scale attention network ... for tumor segmentation in kidney, liver and pancreas | BSPC 93 (2024) 106163 | [doi](https://doi.org/10.1016/j.bspc.2024.106163) | 多器官肿瘤 |
| P2 | `2024_You_ContourGraphReasoning_BSPC.pdf` | Contour-induced parallel graph reasoning for liver tumor segmentation | BSPC 92 (2024) 106111 | [doi](https://doi.org/10.1016/j.bspc.2024.106111) | 轮廓引导 |

## 02_基线与预训练骨干（9 篇）

| 优先级 | 文件 | 论文 | 出处 | 链接 | 用途 |
|---|---|---|---|---|---|
| P0 | `2021_Isensee_nnU-Net_NatMethods-arXiv.pdf` | nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation（预印本题为 *Automated Design of Deep Learning Methods for Biomedical Image Segmentation*） | Nature Methods 18 (2021) | [doi](https://doi.org/10.1038/s41592-020-01008-z) · [arXiv](https://arxiv.org/abs/1904.08128) | A0 基线；**正文引用 Nature Methods 版** |
| P2 | `2018_Isensee_nnU-Net-original_arXiv.pdf` | nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation | arXiv 2018 | [arXiv](https://arxiv.org/abs/1809.10486) | 早期版本，一般不需引用 |
| P0 | `2024_Isensee_nnU-Net-Revisited_MICCAI.pdf` | nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation | MICCAI 2024 | [arXiv](https://arxiv.org/abs/2404.09556) | 公平对比的依据；ResEnc 强基线 |
| P0 | `2024_Wu_VoCo_CVPR.pdf` | VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis | CVPR 2024 | [arXiv](https://arxiv.org/abs/2402.17300) | 预训练方法 |
| P0 | `2024_Wu_LargeScale-VoCo_arXiv.pdf` | Large-Scale 3D Medical Image Pre-training with Geometric Context Priors | arXiv 2024 | [arXiv](https://arxiv.org/abs/2410.09890) | VoCo 大规模版；**VoComni 权重的引用出处待核对**（见 [代码仓库](https://github.com/Luffy03/Large-Scale-Medical)） |
| P1 | `2024_Ma_U-Mamba_arXiv.pdf` | U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation | arXiv 2024 | [arXiv](https://arxiv.org/abs/2401.04722) | 外部强基线候选 |
| P1 | `2023_Roy_MedNeXt_MICCAI.pdf` | MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation | MICCAI 2023 | [arXiv](https://arxiv.org/abs/2303.09975) | 外部强基线候选 |
| P1 | `2022_Hatamizadeh_SwinUNETR_BrainLes.pdf` | Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images | BrainLes 2021 (LNCS 2022) | [arXiv](https://arxiv.org/abs/2201.01266) | 常见对比方法 |
| P1 | `2018_Li_H-DenseUNet_TMI.pdf` | H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes | IEEE TMI 2018 | [arXiv](https://arxiv.org/abs/1709.07330) | LiTS 经典方法，引言必引 |

## 03_相位一致性与频域（6 篇）

| 优先级 | 文件 | 论文 | 出处 | 链接 | 用途 |
|---|---|---|---|---|---|
| P0 | `1999_Kovesi_PhaseCongruency_Videre.pdf` | Image Features from Phase Congruency | Videre 1(3), 1999 | [摘要页](https://www.cs.rochester.edu/~brown/Videre/001/abstracts/v1n3001.html)（PDF 取自 Internet Archive） | PC 定义与**噪声阈值 T 的出处**（Step 2 的核心机制问题） |
| P0 | `2001_Felsberg_MonogenicSignal_TSP.pdf` | The Monogenic Signal | IEEE TSP 49(12), 2001 | [doi](https://doi.org/10.1109/78.969520) | Riesz 变换 / 单演信号，PC 算子的理论基础 |
| P1 | `2011_Zhang_FSIM_TIP.pdf` | FSIM: A Feature SIMilarity Index for Image Quality Assessment（作者版） | IEEE TIP 20(8), 2011 | [doi](https://doi.org/10.1109/TIP.2011.2109730) | PC₁ 形式的出处；PC 与梯度互补的论述 |
| P1 | `2020_Li_WaveletIntegratedCNN_CVPR.pdf` | Wavelet Integrated CNNs for Noise-Robust Image Classification | CVPR 2020 | [arXiv](https://arxiv.org/abs/2005.03337) | 小波嵌入 CNN 的代表工作 |
| P1 | `2025_Jiang_TexLiverNet_arXiv.pdf` | TexLiverNet: Leveraging Medical Knowledge and Spatial-Frequency Perception for Enhanced Liver Tumor Segmentation | arXiv（发表出处待核对） | [arXiv](https://arxiv.org/abs/2411.04595) | 空间-频率感知的肝肿瘤分割 |
| P1 | `2025_Han_FrequencyDomain-Abdominal_arXiv.pdf` | Frequency Domain Unlocks New Perspectives for Abdominal Medical Image Segmentation | arXiv 2025 | [arXiv](https://arxiv.org/abs/2510.11005) | 频域腹部肿瘤分割 |

## 04_评测与数据集（5 篇）

| 优先级 | 文件 | 论文 | 出处 | 链接 | 用途 |
|---|---|---|---|---|---|
| P0 | `2023_Bilic_LiTS-Benchmark_MedIA.pdf` | The Liver Tumor Segmentation Benchmark (LiTS) | Medical Image Analysis 84 (2023) | [arXiv](https://arxiv.org/abs/1901.04056) | 数据集；官方挑战赛肿瘤 Dice 0.674 / 0.702 / 0.739 |
| P0 | `2024_Maier-Hein_MetricsReloaded_NatMethods.pdf` | Metrics Reloaded: Recommendations for image analysis validation | Nature Methods 2024 | [arXiv](https://arxiv.org/abs/2206.01653) | 病灶级指标与统计口径的依据 |
| P1 | `2024_Reinke_MetricPitfalls_NatMethods.pdf` | Understanding metric-related pitfalls in image analysis validation | Nature Methods 2024 | [arXiv](https://arxiv.org/abs/2302.01790) | 说明为什么不同口径的 Dice 不可比 |
| P1 | `2023_Kofler_BlobLoss_IPMI.pdf` | blob loss: instance imbalance aware loss functions for semantic segmentation | IPMI 2023 | [arXiv](https://arxiv.org/abs/2205.08209) | 实例级（病灶级）损失与评测 |
| P1 | `2019_Simpson_MSD_arXiv.pdf` | A large annotated medical image dataset for the development and evaluation of segmentation algorithms（Medical Segmentation Decathlon） | arXiv 2019 | [arXiv](https://arxiv.org/abs/1902.09063) | 说明 LiTS 与 MSD Task03 是同一批数据 |

## 外部测试候选数据集（非论文）

- [TCIA Colorectal-Liver-Metastases](https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/)
- [TCIA HCC-TACE-Seg](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/)

## 待办

- [x] 从 BSPC 论文中提取数据划分、统计口径和结果，整理成[文献对比表](文献对比表.md)
- [ ] `reproduce_metrics.py` 增加全局 Dice（见对比表发现 1）
- [ ] 核对 VoComni 权重的正式引用出处
- [ ] 核对 TexLiverNet 的正式发表出处

---

# 顶刊顶会补充（2026-10-15）

按肝肿瘤分割的难点分 9 类，共 101 篇，全部已下载（其中 13 篇付费墙论文由手动下载）。由 5 个检索子代理分方向收集，出处已用 Crossref / arXiv / CVF / MICCAI 官方页核对，所有 PDF 均抽取首页文字核对过标题。"出处待核"表示子代理未能完全核实。

**最值得先读的几篇**：E²Net（MICCAI 2020，肝肿瘤边缘增强，与相位边界证据最接近）、WaveFormer / XNet / FreqDomain-BrainTumor（小波与频带分解的直接竞争工作）、FACT 与 Amplitude-Phase Recombination（相位携带结构的依据）、Label-Free Liver Tumor（合成小肿瘤）、Loss Odyssey（LiTS 上的损失基准）、CC-Metrics（病灶级评测）、Maier-Hein 2018（排名稳定性与统计报告）。

**检索缺口**（子代理没找到，不是遗漏）：顶刊顶会中没有"可学习的相位一致性 / 单演信号层"，这可以作为 PhASER-3D 的新颖性论据；TMI / MedIA / MICCAI 中专门针对肝肿瘤边界的网络很少，只有 E²Net；STU-Net、SAM-Med3D、nnInteractive、MedFormer 只有 arXiv 版，没有收录。

## 05_肝肿瘤专题（15 篇）

直接做肝脏 / 肝肿瘤分割或肝病灶诊断的顶刊顶会工作，含多期相融合、边缘增强、弱监督，以及 TCIA 外部测试集的数据描述论文（Scientific Data，作数据集引用）。

| 优先级 | 文件 | 论文 | 出处 | 链接 | 解决的难点 | 对本课题的用处 |
|---|---|---|---|---|---|---|
| P0 | `2024_Simpson_CRLM-Dataset_SciData.pdf` | Preoperative CT and survival data for patients undergoing resection of colorectal liver metastases (Colorectal-Liver-Metastases) | Scientific Data 11 (2024) | [doi](https://doi.org/10.1038/s41597-024-02981-2) | 外部测试数据：TCIA上197例结直肠肝转移门脉期CT，含肝/血管/肿瘤分割 | 可作为TCIA外部测试集的数据引用（转移瘤，与LiTS中原发/继发混合分布不同，可检验泛化）；非顶刊，仅作数据集引用 |
| P0 | `2023_Moawad_HCC-TACE-Seg-Dataset_SciData.pdf` | Multimodality annotated hepatocellular carcinoma data set including pre- and post-TACE with imaging segmentation (HCC-TACE-Seg) | Scientific Data 10:33 | [doi](https://doi.org/10.1038/s41597-023-01928-3) | 外部测试数据：TCIA上105例HCC(TACE前后CT)含肿瘤分割 | 可作为TCIA外部测试集的数据引用(非顶刊，仅作数据集引用)；多期相、单中心(MD Anderson) |
| P0 | `2021_Zhang_MultiPhase-UncertainInpainting_MICCAI.pdf` | Multi-phase Liver Tumor Segmentation with Spatial Aggregation and Uncertain Region Inpainting | MICCAI 2021 (LNCS 12901, pp. 68-77) | [doi](https://doi.org/10.1007/978-3-030-87193-2_7) · [arXiv](https://arxiv.org/abs/2108.00911) | 多期相CT(动脉/门脉等)像素级跨期相融合；肿瘤边界不确定区域的细化 | 多期相融合的代表方法，并显式处理边界低置信区域；用于相关工作中说明多期相信息与边界不确定性问题，也可论证本方法在单期相(LiTS/MSD)下的定位 |
| P0 | `2020_Tang_E2Net-EdgeEnhanced_MICCAI.pdf` | E²Net: An Edge Enhanced Network for Accurate Liver and Tumor Segmentation on CT Scans | MICCAI 2020 (LNCS 12264, pp. 512-522) | [doi](https://doi.org/10.1007/978-3-030-59719-1_50) · [arXiv](https://arxiv.org/abs/2007.09791) | 肝/肿瘤边界模糊；显式建模肝与肿瘤边缘并用边缘距离图监督，小标注集下更易训练 | 与PhASER-3D的“边界证据”思路最接近的肝肿瘤专用工作：以边缘预测分支+边缘监督提升边界；是论证相位一致性(PC)作为无监督边界证据优于可学习边缘头的直接对比/相关工作 |
| P1 | `2023_Yan_PLAN-PixelLesionPatient_MICCAI.pdf` | Liver Tumor Screening and Diagnosis in CT with Pixel-Lesion-Patient Network | MICCAI 2023 (LNCS 14221, pp. 72-82) | [doi](https://doi.org/10.1007/978-3-031-43904-9_8) · [arXiv](https://arxiv.org/abs/2307.08268) | 肝肿瘤筛查与多类型诊断；病灶级而非仅像素级的评测（像素-病灶-患者三层） | 提出应同时汇报病灶级检出指标，支撑本文在逐病例Dice之外补充病灶级召回/F1的评测口径；也是多发/小病灶检出的参考 |
| P1 | `2023_Xiao_EdgeAware-MultiTask_MICCAI.pdf` | Edge-Aware Multi-task Network for Integrating Quantification Segmentation and Uncertainty Prediction of Liver Tumor on Multi-modality Non-contrast MRI | MICCAI 2023 (LNCS 14223, pp. 652-661) | [doi](https://doi.org/10.1007/978-3-031-43901-8_62) | 肝肿瘤边界模糊；边缘感知多任务+不确定性预测(非增强MRI) | 又一个以边缘感知处理肝肿瘤边界模糊的MICCAI工作，可与E2Net并列作为边缘/边界相关工作 |
| P1 | `2022_Lyu_SyntheticCT-TTT-LiverTumor_TMI.pdf` | Learning From Synthetic CT Images via Test-Time Training for Liver Tumor Segmentation | IEEE TMI 41(9) | [doi](https://doi.org/10.1109/TMI.2022.3166230) | 肝肿瘤标注稀缺；合成到真实的域差异 | 向健康CT注入合成肿瘤并通过测试时训练(SR-TTT)缓解合成-真实域偏移，在LiTS等公开集验证；属于TMI上的肝肿瘤合成增强工作，可作相关工作。 |
| P1 | `2022_Lyu_Couinaud-WeakSup_TMI.pdf` | Weakly Supervised Liver Tumor Segmentation Using Couinaud Segment Annotation | IEEE TMI 41(5):1138-1149 | [doi](https://doi.org/10.1109/TMI.2021.3132905) | 肿瘤体素标注昂贵；用Couinaud肝段级弱标注训练肿瘤分割（LiTS） | 弱监督肝肿瘤分割的TMI代表作，可放在标注稀缺/弱监督相关工作中，并说明本文使用全监督设定的比较范围 |
| P1 | `2020_Seo_mU-Net_TMI.pdf` | Modified U-Net (mU-Net) With Incorporation of Object-Dependent High Level Features for Improved Liver and Liver-Tumor Segmentation in CT Images | IEEE TMI 39(5):1316-1325 | [doi](https://doi.org/10.1109/TMI.2019.2948320) | 肝及肝肿瘤分割中细节/边界丢失；在跳跃连接中加入对象相关高层特征(LiTS) | LiTS上的TMI肝肿瘤U-Net改进基线之一，可作为2D/2.5D类对比与相关工作，说明跳连特征重标定的既有思路 |
| P2 | `2024_Ying_LiAIDS-FocalLiverLesion_NatComm.pdf` | A multicenter clinical AI system study for detection and diagnosis of focal liver lesions (LiAIDS) | Nature Communications 15, 1131 | [doi](https://doi.org/10.1038/s41467-024-45325-9) | 肝局灶性病灶的检出与诊断；多中心泛化 | 多中心、多期相CT的肝局灶病灶检测+诊断临床系统，可作为肝病灶检出口径与多中心验证的相关工作。 |
| P2 | `2024_Huang_LIDIA-MultiPhase_MICCAI.pdf` | LIDIA: Precise Liver Tumor Diagnosis on Multi-Phase Contrast-Enhanced CT via Iterative Fusion and Asymmetric Contrastive Learning | MICCAI 2024 (LNCS) | [doi](https://doi.org/10.1007/978-3-031-72114-4_38) · [arXiv](https://arxiv.org/abs/2407.13217) | 多期相CT中小/异质肝肿瘤的分割与诊断 | 基于Mask2Former式局部注意力处理小肝病灶，使用1921例/8138病灶的多期相数据集；可作为多期相与小病灶检测诊断的相关工作(与多期相方向重叠)。 |
| P2 | `2024_Wei_LiLNet_NatComm.pdf` | Focal liver lesion diagnosis with deep learning and multistage CT imaging (LiLNet) | Nature Communications 15:7040 | [doi](https://doi.org/10.1038/s41467-024-51260-6) | 多中心多期相CT肝局灶性病变检测(YOLOv8)与6类分类；4039例六中心 | 顶刊大规模多中心多期相肝病灶AI研究，用于引言中说明肝病灶AI的临床需求与多期相/多中心设定 |
| P2 | `2024_Lyu_Superpixel-SAM-LiverTumor_MICCAI.pdf` | Superpixel-Guided Segment Anything Model for Liver Tumor Segmentation with Couinaud Segment Prompt | MICCAI 2024 (LNCS 15009, pp. 678-688) | [doi](https://doi.org/10.1007/978-3-031-72111-3_64) | SAM用于肝肿瘤分割；以超像素与Couinaud段提示 | SAM/基础模型路线下的肝肿瘤分割对比参考 |
| P2 | `2023_Konkel_Generalizability-LiverSeg_RadiolAI.pdf` | Systematic Analysis of Common Factors Impacting Deep Learning Model Generalizability in Liver Segmentation | Radiology: Artificial Intelligence 5(3):e220080 | [doi](https://doi.org/10.1148/ryai.220080) | 肝分割跨域偏移：软组织对比度差异是主因，多源数据混合训练大幅提升泛化 | 支撑外部测试(TCIA)中对比度/期相差异导致域偏移的论述 |
| P2 | `2019_Vorontsov_LiverLesion-CRLM_RadiolAI.pdf` | Deep Learning for Automated Segmentation of Liver Lesions at CT in Patients with Colorectal Cancer Liver Metastases | Radiology: Artificial Intelligence 1(2):180014 | [doi](https://doi.org/10.1148/ryai.2019180014) | 结直肠肝转移小病灶漏检：<10mm病灶全自动敏感度仅0.10 | Radiology: AI上的肝病灶FCN分割经典临床评估，给出按病灶大小分层的敏感度，可支撑小病灶分层评测 |

## 06_小病灶与病灶检测（15 篇）

小病灶漏检、前景不平衡、病灶级损失与评测、通用病灶检测。

| 优先级 | 文件 | 论文 | 出处 | 链接 | 解决的难点 | 对本课题的用处 |
|---|---|---|---|---|---|---|
| P0 | `2025_Jaus_CC-Metrics_AAAI.pdf` | Every Component Counts: Rethinking the Measure of Success for Medical Semantic Segmentation in Multi-Instance Segmentation Tasks (CC-Metrics) | AAAI 2025 (Vol.39 No.4, pp.3904-3912) | [doi](https://doi.org/10.1609/aaai.v39i4.32408) · [arXiv](https://arxiv.org/abs/2410.18684) | 多发病灶评测被大病灶主导；小病灶漏检在Dice等体素指标中被掩盖 | 提出连通域(病灶)级评测CC-Metrics，让每个病灶权重相同；可作为在逐病例Dice之外补充病灶级(lesion-wise)指标的评测依据，直接回应评测口径不一与小病灶漏检问题。 |
| P0 | `2021_Ma_LossOdyssey_MedIA.pdf` | Loss odyssey in medical image segmentation | Medical Image Analysis 71, 102035 | [doi](https://doi.org/10.1016/j.media.2021.102035) · [arXiv](https://arxiv.org/abs/2005.13449) | 前景极度不平衡下的损失函数选择；缺乏系统基准 | 首个对20种分割损失在含肝肿瘤(LiTS)等4个3D任务上的大规模基准，结论是复合损失(Dice+CE/TopK/Focal等)最稳健；为本方法沿用nnU-Net的Dice+CE损失及损失设计提供依据。 |
| P1 | `2025_Wu_MSWAL_MICCAI.pdf` | MSWAL: 3D Multi-class Segmentation of Whole Abdominal Lesions Dataset | MICCAI 2025 (LNCS, pp. 378-388) | [doi](https://doi.org/10.1007/978-3-032-04937-7_36) · [arXiv](https://arxiv.org/abs/2503.13560) | 腹部多类病灶(含肝肿瘤/肝囊肿)数据集；跨数据集迁移到LiTS提升肝肿瘤Dice 3.00% | 提供新的多期相肝病灶数据与基于nnU-Net的Inception nnU-Net基线，可作为nnU-Net系基线和额外外部/预训练数据来源 |
| P1 | `2025_Grauw_ULS23_MedIA.pdf` | The ULS23 challenge: A baseline model and benchmark dataset for 3D universal lesion segmentation in computed tomography | Medical Image Analysis 102:103525 | [doi](https://doi.org/10.1016/j.media.2025.103525) · [arXiv](https://arxiv.org/abs/2406.05231) | 通用3D病灶分割基准；训练集含大量肝等腹部病灶，基线为nnU-Net ResEnc | 作为nnU-Net系病灶分割基线与评测基准的最新MedIA依据，可对照其病灶级评估设置 |
| P1 | `2021_Wang_PairwiseLearning_MedIA.pdf` | Pairwise learning for medical image segmentation | Medical Image Analysis 67:101876 | [doi](https://doi.org/10.1016/j.media.2020.101876) | 小样本下病灶分割过拟合；成对样本联合分割+代理监督(LiTS/CHAOS) | MedIA上基于LiTS的肝肿瘤分割工作，可作为小训练集/数据高效学习的相关工作 |
| P1 | `2020_Shirokikh_LossReweighting-LesionSize_MICCAI.pdf` | Universal Loss Reweighting to Balance Lesion Size Inequality in 3D Medical Image Segmentation | MICCAI 2020 (LNCS 12264, pp. 523-532) | [doi](https://doi.org/10.1007/978-3-030-59719-1_51) · [arXiv](https://arxiv.org/abs/2007.10033) | 多发病灶中大病灶淹没小病灶(病灶尺寸不平衡)；按病灶体积反比加权体素 | 与小病灶漏检/前景极度不平衡直接相关的损失设计，可作为损失函数对比或消融依据 |
| P2 | `2023_Chen_AdaptiveRegionLoss_TPAMI.pdf` | Adaptive Region-Specific Loss for Improved Medical Image Segmentation | IEEE TPAMI 45(11), 13408-13421 | [doi](https://doi.org/10.1109/TPAMI.2023.3289667) | 难分割/小子区域被整体损失淹没；类别/区域不平衡 | 在训练中按区域假阳/假阴误差自适应调整区域损失权重，无需改网络；可作为针对小病灶/难区域的自适应损失的对比思路。 |
| P2 | `2023_Wang_DiceSemimetricLoss_MICCAI.pdf` | Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels | MICCAI 2023 (LNCS) | [doi](https://doi.org/10.1007/978-3-031-43898-1_46) · [arXiv](https://arxiv.org/abs/2303.16296) | 软标签下soft Dice损失不适用；标注不确定性与模型校准 | 在LiTS/KiTS上验证的Dice损失软标签扩展，在硬标签下与soft Dice等价；若本方法使用深监督/平滑标签或需讨论Dice损失与校准，可作为依据。 |
| P2 | `2023_Cao_PANDA-Pancreas_NatMed.pdf` | Large-scale pancreatic cancer detection via non-contrast CT and deep learning (PANDA) | Nature Medicine 29(12), 3033-3043 | [doi](https://doi.org/10.1038/s41591-023-02640-w) | 小而低对比度病灶的检出；病灶级检出与分割结合 | 分割+检测联合的多中心大规模验证，展示低对比度小病灶的病灶级检出敏感度/特异度评估方式，可作为临床导向的病灶级评测参考。 |
| P2 | `2021_Baumgartner_nnDetection_MICCAI.pdf` | nnDetection: A Self-configuring Method for Medical Object Detection | MICCAI 2021 (LNCS 12905, pp.530-539) | [doi](https://doi.org/10.1007/978-3-030-87240-3_51) · [arXiv](https://arxiv.org/abs/2106.00817) | 病灶检测与分割分离；小病灶需要检测式(实例级)优化 | nnU-Net作者团队的自配置3D病灶检测框架，可作为『检测+分割联合/病灶级』路线的参照，说明本方法为何仍采用体素分割范式并补充病灶级评测。 |
| P2 | `2021_Heller_KiTS19_MedIA.pdf` | The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 challenge | Medical Image Analysis 67, 101821 | [doi](https://doi.org/10.1016/j.media.2020.101821) · [arXiv](https://arxiv.org/abs/1912.01054) | 肾肿瘤分割基准；肿瘤Dice显著低于器官且低于人类一致性 | 与LiTS类似的肿瘤分割挑战总结，显示肿瘤分割相对器官分割的差距并采用Dice排名；可作为肿瘤分割难点与评测口径的类比参考。 |
| P2 | `2021_Yan_UniversalLesionDetection_TMI.pdf` | Learning From Multiple Datasets With Heterogeneous and Partial Labels for Universal Lesion Detection in CT | IEEE TMI 40(10):2759-2770 | [doi](https://doi.org/10.1109/TMI.2020.3047598) · [arXiv](https://arxiv.org/abs/2009.02577) | DeepLesion漏标与LiTS等数据集标注范围异构；联合多数据集训练 | 同时使用LiTS肝肿瘤数据解决部分标注问题，可作为多数据集/部分标注学习相关工作 |
| P2 | `2021_Cai_LesionHarvester_TMI.pdf` | Lesion Harvester: Iteratively Mining Unlabeled Lesions and Hard-Negative Examples at Scale | IEEE TMI 40(1):59-70 | [doi](https://doi.org/10.1109/TMI.2020.3022034) · [arXiv](https://arxiv.org/abs/2001.07776) | 大规模CT档案中漏标病灶与难负样本挖掘 | 病灶挖掘与难负样本抑制思路，可用于讨论小病灶漏检与假阳性 |
| P2 | `2018_Yan_3DCE_MICCAI.pdf` | 3D Context Enhanced Region-based Convolutional Neural Network for End-to-End Lesion Detection | MICCAI 2018 (LNCS 11070, pp. 511-519) | [doi](https://doi.org/10.1007/978-3-030-00928-1_58) · [arXiv](https://arxiv.org/abs/1806.09648) | 通用病灶检测中的3D上下文：聚合相邻切片特征 | 3D上下文建模在病灶检测中的奠基工作(DeepLesion)，用于说明3D/2.5D上下文的必要性 |
| P2 | `2017_Lin_FocalLoss_ICCV.pdf` | Focal Loss for Dense Object Detection | ICCV 2017 (pp.2980-2988); 扩展版 IEEE TPAMI 42(2), 2020 | [doi](https://doi.org/10.1109/TPAMI.2018.2858826) · [arXiv](https://arxiv.org/abs/1708.02002) | 前景/背景极度不平衡；难样本被大量易样本淹没 | Focal loss奠基工作，是医学分割中处理极度不平衡与难样本的基础损失，可作为损失对比的方法依据。 |

## 07_肿瘤合成与标注稀缺（7 篇）

用合成肿瘤缓解小肿瘤样本稀缺，以及半监督。

| 优先级 | 文件 | 论文 | 出处 | 链接 | 解决的难点 | 对本课题的用处 |
|---|---|---|---|---|---|---|
| P0 | `2023_Hu_LabelFree-LiverTumor_CVPR.pdf` | Label-Free Liver Tumor Segmentation | CVPR 2023 | [doi](https://doi.org/10.1109/CVPR52729.2023.00717) · [arXiv](https://arxiv.org/abs/2303.14869) | 肝肿瘤标注稀缺、小/微小肿瘤样本不足导致漏检 | 提出向健康肝脏CT中合成含小/微小肿瘤的数据，仅用合成肿瘤训练即可达到接近真实标注的分割效果；可作为小肿瘤数据增强与『合成数据缓解小病灶稀缺』的相关工作，并可在讨论中说明PhASER-3D未使用合成数据。 |
| P1 | `2025_Wu_FreeTumor_NatComm.pdf` | Large-scale generative tumor synthesis in computed tomography images for improving tumor recognition (FreeTumor) | Nature Communications 16, 11053 | [doi](https://doi.org/10.1038/s41467-025-66071-6) · [arXiv](https://arxiv.org/abs/2502.18519) | 标注肿瘤数据稀缺(161k CT中仅2.3%有肿瘤标注)；早期小肿瘤检出敏感度低 | 利用大规模无标注CT做对抗式肿瘤合成并过滤低质量合成样本，Dice平均提升约6.7%、早期肿瘤敏感度提升16.4%；说明数据层面缓解小肿瘤稀缺的上限，可作为相关工作。 |
| P1 | `2025_Chen_ScalingTumorSeg_ICCV.pdf` | Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data | ICCV 2025 (pp.24001-24013) | [arXiv](https://arxiv.org/abs/2510.14831) | 真实肿瘤标注规模与小肿瘤数据受限；OOD泛化 | 提出AbdomenAtlas 2.0(含肝等六器官肿瘤)并系统研究真实+合成数据的scaling规律，合成数据使所需真实扫描量大幅下降，ID/OOD Dice分别提升约7%/16%；可作数据规模与泛化讨论依据。 |
| P1 | `2024_Chen_DiffTumor_CVPR.pdf` | Towards Generalizable Tumor Synthesis | CVPR 2024 | [doi](https://doi.org/10.1109/CVPR52733.2024.01060) · [arXiv](https://arxiv.org/abs/2402.19470) | 早期小肿瘤(<2cm)训练样本稀缺；跨器官/跨中心泛化 | DiffTumor：基于扩散模型合成肝/胰/肾小肿瘤，提升早期小肿瘤检出敏感度并改善跨医院泛化；是小肿瘤合成增强的代表作，可作为相关工作与外部测试(TCIA)泛化讨论的依据。 |
| P2 | `2024_Lai_Pixel2Cancer_MICCAI.pdf` | From Pixel to Cancer: Cellular Automata in Computed Tomography | MICCAI 2024 (LNCS 15001, pp.36-46) | [doi](https://doi.org/10.1007/978-3-031-72378-0_4) · [arXiv](https://arxiv.org/abs/2403.06459) | 小肿瘤数据稀缺；跨器官肿瘤合成 | 用元胞自动机按生长/侵袭/死亡规则合成多阶段肝/胰/肾肿瘤，仅用无标注CT训练即超过先前合成方法；补充肿瘤合成增强一脉的MICCAI代表作。 |
| P2 | `2023_Bai_BCP-SemiSup_CVPR.pdf` | Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation | CVPR 2023 | [doi](https://doi.org/10.1109/CVPR52729.2023.01108) · [arXiv](https://arxiv.org/abs/2305.00673) | 半监督医学分割:双向copy-paste缓解标注/无标注分布不一致 | 半监督方法代表(验证集为心脏/胰腺,非肝),作标注稀缺相关工作 |
| P2 | `2019_Yu_UA-MT-SemiSup_MICCAI.pdf` | Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation | MICCAI 2019 | [doi](https://doi.org/10.1007/978-3-030-32245-8_67) · [arXiv](https://arxiv.org/abs/1907.07034) | 3D半监督分割:不确定性加权的mean-teacher一致性 | 半监督3D分割奠基经典(UA-MT),作相关工作 |

## 08_边界与形状（7 篇）

低对比度与边界模糊：边界损失、距离图、边界感知网络与边界评测。

| 优先级 | 文件 | 论文 | 出处 | 链接 | 解决的难点 | 对本课题的用处 |
|---|---|---|---|---|---|---|
| P0 | `2021_Kervadec_BoundaryLoss_MedIA.pdf` | Boundary loss for highly unbalanced segmentation | Medical Image Analysis 67, 101851 | [doi](https://doi.org/10.1016/j.media.2020.101851) · [arXiv](https://arxiv.org/abs/1812.07032) | 前景极度不平衡；区域损失对小目标梯度失衡 | 提出基于轮廓距离的边界损失以应对极度不平衡，与本课题用相位一致性提供『边界证据』的思路相互呼应，可作为边界/距离类损失的对比或相关工作。 |
| P1 | `2022_Wang_BA-Net_MedIA.pdf` | Boundary-aware Context Neural Network for Medical Image Segmentation (BA-Net) | Medical Image Analysis 78: 102395 | [doi](https://doi.org/10.1016/j.media.2022.102395) · [arXiv](https://arxiv.org/abs/2005.00966) | 边界模糊：金字塔边缘提取+分割/边界多任务+交叉特征融合 | MedIA中‘边缘/边界引导医学分割网络’的代表作(2D)，作为边界先验类相关工作 |
| P1 | `2020_Karimi_HausdorffLoss_TMI.pdf` | Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks | IEEE TMI 39(2): 499-513 | [doi](https://doi.org/10.1109/TMI.2019.2930068) · [arXiv](https://arxiv.org/abs/1904.10030) | 边界误差(HD)大、离群误分：三种可微HD近似损失(距离变换/腐蚀/卷积核)，含肝脏CT | 直接优化HD的TMI代表作，可作为边界相关损失基线，同时为HD/HD95评测口径提供依据 |
| P2 | `2023_Sun_BoundaryDoULoss_MICCAI.pdf` | Boundary Difference Over Union Loss For Medical Image Segmentation | MICCAI 2023 (LNCS, pp. 292-301) | [doi](https://doi.org/10.1007/978-3-031-43901-8_28) · [arXiv](https://arxiv.org/abs/2308.00220) | 边界区域精度：基于区域计算的边界差异IoU损失，无需额外损失 | 轻量边界损失，可作为nnU-Net中替换/叠加损失的对比项 |
| P2 | `2021_Cheng_BoundaryIoU_CVPR.pdf` | Boundary IoU: Improving Object-Centric Image Segmentation Evaluation | CVPR 2021 (pp. 15329-15337) | [doi](https://doi.org/10.1109/CVPR46437.2021.01508) · [arXiv](https://arxiv.org/abs/2103.16562) | 评测口径：对边界误差更敏感、对大目标不被区域重叠淹没的指标 | 若需补充边界质量指标(除Dice/HD95)，可作为评测依据 |
| P2 | `2020_Xue_SDM-ShapeAware_AAAI.pdf` | Shape-Aware Organ Segmentation by Predicting Signed Distance Maps | AAAI 2020, 34(07): 12565-12572 | [doi](https://doi.org/10.1609/aaai.v34i07.6946) · [arXiv](https://arxiv.org/abs/1912.03849) | 形状先验/边界平滑：3D网络回归有符号距离图并与分割图联合训练，HD与ASD更低 | 距离图回归作形状/边界先验的3D实例，可作为边界证据的另一种表示的对比 |
| P2 | `2020_Kirillov_PointRend_CVPR.pdf` | PointRend: Image Segmentation as Rendering | CVPR 2020 (pp. 9796-9805) | [doi](https://doi.org/10.1109/CVPR42600.2020.00982) · [arXiv](https://arxiv.org/abs/1912.08193) | 边界过平滑/不确定区域：仅在高不确定点上细化预测 | 基于不确定性点采样的边界细化，作为‘不确定边界区域精修’路线的代表 |

## 09_频域小波与相位_顶会（10 篇）

与 PhASER-3D 的小波频带路由最相关的顶会顶刊工作，需在相关工作里讲清区别（尤其 WaveFormer、XNet、FreqDomain-BrainTumor）。

| 优先级 | 文件 | 论文 | 出处 | 链接 | 解决的难点 | 对本课题的用处 |
|---|---|---|---|---|---|---|
| P0 | `2025_Shao_FreqDomain-BrainTumor_TMI.pdf` | Rethinking Brain Tumor Segmentation from the Frequency Domain Perspective | IEEE TMI 44(11): 4536-4553 | [doi](https://doi.org/10.1109/TMI.2025.3579213) · [arXiv](https://arxiv.org/abs/2506.10142) | 肿瘤边界/细节(高频)易被丢失，对比增强区域分割难：频域分解+自适应拉普拉斯卷积强调高频边界+频域交叉注意力 | TMI顶刊中‘肿瘤分割+频域分解突出边界’的直接先例（HFF-Net），可作为频带分解增强边界的相关工作与对比思路；是否为3D需阅读全文确认 |
| P0 | `2025_Hasan_WaveFormer_MICCAI.pdf` | WaveFormer: A 3D Transformer with Wavelet-Driven Feature Representation for Efficient Medical Image Segmentation | MICCAI 2025 (LNCS 15963, pp. 684-694) | [doi](https://doi.org/10.1007/978-3-032-04965-0_64) · [arXiv](https://arxiv.org/abs/2503.23764) | 3D上下文与高频细节的权衡：多尺度3D离散小波分解，自注意力主要作用于低频，高频保留局部细节，小波重建替代重上采样 | 3D医学分割中使用DWT分频带的最新MICCAI工作（BraTS/FLARE/KiTS），与PhASER-3D的3D小波频带处理最接近，应作为必比较/必引的相关工作 |
| P0 | `2023_Zhou_XNet_ICCV.pdf` | XNet: Wavelet-Based Low and High Frequency Fusion Networks for Fully- and Semi-Supervised Semantic Segmentation of Biomedical Images | ICCV 2023 | [doi](https://doi.org/10.1109/ICCV51070.2023.01928) | 边界模糊/低对比：用小波把图像分成低频(整体形态)与高频(边缘纹理)两路再融合，在2D与3D生物医学数据上验证 | 小波低/高频双分支融合用于医学分割的ICCV代表作，是PhASER-3D“小波频带路由”最直接的对比与方法依据（需在相关工作中区分：XNet是输入级分解，本方法是特征级路由+相位一致性） |
| P1 | `2025_Zhou_GobletNet_TMI.pdf` | GobletNet: Wavelet-Based High-Frequency Fusion Network for Semantic Segmentation of Electron Microscopy Images | IEEE TMI 44(2): 1058-1069 | [doi](https://doi.org/10.1109/TMI.2024.3474028) | 边界细节/低对比：把小波高频成分单独融合以保持边缘 | TMI上将小波高频融合用于边界保持的工作（电镜图像，非CT），可作为‘高频带携带边界证据’的佐证 |
| P1 | `2024_Chang_EM-Net_MICCAI.pdf` | EM-Net: Efficient Channel and Frequency Learning with Mamba for 3D Medical Image Segmentation | MICCAI 2024 (LNCS, pp. 266-275) | [doi](https://doi.org/10.1007/978-3-031-72114-4_26) · [arXiv](https://arxiv.org/abs/2409.17675) | 3D长程依赖与效率：Mamba + 通道/频域学习 | 3D医学分割中引入频域学习的MICCAI工作，可作为频域+3D+nnU-Net类基线或相关工作 |
| P1 | `2024_Chen_FreqFusion_TPAMI.pdf` | Frequency-aware Feature Fusion for Dense Image Prediction (FreqFusion) | IEEE TPAMI 46(12) | [doi](https://doi.org/10.1109/TPAMI.2024.3449959) · [arXiv](https://arxiv.org/abs/2408.12879) | 边界位移/类内不一致：自适应低通滤波使类内特征一致，自适应高通滤波恢复被下采样模糊的边界 | 频率感知的特征融合(skip/上采样)的TPAMI代表作，可作为‘跳连中按频带处理’的方法依据；亦可作为2D对比思路 |
| P1 | `2024_Finder_WTConv_ECCV.pdf` | Wavelet Convolutions for Large Receptive Fields (WTConv) | ECCV 2024 (LNCS, pp. 363-380) | [doi](https://doi.org/10.1007/978-3-031-72949-2_21) · [arXiv](https://arxiv.org/abs/2407.05848) | 大感受野与形状偏置：小波域做多级卷积，参数随核增大仅对数增长 | 小波分解实现大感受野的即插即用层，可与nnU-Net/MedNeXt式大核设计对比，支撑‘小波路由’的效率与长程依赖论据 |
| P2 | `2025_Chen_SpatialFreqModulation_TPAMI.pdf` | Spatial Frequency Modulation for Semantic Segmentation | IEEE TPAMI 47(11): 9767-9784 | [doi](https://doi.org/10.1109/TPAMI.2025.3592621) · [arXiv](https://arxiv.org/abs/2507.11893) | 高频细节在下采样中混叠/丢失，影响小目标与边界分割 | FreqFusion同组的后续工作，从下采样角度保护高频，可作为‘下采样阶段频带保持’的依据（与小病灶漏检相关） |
| P2 | `2023_Hanif_FreqDomainAdvTraining_MICCAI.pdf` | Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation | MICCAI 2023 (LNCS, pp. 457-467) | [doi](https://doi.org/10.1007/978-3-031-43895-0_43) · [arXiv](https://arxiv.org/abs/2307.07269) | 3D医学分割的鲁棒性：在3D频域上做对抗训练 | 3D体数据频域处理的MICCAI工作，可用于说明3D FFT/频域扰动对鲁棒性/泛化的影响 |
| P2 | `2021_Rao_GFNet_NeurIPS.pdf` | Global Filter Networks for Image Classification | NeurIPS 2021 | [arXiv](https://arxiv.org/abs/2107.00645) | 3D/高分辨率下注意力二次复杂度：用FFT+可学习全局频域滤波做token混合 | 频域全局滤波的代表性方法，可作为‘频域建模长程依赖’的方法依据 |

## 10_频谱域泛化与跨中心（9 篇）

幅度-相位分解与跨域泛化：支撑"相位携带结构、对强度/对比度变化稳健"，也支撑 TCIA 外部测试的动机。

| 优先级 | 文件 | 论文 | 出处 | 链接 | 解决的难点 | 对本课题的用处 |
|---|---|---|---|---|---|---|
| P0 | `2021_Xu_FACT_CVPR.pdf` | A Fourier-based Framework for Domain Generalization (FACT) | CVPR 2021 | [doi](https://doi.org/10.1109/CVPR46437.2021.01415) · [arXiv](https://arxiv.org/abs/2105.11120) | 跨域偏移：相位携带高层语义且对域偏移更稳定，幅度谱混合作为增广 | ‘相位=结构/幅度=风格’这一假设的直接出处，是PhASER-3D用相位作边界证据、外部TCIA泛化的理论依据 |
| P0 | `2021_Chen_AmplitudePhaseRecombination_ICCV.pdf` | Amplitude-Phase Recombination: Rethinking Robustness of Convolutional Neural Networks in Frequency Domain | ICCV 2021 (pp. 458-467) | [doi](https://doi.org/10.1109/ICCV48922.2021.00051) · [arXiv](https://arxiv.org/abs/2108.08487) | 鲁棒性/域偏移：CNN过度依赖幅度谱；以相位谱+干扰图幅度谱重组迫使网络利用相位中的结构信息 | 为‘相位谱对强度/对比度变化更稳健’提供CNN层面的实证依据，支持以相位一致性(强度不变)作边界证据 |
| P1 | `2026_Sharshar_PhaseAT_MICCAI.pdf` | PhaseAT: Fourier Phase Adversarial Training for Medical Image Domain Generalization | MICCAI 2026 (LNCS vol. 16881, pp. 413-423) | [doi](https://doi.org/10.1007/978-3-032-38072-2_40) · [arXiv](https://arxiv.org/abs/2610.01807) | 医学图像域泛化：对傅里叶相位做对抗训练 | 目前最新的‘相位作为医学分割/泛化关键量’的MICCAI论文，可论证相位信息的新近研究趋势 |
| P1 | `2023_Lin_DeepFrequencyFiltering_CVPR.pdf` | Deep Frequency Filtering for Domain Generalization | CVPR 2023 (pp. 11797-11807) | [doi](https://doi.org/10.1109/CVPR52729.2023.01135) · [arXiv](https://arxiv.org/abs/2203.12198) | 域泛化：对特征图做FFT并学习频域注意力掩码，增强可迁移频率分量 | 在特征(而非输入)上做频域滤波以提升泛化，与‘对特征计算相位一致性/频带路由’的特征级设计思路一致 |
| P1 | `2023_Ouyang_CausalityDG_TMI.pdf` | Causality-inspired Single-source Domain Generalization for Medical Image Segmentation | IEEE TMI 42(4): 1095-1106 | [doi](https://doi.org/10.1109/TMI.2022.3224067) · [arXiv](https://arxiv.org/abs/2111.12525) | 单源域泛化:随机浅层网络做强度/纹理增广,在CT到MRI腹部分割上验证 | 域泛化代表作(含腹部CT),可作为跨中心(TCIA)外部测试与增广策略的相关工作 |
| P1 | `2021_Liu_FedDG_CVPR.pdf` | FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space | CVPR 2021 (DOI 10.1109/CVPR46437.2021.00107) | [doi](https://doi.org/10.1109/CVPR46437.2021.00107) · [arXiv](https://arxiv.org/abs/2103.06030) | 多中心医学分割的域泛化：连续频域空间插值共享幅度谱+边界导向的episodic学习 | 医学分割中频域幅度/相位解耦用于跨中心泛化的CVPR代表作，同时使用边界导向学习，对外部测试设计有参考价值 |
| P1 | `2020_Yang_FDA_CVPR.pdf` | FDA: Fourier Domain Adaptation for Semantic Segmentation | CVPR 2020 (pp. 4085-4095) | [doi](https://doi.org/10.1109/CVPR42600.2020.00414) · [arXiv](https://arxiv.org/abs/2004.05498) | 跨域分割：交换源/目标图像低频幅度谱即可对齐风格，保留相位结构 | 频谱幅度替换做域适应的奠基工作，可作为跨中心(LiTS→TCIA)偏移处理的对比/依据 |
| P2 | `2024_Qiao_RAS4DG_MICCAI.pdf` | Medical Image Segmentation via Single-Source Domain Generalization with Random Amplitude Spectrum Synthesis (RAS4DG) | MICCAI 2024 (LNCS, DOI 10.1007/978-3-031-72114-4_42) | [doi](https://doi.org/10.1007/978-3-031-72114-4_42) · [arXiv](https://arxiv.org/abs/2409.04768) | 单源域泛化：随机扰动幅度谱(高频扰动更强)，保持相位结构，含3D胎儿脑MRI | 医学3D上‘扰动幅度、保持相位’增广的MICCAI实例，可用于TCIA外部泛化的增广/对比方案 |
| P2 | `2022_Yang_DLTTA_TMI.pdf` | DLTTA: Dynamic Learning Rate for Test-time Adaptation on Cross-domain Medical Images | IEEE TMI 41(12): 3575-3586 | [doi](https://doi.org/10.1109/TMI.2022.3191535) · [arXiv](https://arxiv.org/abs/2205.13723) | 跨域测试时自适应:按样本偏移程度动态调整学习率 | 测试时自适应相关工作,用于讨论外部测试集域偏移的可能处理(非腹部CT任务) |

## 11_基础模型与预训练（16 篇）

通用分割大模型、3D 预训练与大规模腹部数据集，可作强基线或相关工作。

| 优先级 | 文件 | 论文 | 出处 | 链接 | 解决的难点 | 对本课题的用处 |
|---|---|---|---|---|---|---|
| P0 | `2023_Liu_CLIP-UniversalModel_ICCV.pdf` | CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection | ICCV 2023, pp. 21095-21107 | [doi](https://doi.org/10.1109/ICCV51070.2023.01934) · [arXiv](https://arxiv.org/abs/2301.00785) | 多数据集部分标注下的器官+多类肿瘤(含肝肿瘤)统一分割，MSD榜单第一 | MSD(含Task03肝)排行榜强基线/参考，可在相关工作与结果表中作为通用模型对照；说明评测口径与LiTS/MSD使用方式 |
| P0 | `2022_Tang_SwinUNETR-Pretrain_CVPR.pdf` | Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis | CVPR 2022 | [doi](https://doi.org/10.1109/CVPR52688.2022.02007) · [arXiv](https://arxiv.org/abs/2111.14791) | 标注稀缺:在5050例CT上做对比/掩码重建/旋转预测的自监督预训练,在MSD(含肝肿瘤)与BTCV上微调 | 3D自监督预训练代表作,Swin UNETR的CVPR正式版本(已有库中仅BrainLes版);与VoCo同属预训练线,可作对比基线与相关工作 |
| P1 | `2025_He_VISTA3D_CVPR.pdf` | VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging | CVPR 2025 | [arXiv](https://arxiv.org/abs/2406.05285) | 3D CT通用分割+交互式编辑:127类解剖结构及多种病灶,零样本泛化 | 最新的3D CT基础模型之一(含肝肿瘤),可作为大模型对照与近期相关工作 |
| P1 | `2024_Li_SuPreM_ICLR.pdf` | How Well Do Supervised 3D Models Transfer to Medical Imaging Tasks? (SuPreM / AbdomenAtlas 1.1) | ICLR 2024 | [arXiv](https://arxiv.org/abs/2501.11253) | 大规模监督预训练(9262例CT，含7类肿瘤伪标签)用于3D分割迁移 | 3D分割预训练基线与评测依据（含LiTS/MSD下游），可与自监督预训练(VoCo等)对照 |
| P1 | `2024_Ma_MedSAM_NatComm.pdf` | Segment anything in medical images | Nature Communications 15 (654) | [doi](https://doi.org/10.1038/s41467-024-44824-z) · [arXiv](https://arxiv.org/abs/2304.12306) | 通用交互式分割基础模型(MedSAM),跨模态泛化,用框提示降低标注成本 | 医学分割基础模型代表;可作为'提示式/半自动'对比或相关工作,说明其2D特性对3D肝肿瘤边界仍不足 |
| P1 | `2024_Du_SegVol_NeurIPS.pdf` | SegVol: Universal and Interactive Volumetric Medical Image Segmentation | NeurIPS 2024 (Spotlight) | [arXiv](https://arxiv.org/abs/2311.13385) | 3D通用分割基础模型:9万无标注+6千标注CT预训练,语义与空间提示,200+类别含肿瘤,zoom-out-zoom-in推理 | 3D CT基础模型强基线,可对比其对肝肿瘤的零样本/提示分割能力 |
| P1 | `2024_Li_AbdomenAtlas_MedIA.pdf` | AbdomenAtlas: A large-scale, detailed-annotated, & multi-center dataset for efficient transfer learning and open algorithmic benchmarking | Medical Image Analysis 97 (103285) | [doi](https://doi.org/10.1016/j.media.2024.103285) · [arXiv](https://arxiv.org/abs/2407.16697) | 大规模多中心腹部CT标注数据集(2万+例),用于迁移学习与开放基准测试,含肿瘤伪标注 | 可作外部多中心评测/预训练数据来源,也是评测口径与泛化基准的依据 |
| P1 | `2024_Ma_FLARE22_LancetDH.pdf` | Unleashing the strengths of unlabelled data in pan-cancer abdominal organ quantification: the FLARE22 challenge | The Lancet Digital Health 6(11): e815-e826 | [doi](https://doi.org/10.1016/S2589-7500(24)00154-7) · [arXiv](https://arxiv.org/abs/2308.05862) | 半监督腹部器官分割(50标注+2000无标注),并在北美/欧洲/亚洲外部队列上检验泛化 | 半监督/少标注与跨中心泛化的挑战赛级证据;nnU-Net类方法在其中的表现可作评测依据 |
| P1 | `2023_Wasserthal_TotalSegmentator_RadiolAI.pdf` | TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images | Radiology: Artificial Intelligence 5(5): e230024 | [doi](https://doi.org/10.1148/ryai.230024) · [arXiv](https://arxiv.org/abs/2208.05868) | 基于nnU-Net的大规模多器官CT分割,覆盖多中心/多扫描仪真实临床数据,提供可迁移预训练 | nnU-Net系统在多中心数据泛化的实证;肝脏器官分割可作肝区先验/预训练来源 |
| P1 | `2023_Ulrich_MultiTalent_MICCAI.pdf` | MultiTalent: A Multi-Dataset Approach to Medical Image Segmentation | MICCAI 2023 | [arXiv](https://arxiv.org/abs/2303.14444) | 多数据集联合训练、标签定义冲突;对病灶(含肝肿瘤)分割提升明显,可作预训练 | nnU-Net框架下的多数据集预训练强基线,与本课题同框架,便于对比 |
| P1 | `2022_Ma_AbdomenCT-1K_TPAMI.pdf` | AbdomenCT-1K: Is Abdominal Organ Segmentation a Solved Problem? | IEEE TPAMI 44(10):6695-6714 | [doi](https://doi.org/10.1109/TPAMI.2021.3100536) · [arXiv](https://arxiv.org/abs/2010.14808) | 多中心/多期相/多厂商CT下腹部器官(含肝)分割泛化性评估 | 论证跨中心、跨期相域偏移会显著降低性能，支撑外部TCIA测试设计与泛化讨论；提供肝的多中心基准 |
| P2 | `2025_Zhao_BiomedParse_NatMethods.pdf` | A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities (BiomedParse) | Nature Methods (online 2024-11-18; print 2025) | [doi](https://doi.org/10.1038/s41592-024-02499-w) · [arXiv](https://arxiv.org/abs/2405.12971) | 文本提示的统一医学图像解析基础模型,9种模态,对不规则形状目标更好 | 文本提示基础模型相关工作,可简述其对不规则肿瘤的优势 |
| P2 | `2024_Huang_SAM4MedImages_MedIA.pdf` | Segment Anything Model for Medical Images? | Medical Image Analysis 92 (103061) | [doi](https://doi.org/10.1016/j.media.2023.103061) · [arXiv](https://arxiv.org/abs/2304.14660) | 评估SAM在医学图像上的零样本能力:对边界模糊与小目标不稳定 | 为'通用大模型在低对比度肝肿瘤边界上不足'提供评测依据 |
| P2 | `2022_Xie_UniMiSS_ECCV.pdf` | UniMiSS: Universal Medical Self-Supervised Learning via Breaking Dimensionality Barrier | ECCV 2022 (LNCS 13681) | [doi](https://doi.org/10.1007/978-3-031-19803-8_33) · [arXiv](https://arxiv.org/abs/2112.09356) | 标注稀缺:利用2D与3D无标注数据的跨维度自监督预训练 | 3D自监督预训练补充相关工作 |
| P2 | `2021_Zhou_ModelsGenesis_MedIA.pdf` | Models Genesis | Medical Image Analysis 67 (101840) | [doi](https://doi.org/10.1016/j.media.2020.101840) · [arXiv](https://arxiv.org/abs/2004.07882) | 标注稀缺:3D自监督'自学'预训练模型,在含肝/肺等多个3D分割与分类任务上优于从零训练 | 3D医学自监督预训练奠基性经典,作为预训练路线的相关工作引用 |
| P2 | `2021_Zhang_DoDNet_CVPR.pdf` | DoDNet: Learning to Segment Multi-Organ and Tumors from Multiple Partially Labeled Datasets | CVPR 2021 | [arXiv](https://arxiv.org/abs/2011.10217) | 部分标注数据集:动态头一网多任务,含肝及肝肿瘤任务(MOTS基准) | 部分标注/联合训练的早期代表,可作相关工作与MOTS上肝肿瘤的参照 |

## 12_3D骨干与对比基线（13 篇）

3D Transformer / Mamba / 大核卷积骨干，外部对比基线候选。

| 优先级 | 文件 | 论文 | 出处 | 链接 | 解决的难点 | 对本课题的用处 |
|---|---|---|---|---|---|---|
| P0 | `2024_Chen_TransUNet_MedIA.pdf` | TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers | Medical Image Analysis 97:103280 | [doi](https://doi.org/10.1016/j.media.2024.103280) · [arXiv](https://arxiv.org/abs/2310.07781) | 3D 长程依赖建模；Transformer 解码器(候选区域+cross-attention)专门提升小目标/肿瘤分割，并在 MSD 胰腺肿瘤、肝血管上评测 | 3D TransUNet 基于 nnU-Net 框架，与 PhASER-3D 骨干同源，是最贴近的强基线；其“decoder 对肿瘤更有利”的结论可作为动机依据 |
| P0 | `2024_Shaker_UNETR++_TMI.pdf` | UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation | IEEE TMI 43(9):3377-3390 | [doi](https://doi.org/10.1109/TMI.2024.3398728) · [arXiv](https://arxiv.org/abs/2212.04497) | 3D 自注意力二次复杂度过高；高效配对注意力(空间+通道)兼顾精度与效率，Synapse/BraTS/MSD Lung 评测 | 高效 3D Transformer 对比基线；可与相位一致性/频带路由等低成本模块做参数量-FLOPs-精度对比 |
| P0 | `2024_Xing_SegMamba_MICCAI.pdf` | SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation | MICCAI 2024, LNCS 15008:578-588 | [doi](https://doi.org/10.1007/978-3-031-72111-3_54) · [arXiv](https://arxiv.org/abs/2401.13560) | 全体积特征的长程依赖建模，线性复杂度，含结直肠癌 3D 肿瘤分割 | Mamba 系 3D 对比基线（与已收录 U-Mamba 同类），评估长程依赖对肿瘤分割的增益 |
| P1 | `2023_He_SwinUNETR-V2_MICCAI.pdf` | Swin UNETR-V2: Stronger Swin Transformers with Stagewise Convolutions for 3D Medical Image Segmentation | MICCAI 2023（出处待核） | [doi](https://doi.org/10.1007/978-3-031-43901-8_40) | 在Swin UNETR中加入卷积以兼顾局部与长程依赖,在WORD/FLARE21/MSD上取得强结果 | Transformer类强基线(Swin UNETR改进版),可作对比 |
| P1 | `2023_Zhou_nnFormer_TIP.pdf` | nnFormer: Volumetric Medical Image Segmentation via a 3D Transformer | IEEE TIP 32:4036-4045 | [doi](https://doi.org/10.1109/TIP.2023.3293771) · [arXiv](https://arxiv.org/abs/2109.03201) | 3D 体数据的局部-全局交错注意力，缓解纯卷积感受野受限 | 3D Transformer 对比基线（TIP 正式版，与 nnU-Net 对比充分） |
| P1 | `2023_Lee_3DUX-Net_ICLR.pdf` | 3D UX-Net: A Large Kernel Volumetric ConvNet Modernizing Hierarchical Transformer for Medical Image Segmentation | ICLR 2023 | [arXiv](https://arxiv.org/abs/2209.15076) | 大核(7x7x7)深度可分离 3D 卷积模拟 Swin 的大感受野，在 FLARE/MSD 等腹部数据评测 | 大核 3D 卷积基线，是 MedNeXt 之外另一类“纯卷积长程建模”代表，可用于相关工作中论证大感受野 vs 频域全局信息 |
| P1 | `2023_Lee_RepUX-Net_MICCAI.pdf` | Scaling Up 3D Kernels with Bayesian Frequency Re-parameterization for Medical Image Segmentation (RepUX-Net) | MICCAI 2023, LNCS 14223:632-641 | [doi](https://doi.org/10.1007/978-3-031-43901-8_60) · [arXiv](https://arxiv.org/abs/2303.05785) | 超大 3D 卷积核训练不稳定；以空间频率的贝叶斯先验对卷积权重重参数化 | 大核卷积 + 频率视角，与“小波频带路由”的频域思想相近，可作为相关工作与讨论依据 |
| P2 | `2024_Liu_Swin-UMamba_MICCAI.pdf` | Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining | MICCAI 2024, LNCS 15009:615-625 | [doi](https://doi.org/10.1007/978-3-031-72114-4_59) · [arXiv](https://arxiv.org/abs/2402.03302) | Mamba 分割网络通常从零训练；引入 ImageNet 预训练提升数据效率 | Mamba 系基线；其“预训练很关键”结论可用于讨论对比公平性（2D 方法，仅作相关工作） |
| P2 | `2024_Wang_LKM-UNet_MICCAI.pdf` | LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image Segmentation | MICCAI 2024, LNCS 15008:360-370 | [doi](https://doi.org/10.1007/978-3-031-72111-3_34) · [arXiv](https://arxiv.org/abs/2403.07332) | 大核 Mamba 扩大感受野并保持局部细节，层级双向 Mamba 块建模全局/邻域 | 大感受野 + 局部细节折中的思路，与相位一致性(局部边界)+全局路由的设计取舍可对照；含 3D 腹部评测 |
| P2 | `2022_Hatamizadeh_UNETR_WACV.pdf` | UNETR: Transformers for 3D Medical Image Segmentation | WACV 2022 | [doi](https://doi.org/10.1109/WACV51458.2022.00181) · [arXiv](https://arxiv.org/abs/2103.10504) | 3D 长程依赖：纯 ViT 编码器 + U 型解码器，BTCV/MSD 肝脾脑肿瘤 | 3D Transformer 分割的奠基性基线，几乎所有对比表都会包含 |
| P2 | `2021_Xie_CoTr_MICCAI.pdf` | CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation | MICCAI 2021, LNCS 12903:171-180 | [doi](https://doi.org/10.1007/978-3-030-87199-4_16) · [arXiv](https://arxiv.org/abs/2103.03024) | 3D 长程依赖：可变形自注意力以低开销连接 CNN 与 Transformer | 早期 3D CNN-Transformer 混合代表，Synapse/MSD 对比常见 |
| P2 | `2018_Oktay_AttentionUNet_MIDL.pdf` | Attention U-Net: Learning Where to Look for the Pancreas | MIDL 2018 | [arXiv](https://arxiv.org/abs/1804.03999) | 注意力门抑制无关区域、突出小器官/小目标，免去级联定位网络 | 小目标/低对比度场景的经典对比基线，已有 BSPC 论文以其为骨干(MSML-AttUNet) |
| P2 | `2016_Dou_3D-DSN-Liver_MICCAI.pdf` | 3D Deeply Supervised Network for Automatic Liver Segmentation from CT Volumes | MICCAI 2016（此 PDF 为会议版；MedIA 2017 期刊扩展版 doi:10.1016/j.media.2017.05.001 未下载） | [doi](https://doi.org/10.1007/978-3-319-46723-8_18) · [arXiv](https://arxiv.org/abs/1607.00582) | 3D全卷积网络+深监督缓解梯度不稳；肝CT分割 | 3D上下文+深监督在肝分割中的经典依据，可作为深监督设计的引用 |

## 13_评测不确定性与校准（9 篇）

排名稳定性、统计报告与不确定性 / 校准，配合已有的 Metrics Reloaded。

| 优先级 | 文件 | 论文 | 出处 | 链接 | 解决的难点 | 对本课题的用处 |
|---|---|---|---|---|---|---|
| P0 | `2018_Maier-Hein_ChallengeRankings_NatComm.pdf` | Why rankings of biomedical image analysis competitions should be interpreted with care | Nature Communications 9:5217 | [doi](https://doi.org/10.1038/s41467-018-07619-7) | 评测可靠性：对 150 项挑战赛分析，排名随指标、测试集、标注者、排名方案而变，不稳定 | 方法依据：说明为何要报告逐病例 Dice 分布、bootstrap 置信区间与多种指标，避免过度解读 SOTA 排名 |
| P1 | `2024_Kahl_ValUES_ICLR.pdf` | ValUES: A Framework for Systematic Validation of Uncertainty Estimation in Semantic Segmentation | ICLR 2024 (oral) | [arXiv](https://arxiv.org/abs/2401.08501) | 分割不确定性方法缺乏系统验证：不确定性聚合方式被忽视，集成最稳健，TTA 为轻量替代 | 评测依据：设计不确定性/校准/失败检测实验时的标准框架；提示病例级聚合方式影响结论 |
| P1 | `2020_Mehrtash_CalibrationUncertainty_TMI.pdf` | Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation | IEEE TMI 39(12):3868-3878 | [doi](https://doi.org/10.1109/TMI.2020.3006437) · [arXiv](https://arxiv.org/abs/1911.13273) | Dice loss 训练的 U-Net 过度自信、校准差；系统比较 CE/Dice 并证明模型集成改善校准 | 评测依据：若报告不确定性/校准(ECE)，Dice loss 为主的 nnU-Net 需注意校准问题；可作为不确定性分析的方法参考 |
| P1 | `2019_Jungo_UncertaintyReliability_MICCAI.pdf` | Assessing Reliability and Challenges of Uncertainty Estimations for Medical Image Segmentation | MICCAI 2019, LNCS 11765:48-56 | [doi](https://doi.org/10.1007/978-3-030-32245-8_6) · [arXiv](https://arxiv.org/abs/1907.03338) | 体素级不确定性在数据集层面较可靠，但在单病例层面失准 | 评测依据：支持“逐病例而非全局”评估不确定性与分割质量的论点，与逐病例 Dice 口径一致 |
| P1 | `2017_Guo_Calibration-Guo_ICML.pdf` | On Calibration of Modern Neural Networks | ICML 2017 (PMLR 70) | [arXiv](https://arxiv.org/abs/1706.04599) | 现代深度网络过度自信；提出 ECE 与温度缩放 | 校准评测奠基性文献，引用 ECE 定义与 temperature scaling |
| P2 | `2023_Ross_BeyondRankings_MedIA.pdf` | Beyond rankings: Learning (more) from algorithm validation | Medical Image Analysis 86:102765 | [doi](https://doi.org/10.1016/j.media.2023.102765) | 仅看排名无法解释失败原因；用 GLMM 结合图像元数据分析算法失败因素 | 评测依据：可借鉴按病灶大小/对比度/数量分层分析失败案例，而非只报整体均值 |
| P2 | `2023_Murugesan_MarginLabelSmoothing_MedIA.pdf` | Calibrating segmentation networks with margin-based label smoothing | Medical Image Analysis 87:102826 | [doi](https://doi.org/10.1016/j.media.2023.102826) · [arXiv](https://arxiv.org/abs/2209.09641) | 分割网络校准：现有校准损失等价于 logit 距离等式约束，提出基于边距的不等式约束 | 评测/方法依据：可作为校准指标(ECE)与校准损失基线，并与前景极度不平衡场景结合讨论 |
| P2 | `2021_Czolbe_SegUncertaintyUseful_IPMI.pdf` | Is segmentation uncertainty useful? | IPMI 2021, LNCS 12729:715-726 | [doi](https://doi.org/10.1007/978-3-030-78191-0_55) · [arXiv](https://arxiv.org/abs/2103.16265) | 不确定性与分割误差正相关，但对主动学习帮助有限，不同概率分割方法能力差异大 | 评测依据/讨论：谨慎使用不确定性图作为边界证据或质量控制信号 |
| P2 | `2017_Kendall_BayesianUncertainty-Kendall_NeurIPS.pdf` | What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? | NeurIPS 2017 | [arXiv](https://arxiv.org/abs/1703.04977) | 区分偶然(aleatoric)与认知(epistemic)不确定性 | 边界模糊对应的偶然不确定性概念依据，可用于解释肿瘤边界不确定区域 |
