# 文献索引

共 33 篇，2026-10-09 整理。各论文的划分、统计口径和结果对比见 [文献对比表.md](文献对比表.md)。命名规则：`年份_第一作者_简称_期刊.pdf`。"BSPC-arXiv"表示已在 BSPC 发表、此处存的是 arXiv 预印本。

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
