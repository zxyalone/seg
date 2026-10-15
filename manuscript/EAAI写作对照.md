# EAAI / ESWA / KBS 写作对照

本文件说明 `论文_中文_EAAI初稿.md` 参照了哪些论文、借鉴了它们的哪些写法，以及与旧稿 `论文_中文_可检出性训练.md` 相比改了什么。

## 1 参照论文

检索方法：在 Crossref 上按期刊 ISSN（EAAI 0952-1976，ESWA 0957-4174，KBS 0950-7051）检索 2023 年以后的文章，挑出与本文方向最接近的几篇（肝肿瘤分割、病灶级增强、一致性训练、CT 上的训练策略）。ScienceDirect 拒绝自动抓取，下表中标"全文"的几篇是通过 arXiv 读到的全文，其余只核对了题录。

| 论文 | 期刊 | 与本文的关系 | 读到的内容 |
|---|---|---|---|
| Jin 等，Iterative pseudo-labeling based adaptive copy-paste supervision for semi-supervised tumor segmentation | KBS 324, 2025 | 最接近：针对小肿瘤的复制粘贴增强 + 一致性，也用 LiTS | 全文（arXiv 2508.04044） |
| Hu 等，Trustworthy multi-phase liver tumor segmentation via evidence-based uncertainty | EAAI 133, 2024 | 同为 CT 肝肿瘤分割 | 全文（arXiv 2305.05344） |
| Zheng 等，Tissue-contrastive semi-masked autoencoders for segmentation pretraining on chest computed tomography | EAAI 163, 2026 | 思路最接近 CFC：按 CT 值区间遮挡 + 原图与遮挡视图的特征一致；同为"训练策略型"论文 | 全文（arXiv 2407.08961） |
| Yang 等，Multi-scale camouflaged feature mining and fusion network for liver tumor segmentation | EAAI 148, 2025 | 同样针对与肝实质难区分、易漏检的肿瘤（结构路线） | 题录 |
| Yu 等，IAPA: Intensity-anatomy synergistic lesion sparse atlas and prior-constrained positive sample augmentation for multi-lesion segmentation | ESWA 333, 2027 | 基于强度先验的病灶正样本增强，与 DCA 同属"按先验构造训练样本" | 题录 |
| Ju 等，An uncertain boundary region-aware network for multi-scale liver tumor segmentation | ESWA 304, 2026 | 肝肿瘤分割（结构路线） | 题录 |

以上 6 篇都已核对 Crossref 题录并加入 `refs.bib`，在初稿的引言和相关工作中引用。

## 2 EAAI 的硬性要求

来源：EAAI 投稿指南（通过检索结果核对，ScienceDirect 页面无法直接抓取）。

| 要求 | 初稿中的处理 |
|---|---|
| 摘要须分别说明"人工智能方面的贡献"和"工程应用"（否则直接拒稿） | 英文摘要用 "As the contribution in artificial intelligence…" 和 "As the engineering application…" 两句明确分开；中文摘要对应加粗 |
| 标题和摘要中不得出现未定义的缩写（否则直接拒稿） | 英文标题和摘要中没有任何缩写：CT 写作 computed tomography，CNR 写作 contrast-to-noise ratio，DCA/CFC/nnU-Net/LiTS 都不出现（脚本已检查） |
| 单栏排版（否则直接拒稿） | 最终英文稿用 Elsevier 单栏模板 |
| 摘要不超过 250 词 | 249 词 |
| 亮点 3–5 条，每条不超过 85 个字符（含空格），单独提交 | 5 条，73–80 个字符 |

## 3 从参照论文借鉴的写法

| 写法 | 参照论文中的做法 | 初稿中的对应 |
|---|---|---|
| 引言中编号列出现有方法的局限 | KBS 2025 列了（1）到（5）五条局限 | 第 1 节列四条局限：结构收益不稳定、实例级损失牺牲精确率、增强不以可检出性为目标、缺少按对比度分层和冻结协议的评估 |
| 用一句话单独写出研究问题 | EAAI 2024 在引言中用引号单独写出 "Can we design…?" | 第 1 节用引用块写出研究问题 |
| 贡献分三条，每条有加粗的引导词 | KBS 2025：研究问题 / 技术设计 / 有效性与泛化性 | 第 1 节：研究问题 / 技术设计 / 有效性与可信度 |
| 引言末尾说明论文结构 | 三篇都有"本文其余部分安排如下" | 第 1 节末尾 |
| 相关工作分小节，每节结尾说明与本文的区别 | KBS 2025 在相关工作末尾写 "These constraints motivate our approach" | 第 2 节四个小节，各以"与本文的区别"收尾；新增 2.3 节"不变性与一致性学习"，说明 CFC 与 TCS-MAE 的区别 |
| 方法中给出伪代码 | KBS 2025 给出算法 1 | 新增算法 1（一次训练迭代） |
| 符号表 | EAAI 论文常见 | 新增表 2 |
| 数据集单独成节 | KBS 2025 第 4 节"数据收集与预处理" | 新增第 5 节"数据集与预处理" |
| 实验按"实现细节 → 指标 → 对比方法 → 定量比较 → 定性比较 → 消融"组织 | KBS 2025 第 5 节 | 第 6 节按此顺序组织，新增 6.10 节定性比较（图 5 待做） |
| 讨论中写局限与未来工作 | 三篇都有 "Limitations and future work" | 第 7.5 节 |
| 结论只写一段 | 三篇都是 | 第 8 节一段 |
| 文末声明 | Elsevier 要求：CRediT、利益冲突、数据可用性、生成式 AI 使用声明 | 已加入各节标题，内容待填 |

## 4 与旧稿相比的主要改动

1. **新增：** 英文标题、亮点、英文摘要、关键词（6 个）。
2. **摘要：** 从"背景 / 方法 / 实验 / 结果 / 结论"的结构化摘要改为单段，明确分开 AI 贡献与工程应用。
3. **引言：** 1.1–1.4 小节合并为连续叙述，加入编号局限、研究问题和论文结构段。
4. **相关工作：** 加入 6 篇 EAAI/ESWA/KBS 文献，新增"不变性与一致性学习"小节。
5. **漏检诊断：** 改为第 3 节"问题分析"，内容不变，表述更紧凑。
6. **方法：** 新增算法 1 和符号表；"计算开销"改为"计算复杂度"，强调不引入可学习参数、推理成本不变。
7. **实验：**
   - 数据集单独成节。
   - 正文表格用方法名（基线、门控基线、DCA、本文方法），结果文件中的组名放在第 6.3 节表格的括号里。
   - 验证集和测试集各用一张完整表（表 4、表 5）。
   - 迁移实验中写入了已确认的训练不稳定现象。
8. **讨论：** 新增 7.3 节"CFC 的作用与适用条件"（含 SegResNet 上的发现）和 7.4 节"工程应用价值"（EAAI 看重）。
9. **声明：** 新增 CRediT、利益冲突、数据可用性和生成式 AI 使用声明的位置。

## 5 待完成

- 6.9 节表 11：SegResNet 迁移结果（服务器上正在推理和评估，完成后填入）。
- 6.10 节图 5：定性比较图（需要从测试集预测中挑选病例并出图）。
- 作者信息与各项声明。
- 中文稿定稿后写英文全文，使用 Elsevier 单栏模板。
