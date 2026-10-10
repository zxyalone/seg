# 外部肝肿瘤 CT 队列候选

核查日期：2026-10-10。用户要求等待 E11–E13 验证结果后再定阶段 7 终点。本次只调研元数据和公开来源，没有下载整套影像、冻结协议或运行外部推理。

**建议优先准备 Colorectal-Liver-Metastases，HCC-TACE-SEG 作为备选或第二个域外队列。** 这是根据任务、标注和服务器空间作出的适配判断，尚未确定最终外部集；两者的预训练独立性仍需核验。

## 候选比较

| 候选与官方来源 | 当前规模 / 全包容量 | 数据与标注 | 本项目的适配判断 |
| --- | --- | --- | --- |
| [TCIA Colorectal-Liver-Metastases](https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/) | 197 位患者，约 10.91 GB | 术前门静脉期增强 CT；DICOM 与 DICOM-SEG；肝、多个肿瘤、血管及未来剩余肝标注 | 首选：适合当前单通道肝肿瘤分割与多病灶召回问题，容量较合适 |
| [TCIA HCC-TACE-SEG](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/) | 105 位患者，约 28.57 GB | 治疗前后多期 CT；治疗前肝、肿瘤、血管的临床审核分割，DICOM-SEG | 第二选择：原发肝癌与时相 / 层厚的域差异更大；只选有匹配标注的治疗前 CT，不能将所有期相 / 术后扫描混作独立病例 |

规模和容量来自当前 TCIA 页面。CRLM 的 [原始数据论文](https://www.nature.com/articles/s41597-024-02981-2) 也明确描述 197 位患者的术前门静脉期 CT 及对应分割。旧 Wiki 的 CRLM 196 / 10 GB 和 HCC 26.6 GB 与当前版本不同，下载时按实际清单核对。

HCC 的 [官方质量说明](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230229) 记录 HCC_001 的 CT/SEG 网格尺寸差异；转换必须按引用的 Series/SOP UID 和物理坐标对齐，不能只按数组尺寸贴标签。

当前数据盘约 29 GiB 可用。HCC 全包与转换影像、临时文件及分析预测同时保留会使空间紧张，宜先筛选标注对应的治疗前序列并核算容量。CRLM 相对更容易准备。两个队列以阳性病例为主，不能替代当前 LiTS 阴性病例的假阳性分析。

## 预训练独立性：仍未完成核验

已检查 [作者仓库](https://github.com/Luffy03/Large-Scale-Medical) 与 [PreCT-160K 数据卡](https://huggingface.co/datasets/Luffy503/PreCT-160K)。作者公开的 [Omni-supervised LiTS 清单](https://github.com/Luffy03/Large-Scale-Medical/blob/main/Omni-supervised/jsons/dataset_lits.json) 包含 training 108 / validation 23。按 `liver_<ID>` 与 `LiTS_<ID>` 在本地对照，当前 27 例留出测试中，22 例出现在该 training 列表，5 例出现在 validation 列表；[healthy_ct_liver 清单](https://github.com/Luffy03/Large-Scale-Medical/blob/main/Omni-supervised/jsons/healthy_ct_liver.json) 也有显式 LiTS 路径。

这只是公开清单的编号匹配，不能证明当前 MD5 对应 checkpoint 的确切训练来源及曝光方式，但不能将本轮内部留出划分直接等同于“未被预训练见过”。[VoComni 清单](https://github.com/Luffy03/Large-Scale-Medical/blob/main/VoComni/VoComni.json) 使用重新编号的 `VoComni_*` 路径，没有 HCC/CRLM 字符串不能证明无重叠。

目前缺少覆盖本次 checkpoint 来源的完整原始患者 / Study / Series UID 映射。因此两个候选均记为**外部候选，预训练独立性未核验**，不写成已确认的干净外部证据；A8 保持取消。

## 准备顺序（尚未执行）

1. 获取患者、CT/SEG 引用关系和标注名称元数据，按患者选一套标注对应的术前 / 门静脉期 CT，预先记录数据质量排除规则，不根据模型外部成绩筛选病例。
2. 所有 `Tumor_*` 并集映射 label 2，肝映射 label 1，背景为 0，肿瘤优先；区分整肝、未来剩余肝、血管和腹主动脉。
3. 先对少量病例核查方向、物理网格、HU 和标签叠加，再核算全部转换空间与独立性。转换检查不需要模型外部推理。
4. E11–E13 与 E10 完成后，再记录主要终点、对比方法、后处理、外部病例清单及排除规则；冻结后一次性推理，外部集不用于调参。

本次只筛选当前最相关的两个候选，没有开展全医学影像数据集检索。外部集尚未准备，阶段 7 仍未冻结。
