# TriALS 参考模型的补充结果（计划 §11，2026-10-14 计算前登记；描述性）

TriALS 测试集预测由 `TriALS-main/train_new_split/09_test_inference.sh` 生成：checkpoint_best（训练中按验证折 EMA 伪 Dice 选出）、`--disable_tta`。早先的说明写成 checkpoint_final，已更正。SAMed-H 最后约 15 个 epoch 伪 Dice 降为 0，checkpoint_best 为第 79 epoch。

脚本 `experiments/analysis/reference_extras.py`，日志 `run.log`。

- `params.csv`：各 checkpoint 网络权重的参数量。SAMed 的 checkpoint 只存 LoRA 与解码器权重，总数 = 原始 SAM 权重 + LoRA。本研究模型 31,196,399（A1 多 15 个门控参数）。2026-10-14 更正：nnU-Net 类解码器保存了对编码器的引用，同一张量在 state dict 中出现多次（encoder.* 与 decoder.encoder.*），初版逐键求和导致重复计数（本研究模型、nnU-Net 3D/2D、U-Mamba 偏大）；现按存储去重、不计整数缓冲，结果与优化器中的参数数一致。
- `tta_table.csv`（及 `tta_*_pp4mm.csv`）：nnU-Net 3D、SegResNet、U-Mamba 已有的镜像 TTA 预测（checkpoint_best）按冻结协议同一流程评估。MedNeXt-L 的 TTA 预测只有 2 例，不用。
- `boundary_cases.csv`、`boundary_summary.csv`：全部 8 个参考模型无 TTA 测试预测的边界与体积指标（与 `boundary_metrics.py` 同一定义）。

未使用：各模型 `validation/`（由最终权重生成，与测试集权重不一致）；Swin UNETR 验证预测只有 2 例；LightM-UNet、MedNeXt-B 未完成训练；UBHMNet 只训练 38 epoch；TriALS 设置下的早期 PhASER 无法按相同设置重新推理，按用户指示不纳入。
