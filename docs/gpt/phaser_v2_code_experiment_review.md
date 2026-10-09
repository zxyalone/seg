# PhASER V2：代码、实验记录与服务器清单审阅

审阅日期：2026-10-09。基线：`main`，`f37d353343c9a88ac851f8dbf5321cb414fbb67c`。

结论：当前版本可以继续做合成检查，但正式实验前应修复运行身份与恢复逻辑；旧权重机制分析和最终测试流程也需要修订。发现 **6 项 P1、6 项 P2**。本次只新增审阅报告，没有修改训练、评测、论文或原始结果。

数据盘正在拷贝。数据、权重、预测缺失均不计为代码缺陷。没有运行真实病例训练、Copy2 重新推理、内部测试或外部测试；下面的动态检查全部使用合成输入或临时目录。

## 1. P1：优先处理

### R01：训练后验证中断，重启队列无法恢复

位置：`experiments/run.sh:41–47,53–59`；`nnUNet/nnunetv2/run/run_training.py:203–208`。

触发：训练已经写出 `checkpoint_final.pth`，随后全卷验证因中断或导出失败而没有生成完整预测。

`run.sh` 看到 final 就完全跳过 `nnUNetv2_train`，直接评测 `validation/`。重新启动队列仍走同一路径，因此不会补跑 `perform_actual_validation`，会反复失败。服务器清单第 112 行“直接重新执行即可恢复”的说法没有覆盖这个状态。

验证：在临时目录使用未改动的 launcher、一个 final 占位文件和一个验证病例。预测存在时返回 0；删除该预测后重跑返回 1，报告 `missing predictions`，训练/验证入口没有被调用。

建议：分开判断训练完成、验证完成与评测完成；final 存在但验证不完整时，用 final 权重执行 `--val`，再评测。验证完成标记应包含病例清单、checkpoint 哈希和无 TTA 设置。

### R02：旧配置会被当成当前配置的完成实验

位置：`experiments/run.sh:23,34,41–46`；`nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_phaser.py:799–815,859–929`。

运行目录只由组名和 seed 决定。已有 final 无条件跳过，已有 latest 无条件继续；没有核对 epoch 预算、代码、配置、plans、数据划分或预训练权重身份。恢复时 manifest 的 `git_commit/env/start_time` 又会被当前信息覆盖，丢失原始启动信息。

验证：临时目录中的 manifest 明确写着 2 epoch，而 launcher 使用 A6 的 200 epoch 配置；脚本仍返回 0，跳过训练并生成 `eval_val`。正常 smoke 默认使用单独目录，因此不是说默认 smoke 一定污染正式目录；问题出现在 `EPOCHS`、`RESULTS_ROOT`、配置或代码变化后复用目录。

建议：启动前冻结并校验 run specification；至少包含配置/代码版本、seed、预算、plans/split 和预训练哈希。身份不同应使用新目录或明确报错。保留首次 manifest，恢复记录另列。V2 正式续训不应悄悄容忍网络键缺失或 optimizer 恢复失败。

### R03：要求预训练的组会在权重不存在时静默从头训练

位置：`nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_phaser.py:386–392`；`phaser/unet.py:478–498`；`experiments/run.sh:30–34`。

非空 checkpoint 路径如果不存在，模型构造仍成功；加载函数也允许匹配键很少甚至没有匹配键。A0/A1/A6 因此可能实际从头训练，却继续沿用“VoComni 预训练”的组名和说明。manifest 没有记录实际加载的键数及预训练文件哈希；默认权重路径未显式进入环境时，也不能从 env 还原该来源。

这是静态代码缺陷，**没有把当前拷贝过程中的文件缺失当作一次失败实验**。清单第 39 行的人工 md5 检查有帮助，但训练入口自身没有落实该条件。

建议：明确区分“刻意不加载”（A8 的空字符串）和“要求加载但找不到”。后者直接报错；核对预期层的加载覆盖率，保存权重哈希及实际加载情况。

### R04：清单让 s44 重跑旧权重，但脚本默认走 V2 推理

位置：`step2_mechanism/scripts/s44_routes.py:44–51,65–75`；`step2_mechanism/scripts/s45_verify_fixes.py:213–227`；`experiments/服务器操作清单.md:92–101`。

`s44_routes.py` 加载的 Copy2 等权重来自旧版本，却没有设置 `PHASER_LEGACY=1`。新 trainer 默认 `legacy=False`，会改变 prior 的概率映射和 PC 噪声参考尺度。服务器按清单重跑，得到的不是仅修正选择性统计后的原实验，还混入了推理行为变化。s45 的模型回归部分也有同类问题。

`reproduce_metrics.py` 已显式开启 legacy，说明仓库已有正确兼容入口；s44/s45 没有沿用它。这里依据代码路径判断，真实权重的变化幅度因数据拷贝尚未实测。

建议：分析旧权重时显式使用 legacy、清理影响行为的环境变量并记录配置；分析 V2 时加载 V2 final 和其 manifest。两类结果分开。不能用“权重全部加载成功”代替前向一致性检查。

### R05：s45 对能量比的反推错误，相关机制结论不成立

位置：`step2_mechanism/scripts/s45_verify_fixes.py:140–149`；`phaser/modules.py:358–369`；`step2_mechanism/p0_fix_report.md:73–90`。

模块实际计算：

```text
T_eff = min(T_Kovesi, κ × den)
PC = max(num − T_eff, 0) / (den + floor + 1e−12)
```

脚本设 κ=0.999，然后把 `mean(PC)+0.999` 当作 `mean(num/den)`。这个等式既没有确认相对阈值分支被选中，也忽略了截断和分母保护项；第 144 行注释的阈值不等号还写反了。

已保存 JSON 给出 edge=`1.0028890977131897`、interior=`0.9991614884380441`。按定义，`num=||Σv_s||`、`den=Σ||v_s||`，三角不等式要求比值不超过 1。该反推值不能作为“能量比几乎完全重叠”的测量证据，也不能据此推出“公式没有判别力”“换证据形式才有出路”。

建议：在 forward 中直接取逐体素 `num/den`，用相同边界/内部 mask 计算分布和 AUC。保留四变体的真实 AUC/PC 占比记录，把依赖错误反推的解释单独撤回。κ=0.5 在所测设置下变差的直接结果并不因此作废。

### R06：测试脚本没有落实“冻结协议”和“一次性测试”

位置：`experiments/predict_test.sh:19–20,32–49`；`experiments/服务器操作清单.md:149,155–173`。

脚本只检查 `PROTOCOL_FROZEN` 文件存在，没有读取/校验其 commit，也没有验证病例清单、配置、主要终点或后处理是否改变。成功后的同一 run 可以再次执行，默认预测会覆盖结果。若 E10 选择了后处理，当前脚本仍直接评测原始 `test_pred`，没有执行该规则；单独提供一个外部数据路径时则会静默跳过外部测试。

因此现有脚本提供的是人为流程提醒，尚不能支持论文第 4.3 节所写的冻结保证。

建议：冻结结构化协议及相关文件哈希；保存 checkpoint、推理和评测 provenance；已成功测试的 run 拒绝覆盖。允许同一协议下恢复未完成推理。后处理必须实现或明确冻结为 none；外部输入应成对校验。ResEnc 的测试 wrapper 也需要补齐，当前 A 组脚本不能直接测试 ResEnc。

## 2. P2：结果稳健性与清单一致性

### R07：A7 推理随机置换没有由 PHASER_SEED 固定

位置：`phaser/modules.py:527–531`；`nnUNet/nnunetv2/inference/predict_from_raw_data.py:99–103`；`experiments/predict_test.sh:28`。

置换在 eval 中也执行。预测器调用静态 `build_network_architecture`，不会执行 trainer `__init__` 中的 `_seed_everything`；设置环境变量 `PHASER_SEED` 并没有为预测进程初始化 RNG。相同 checkpoint/输入的 A7 预测可能不同，且受滑窗顺序影响。

验证：对一个 router 权重非零的合成 PFCR 模块重复 eval，两次输出最大差值 `0.012203693389892578`，虽然环境中一直为 `PHASER_SEED=0`。这个数只展示随机路径，不是实际训练模型的差值。

建议：明确 A7 推理协议，并在预测入口设种子。若要求与遍历顺序无关，置换应由病例/patch/层的稳定标识生成。记录随机策略及必要的重复推理波动。

### R08：评测只验证 shape/spacing，接受物理位置不一致的体数据

位置：`experiments/eval/evaluate.py:224–229`。

没有核对 origin 和 direction。形状/间距相同、但坐标原点不同或方向翻转的预测仍被按数组索引比较，可能产生虚假的 Dice、HD95 和病灶匹配。

验证：合成预测与 GT 的 origin 相差 100 mm，数组相同；脚本接受并输出 tumor Dice=1、HD95=0。

建议：验证完整物理网格，默认拒绝不一致。若允许重采样，使用显式、记录在协议中的标签重采样流程。

### R09：跨方法病例不一致时，统计脚本静默取交集

位置：`experiments/eval/stats.py:83–87`。

`load_method` 检查了同一方法不同 seed 的病例一致性，`compare` 却直接取两个方法的病例交集。缺预测/用错病例清单会改变分析队列，且没有列出丢掉的病例。

验证：A 方法病例 A/B，B 方法病例 A/C；比较正常返回，只报告 n=1。

建议：要求两方法覆盖同一冻结队列、病例键唯一，并核对评测 conventions。另需分别调用/汇总 Tumor 与 All 队列：`--cohort Tumor` 的 FP/scan 不包含无肿瘤病例，而论文主表要求 FP/scan 覆盖 All27。不能直接把清单第 124 行的输出当作所有主表指标。

### R10：FP32 路径不防非有限梯度，梯度自检也可能误报 PASS

位置：`nnUNetTrainer_phaser.py:736–751`；`experiments/check_grads.py:167–181,200–202`。

`grad_scaler is None` 时没有检查梯度范数是否有限。有限 loss 也可能产生 Inf 梯度；clip 之后参数可能变 NaN。梯度自检仅以“曾出现非零梯度”判 PASS，NaN 也会被 `count_nonzero` 计为非零；`moved` 没有进入 FAIL 判据。

验证：用有限的 `sqrt(p)`、p=0 构造单参数网络调用实际 FP32 `train_step`，返回 loss=0，但更新后参数为 NaN。该探针验证防护缺口，并不声称当前合成 PhASER 训练出现了 NaN。

建议：两个分支统一检查有限梯度；自检区分非有限梯度、结构性零梯度、有效参数更新和已知无效 bias。当前 9 组 PASS 只证明短程合成输入下没有被判定为死梯度的 PhASER 张量，不证明全部参数有效学习、AMP 稳定或真实病例训练有效。

### R11：A1/A6 比较不能单独归因于“相位优于幅度”

位置：`experiments/configs/A1.env:1–6`；`A6.env:1–9`；`manuscript/实验清单.md:34,43`。

A1 是梯度 skip gate；A6 同时增加 PFCR 路由、prior head、budget 和 ranking。A6−A1 混合了证据类型、结构、容量和训练监督。E12 若同样只比较这几组，也只能观察完整方案的退化曲线。

建议：该比较可以用于阶段 5 的方案决策，但论文归因要限于完整方案。若要隔离证据类型，应在同一 PFCR/prior/budget/监督条件下替换 PC 与梯度证据；A7 则回答 PC 空间排列的增量作用。是否追加训练属于实验设计选择，本次没有擅自增设组。

### R12：清单遗漏必需步骤，实验记录仍有过时状态与过强措辞

位置及具体修订：

- `manuscript/实验清单.md:47` 标 E17 预训练重叠核查为必需，但服务器清单从 E0–E16 排程，没有 E17 和测试前完成条件。把 E17 补入 CPU 排程和冻结前检查；本次未核验实际预训练病例重叠，不能当作已经排除。
- `step2_mechanism/report_step2_mechanism.md:398` 仍称 V2 默认改成 2²²、不再降采样；当前 configs/trainer 和合并说明均为 2²⁰。该句需要更新。
- `manuscript/sections/04_experiments.md:288–294` 的 open questions 仍称当前代码是旧版双 sigmoid、Phase B 延后监督、没有标量 wd 排除等；当前 main 已实现这些 V2 修复，预算也已在 configs 定为 200。逐项标记解决状态，保留真正未决的主要终点、后处理和外部集。
- `step2_mechanism/p0_fix_report.md:149` 用三个舍入后的 route 均值声称“逐位一致”，没有逐体素前向差值或哈希支持。改为均值在所展示精度下一致，完整回归另测。`design_review.md` 对球体无噪声 PC 最大变化的概括也漏掉原报告/JSON 中的 0.0230；不能统一写为 ≤0.0086。
- 清单第 112 行的恢复承诺依赖 R01 修复。`queue.sh:25–28` 会吞掉单任务失败且整体常返回 0；清单中的 `queue.sh | tee` 没有设置 pipefail。应汇总失败并提供整体失败状态。
- E9/E10/E11/E12/E13 的待写项目已在清单诚实标注，应保留待完成状态；阶段 7 需要逐项验收，不应只验冻结文件存在。
- 时间估算基于单张 3090。当前可见 GPU 为 3 张 RTX 3080 Ti，每张 12 GB；显存与耗时应按实际机器冒烟确认。没有执行正式计划的显存验证。

## 3. 已通过的检查与证据矩阵

| 项目 | 本次结果 | 支持的范围 |
|---|---|---|
| 三个实验 shell 脚本 `bash -n` | 通过 | shell 语法；不证明恢复/冻结语义 |
| `experiments/eval/test_eval.py` | `all checks passed` | 现有已知答案病例、基本匹配/表面指标、同方法统计与 Holm；stats 在空 bootstrap 样本上有 RuntimeWarning |
| 9 组梯度检查 | A0–A8 全 PASS | GPU、64³、batch 2、6 步、FP32；未执行清单要求的 128³/50 步或真实 AMP 训练 |
| 参数量 | A0=31,196,399；A6=31,338,040；增量=141,641，约 0.454% | 6 stage 合成 plans；与 README 一致，真实 plans 尚未复核 |
| CNR 记录 | 518；中位 1.615540；阈值 1.278544 / 1.920765 | 已提交 JSON 与表述/冻结常量一致；本次没有从原始 CT 重算 |
| 验证病灶计数 | 总计 91，完成 87，truncated 4 | 已提交 JSON 与报告一致 |
| 修正后的 route 选择性 | Copy2 全尺度/病例平均绝对选择性 0.00030154718 | 与 JSON summary 一致，逐带取绝对值的勘误成立；旧权重重跑需先修 R04 |
| cap 合成 pilot | CNR=2：V0 AUC≈0.999679，V1≈0.502250 | 与报告舍入一致；只支持该几何/配置的直接观测 |
| 能量比机制推论 | 不支持 | R05 的反推错误，不应作为直接测量 |
| V2 真实效果、最终外部泛化、实际预训练重叠 | 未核验 | 数据拷贝未完成，所需实验/材料尚未验证 |

短程 GPU 梯度检查命令：

```bash
PYTHONPATH=nnUNet OMP_NUM_THREADS=2 python experiments/check_grads.py \
  --groups A0 A1 A2 A3 A4 A5 A6 A7 A8 --steps 6 --patch 64 --device cuda
```

边界探针在临时目录验证了 R01/R02/R07/R08/R09/R10，没有读取真实病例或改动数据盘。没有发现需要标为 P0 的问题；没有进行文献/BibTeX 核验或整稿科学审稿。全部新数值均来自本次运行或已提交 JSON，未补写缺失实验结果。

## 4. 建议修订顺序

1. 数据拷贝期间修 R01–R03 和 R08–R10，保证正式运行身份、恢复与评测可靠；修 R07 的预测随机协议。
2. 修 R04/R05，保留原始 JSON，把无效反推和过强结论在报告中单独勘误。
3. 同步清单状态及 E17；数据完整后再执行真实 plans、128³/50 步、AMP 冒烟和 Copy2 legacy 回归。冒烟结果不能复用为 200 epoch 正式结果。
4. E9/E10–E14 的实际工具及协议验收完成后，再落实 R06 并进行最终测试。实验比较的归因按 R11 收紧，原始直接观测继续保留。

代码修订与实验协议变更分别形成可审查 diff；本报告只提出问题与修订建议，不更改方法、配置或原始测量。
