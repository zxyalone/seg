# GPU服务器执行记录

日期：2026-10-09（北京时间）。执行目录：`/root/autodl-tmp/Downstream_2`。

## 阶段0

- 已从远端main同步到`f37d353`，原目录改动与未跟踪文件保存到stash：`server-local-changes-before-main-sync-20261009`。
- nnU-Net导入路径为本仓库`nnUNet/nnunetv2/__init__.py`。
- pandas/scipy/SimpleITK依赖可导入；`experiments/eval/test_eval.py`全部通过。
- train83/val21/test27互不重叠，病例清单已提交为`1e074a4`。
- VoComni权重MD5为`ef44f2fb457ea3652c3566848b40b338`；在清单默认路径`/root/autodl-tmp/Downstream/VoComni_nnunet.pt`建立指向原权重的软链接。
- 实际GPU：RTX3080Ti，12GB显存，torch2.1.0+cu118；不是清单预算假设的24GB RTX3090。
- cgroup内存硬上限30GiB，软上限28.5GiB；开工时磁盘可用约37GB，高于清单要求的20GB。
- 已安装tmux。长任务保存于tmux，阶段日志写入`/root/autodl-tmp/nnUNet_data/v2_server_checks/`。
- 所有训练统一`nnUNet_n_proc_DA=2`、`nnUNet_def_n_proc=1`，OMP/MKL/OpenBLAS线程数4；仅限制并发，不调整patch、batch或PHASER_PC配置。

## 阶段1（已完成）

- Copy2已保存预测指标复算4/4 PASS；独立目录重新生成27例预测后，4项指标再次全部PASS。
- 128³全精度梯度检查遇到CUDA OOM，按清单允许的回退改为96³/50steps/9groups。现已得到完整SUMMARY：A0–A8全部PASS。

## 后续顺序

梯度检查全部通过 → 独立目录重跑Copy2旧路径推理并检查指标 → 阶段3勘误脚本 → 提交检查日志 → 阶段2六组冒烟及manifest核验 → 阶段4 A6/A0/A1 200epoch。

未通过任一步时停止后续训练；尤其冒烟OOM时不自行修改PHASER_PC变量。阶段5使用预先提交的`experiments/DECISION_GATE.md`判断研究分支。阶段6/7及待补wrapper需根据已完成的前置证据继续执行，当前不视为已完成。

## 阶段2（冒烟训练及完整性核验完成）

A0/A1/A5/A6/A7/A8六组，seed0，各完成2epoch、21例无TTA验证和eval_val评测。manifest均记录干净的8746bde；最终权重SHA256、预测病例集合及无non-finite警告核验通过。各组manifest和summary见results/smoke/，耗时见smoke_summary.csv。仅用于管线验证，2epoch分割指标不作方法有效性结论。

逐组训练显存峰值没有连续采集，CSV字段留空，清单对应项保持未勾选；不补写估算值。已完成的六组日志无OOM。

## 阶段3（已完成）

s44b_pc_cap.py和s44_routes.py已完成重跑，s44选择性自检通过，旧checkpoint使用legacy路径。两份JSON和日志见step2_mechanism/out/，已由8746bde记录。

本次提交范围是正式训练前的准备、回归、冒烟和勘误重跑；不包含阶段4的训练权重、预测或未完成结果。阶段4任务独立在tmux继续运行。
