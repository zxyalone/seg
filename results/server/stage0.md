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

## 阶段1已得到的结果

- Copy2已保存预测指标复算4/4 PASS。
- 128³全精度梯度检查遇到CUDA OOM，按清单允许的回退改为96³/50steps/9groups。结果需等完整SUMMARY，不能把部分组PASS当整体完成。

## 后续顺序

梯度检查全部通过 → 独立目录重跑Copy2旧路径推理并检查指标 → 阶段3勘误脚本 → 提交检查日志 → 阶段2六组冒烟及manifest核验 → 阶段4 A6/A0/A1 200epoch。

未通过任一步时停止后续训练；尤其冒烟OOM时不自行修改PHASER_PC变量。阶段5使用预先提交的`experiments/DECISION_GATE.md`判断研究分支。阶段6/7及待补wrapper需根据已完成的前置证据继续执行，当前不视为已完成。
