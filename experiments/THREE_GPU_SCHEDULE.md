# 三 GPU 执行安排

2026-10-09。按 `服务器操作清单.md` 阶段 4 → 5 → 6 执行。
阶段 0–3 使用已归档的 `3773ed0` 及其父提交证据；当前服务器数据完整性及权重 MD5 已检查。

## 阶段 4

| GPU | 正式运行 |
|---|---|
| 0 | A6，seed 0 |
| 1 | A0，seed 0 |
| 2 | A1，seed 0 |

保留已有进度。模型、损失、patch 128³、batch 2、200 epoch 和 seed 不变。
训练输出位于 `$DATA/nnUNet_results_v2/<组>_s<seed>/`。
三组完成 200 epoch、21 例无 TTA 验证及评测后才进入阶段 5。
不使用训练日志中的 pseudo Dice 选择分支。

三组阶段 4 进程全部结束后，自动执行 `audit_stage4_nonfinite.sh`：
从每组最新的 `training_log_*.txt` 汇总 non-finite gradient 和 non-finite loss 次数，
并按 epoch 统计梯度跳过次数（按用户命令显示前 20 个 epoch）。
输出保存为 `$DATA/nnUNet_results_v2/_orchestration/stage4_nonfinite_audit.log`。
该统计仅输出核查日志，non-finite gradient/loss 次数不作为暂停条件。
checkpoint、200 epoch 运行记录和 21 例验证结果完整，且没有进程硬错误时，
核查日志输出后自动进入阶段 5，再按决策结果进入阶段 6；不增加人工确认步骤。

阶段 4 已有后台进程由 tmux 中的调度器接管监控，训练进程不重启。
新任务由该 tmux 调度器启动。GPU 利用率和运行状态持续写入
`$DATA/nnUNet_results_v2/_orchestration/`。

## 阶段 5

使用已冻结的 `DECISION_GATE.md`。判据同步记录在 `manuscript/实验清单.md`。
按清单运行 `stats.py --cohort Tumor`，保存 `results/gate_val.csv`；
另保留 All 队列统计以报告阴性病例的 FP/scan。
三组评测与实际 manifest 一并归档、提交后进入阶段 6。

## 阶段 6

三张卡共用有序待执行队列。空出的卡立即领取下一项就绪任务，
不用固定每张卡必须等某一条长队列。

分支 B 的有序训练队列（2026-10-10 更新）：

1. A6 seed 1
2. A0 seed 1
3. A1 seed 1
4. A6 seed 2
5. A0 seed 2
6. A1 seed 2
7. A7 seed 0
8. A8 seed 0（可选，预算不足时可省略）

E16 成本测量随后执行。理由见 `manuscript/实验清单.md` §4：
分支 B 的 A6/A1 核心比较需要对等的 3 seed 覆盖。
这是执行计划变更，冻结的阶段 5 判据与决策结果不变。

更新时已在运行的 A7_s0、A6_s1、A0_s1 保留原进度，A7 不重复启动。
剩余任务按新队列优先派发；空闲 GPU 无需等待可选的 A8 队尾任务。
分支 A 仍使用原队列，并保留 A5 与 E9。

E9 使用清单指定的 ResEncUNetPlanner 与 200 epoch trainer。
只有 plans 的预处理器、间距、归一化、重采样、轴顺序和强度统计
完全相同时，才复用已有预处理缓存；保留 ResEnc 自己的网络 plans、patch 和 batch。
否则执行独立预处理，并先检查磁盘余量。验证额外生成无 TTA 的 21 例预测。
E16 的当前 `cost.py` 是合成 plans 下的 patch 测量；真实病例总推理时间仍按清单后续记录。

阶段 4 不同组耗时不同，较早完成的卡可能等待阶段 5 的共同依赖。
阶段 6 在就绪任务足够时由三卡并行运行；队尾及协议依赖造成的等待应在日志中如实记录。

## 后续约束与 provenance

E10–E14 的分析工具、后处理选择、主要终点和外部队列完成后，
才冻结阶段 7 协议并执行一次性测试。
此调度器不自动启动测试集预测，不用未完成实验填表。

初始阶段 4 的真实 manifest 保留原始 `f37d353-dirty` 记录；
启动时的 tracked diff 为空，dirty 来自当时未提交的审阅报告和病例清单。
原始状态已在 orchestration 目录归档，不改写已有 manifest。
后续运行使用已提交的工作区，训练与评测实现未因此次排程调整而修改。

tmux 中的调度入口：

```bash
python experiments/server_three_gpu.py --old-controller <现有调度器PID>
```

接管阶段 6 时使用 `--resume-stage6`。更新会暂挂旧监督进程，保留它已启动的 GPU 任务，
全部旧任务退出后才回收旧监督进程；新增任务在新的 tmux 会话窗口中启动。
`--skip-a8` 可省略尚未启动的可选 A8。不得重复启动同一组实验。
