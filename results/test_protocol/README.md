# 测试集一次性评估（冻结协议 c2b8cae）

`docs/测试集冻结协议_草案.md` 冻结后，2026-10-12 19:06 – 2026-10-13 02:15 在服务器上运行一次（HEAD 9ce222f，`experiments/server_test_protocol.sh`）。27 例内部测试集此前从未推理。

## 结果（`stats/PROTOCOL_RESULTS.md`）

- 低 CNR 病灶 62 个，分布在 19 例中。
- 主要终点（4 mm 后处理后的低 CNR 召回，3 seed 均值）：A0 62.37%，A1 62.90%，A11 66.67%，A13a 68.28%，**A13b 71.51%**。
- 确认性比较（配对 bootstrap 10 000 次，种子 20261012，Holm）：
  - **A13b vs A1：+8.60 个百分点 [+2.13, +14.55]，Holm p = 0.011 → 提高**
  - **A13b vs A0：+9.14 个百分点 [+3.03, +14.44]，Holm p = 0.006 → 提高**
- 留一病例：贡献最大的病例 LiTS_000 +1.94 个百分点，去掉后差值 +6.67；增益不依赖单一病例。
- 次要终点（描述）：全部病灶召回 A1 73.78% / A13b 73.12%，FP/scan 1.94 / 2.06，肿瘤 Dice 74.07% / 73.97%，F1 75.90% / 74.94%。分层显示低 CNR 小病灶（< 10 mm）召回 36.8% → 49.4%，同时高 CNR 小病灶 68.5% → 60.9%。
- 同等 FP/scan（≤ A13b argmax 点 2.062）：A13b 74.19%（2 seed 达到预算），A1 63.71%（2 seed），A0 62.90%（3 seed）。

## 一致性

- 15 个 run 的 argmax 逐病灶检出与 eval_test 全部一致，未过滤 FP 差 0（`curves/check_argmax_vs_eval_val.csv`）。
- 4 mm 层面 parity 全部通过（`curves/parity/`）。

## 执行记录（如实）

- 第 1–3 步按运行器原样完成（`runner.log`、`predict_*.log`、`metrics.log`）。
- 第 4 步（测试集阈值曲线）为加速与恢复，由 `execution/` 中的脚本在同一冻结代码上续跑：先把每卡 CPU 工作进程从 2 增至 3（`test_protocol_speedup.sh`）；随后大体积病例导致导出进程被系统终止（无 traceback）三次：shard 1 在 LiTS_092、shard 2 在 LiTS_109 之后、LiTS_117（219 Mvox、75 个病灶）的一个工作池。每次都从断点续跑（已写结果自动跳过、全部可读），工作进程数逐步降到 1（`test_protocol_finish*.sh`）。工作进程数不改变数值（验证集逐字节复核）。最终 405/405 个结果、parity 通过后，运行器从第 5 步继续完成（`finish4.log`）。
- 没有任何测试结果被重算后替换，也没有修改任何代码、配置或协议。
