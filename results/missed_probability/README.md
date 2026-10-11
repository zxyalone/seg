# A0 漏检病灶概率诊断

真实模型运行于 2026-10-11，代码提交 `9385eef`；GPU 0、1、2 三路分片，A0 三 seed × 21 例 fold-0 验证集，共 63 次病例推理。最长主运行分片约 55 分钟。

主要产物：`diag_missed_probability.md`、`decision.json`、`check_argmax_vs_eval_val.csv`。`cases/` 保留逐病例原始统计，`lesions.csv` 与 `threshold_sweep_per_run.csv` 保留汇总明细，配置、checkpoint 与源文件哈希见 `run_manifest.json`。测试影像未读取，预登记条件未修改。

判定 **mixed**：q = 20/39 = 51.28%；预算内最佳阈值 `p_tumor > 0.1`，去掉 <4 mm 碎块后，低 CNR 召回 63.25% → 64.10%（+0.85 个百分点），FP/scan 1.63 → 1.94。未达到决策层的 +10 个百分点条件，也不满足表征层的 q≤20% 条件。按原计划，下一阶段为模块 pilot，并包含 A0 + 调阈值基线以及相同 FP/scan 的低 CNR 召回比较。

每个 seed 的 92 个参考病灶检出及 21 例 FP 计数与既有 eval_val 完全一致。体素级 TP/FP/FN 绝对差合计 283，独立记录在 `integrity_checks.json`，不声称逐体素完全复现。概率合法性、病例完整性、CNR 和自动判定均已核对。

可直接用已归档统计重新汇总，无需 GPU、CT 或 checkpoint：

```bash
python experiments/analysis/missed_lesion_probability.py summarize --out results/missed_probability
```

其他任务停止的 E11/E12 控制器没有恢复；其停止记录和未提交改动未纳入本次归档。
