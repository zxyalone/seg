# TriALS 训练设置下的比较（计划 §11，2026-10-14 训练前登记；描述性）

T0 = nnU-Net（PlainConvUNet，31.2 M，与 TriALS nnU-Net 3D 同一结构），T13b = T0 + DCA + CFC（参数同 A13b）。
训练设置同 TriALS nnUNetTrainer_100epochs：从头训练、SGD lr 1e-2（所有参数）poly 衰减、100 epoch；种子 0–2。
测试集分别用 checkpoint_best（同 TriALS，本块主报告）和 checkpoint_final 预测，无 TTA；指标流程同冻结协议。

- `TRIALS_SETTING_RESULTS.md`、`trials_setting_stats.json`：各组均值与配对病例 bootstrap 95% CI（10 000 次，种子 20261012）。
- `{val,test_best,test_final}_{per_group,per_run,strata}.csv`、`*_tables/`：4 mm 病灶 / 病例表。
- 日志：`runner.log`、`train_*.log`、`predict_*.log`、`val_rerun_T13b_s2.log`、`metrics_*.log`、`stats.log`。

运行中断与恢复（均不影响结果，只是重做未完成的步骤）：
1. T13b_s2 的最终验证在导出第 21 例（LiTS_105）时丢失导出子进程而挂起（三个验证同时导出，90 GB 容器内存上限；与 A14 seed 0 相同）。停止后由 `experiments/server_trials_setting_recover.sh` 单独重跑验证（--val，checkpoint_final）。
2. AutoDL 容器在 2026-10-15 约 03:19 停止、06:17 重启（非本项目脚本所致）；T0_s0、T0_s1 的 checkpoint_final 测试预测中断（24/27），恢复脚本重新预测全部 27 例。
