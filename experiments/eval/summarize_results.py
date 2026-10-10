#!/usr/bin/env python3
"""Audit archived validation results and write the stage 0–6 summary.

Uses the current server's checkpoints/logs plus the archived CSVs, manifests,
and stats.py outputs. Never runs training or reads test-set labels/predictions.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics

REPO = Path(__file__).resolve().parents[2]
FOLD = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
NAMES = [f'{group}_s{seed}' for group in ('A0', 'A1', 'A6') for seed in range(3)] + ['A7_s0']


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def hours(job):
    fmt = '%Y-%m-%d %H:%M:%S CST'
    return (datetime.strptime(job['ended_at'], fmt) - datetime.strptime(job['started_at'], fmt)).total_seconds() / 3600


def audit(root):
    state = load(root / '_orchestration/state.json')
    assert state['stage'] == 'training_complete_analysis_pending'
    expected = set((REPO / 'experiments/eval/val_cases.txt').read_text().split())
    assert len(expected) == 21
    rows = []
    for name in NAMES:
        fold = root / name / FOLD
        archive = REPO / 'results' / name
        manifest = load(fold / 'run_manifest.json')
        summary = load(archive / 'eval_val/summary.json')
        with (archive / 'eval_val/cases.csv').open() as stream:
            cases = list(csv.DictReader(stream))
        assert len(cases) == 21 and {c['case'] for c in cases} == expected
        tumor_cases = [c for c in cases if c['has_tumor'] == 'True']
        assert len(tumor_cases) == 18
        assert summary['All']['n_cases'] == 21 and summary['Tumor']['n_cases'] == 18
        for cohort, selected in [('All', cases), ('Tumor', tumor_cases)]:
            for metric in ['tumor_dice', 'tumor_iou', 'liver_dice', 'liver1_dice']:
                actual = statistics.mean(float(c[metric]) for c in selected)
                assert math.isclose(actual, summary[cohort][metric + '_mean'], abs_tol=1e-12)
            for thr in ['0.1', '0.5']:
                counts = {key: sum(int(c[f'les_{key}@{thr}']) for c in selected) for key in ['tp', 'fp', 'fn']}
                assert all(counts[key] == summary[cohort][f'lesion@{thr}'][key] for key in counts)
        assert manifest['epochs_completed'] == manifest['num_epochs'] == 200
        assert manifest['seed'] == int(name.rsplit('_s', 1)[1])
        assert manifest['env'].get('PHASER_VAL_TTA', '0') == '0'
        config = load(fold.parent / 'plans.json')['configurations']['3d_fullres']
        assert config['patch_size'] == [128, 128, 128] and config['batch_size'] == 2
        actual_sha = sha256(fold / 'checkpoint_final.pth')
        assert actual_sha == manifest['checkpoint_final_sha256'], name + ' checkpoint mismatch'
        for relative in ['run_manifest.json', 'eval_val/summary.json', 'eval_val/cases.csv', 'eval_val/lesions.csv', 'eval_val/pred_lesions.csv']:
            assert sha256(fold / relative) == sha256(archive / relative), name + ' archive mismatch'
        log = sorted(fold.glob('training_log_*.txt'))[-1].read_text(errors='replace')
        times = [float(v) for v in re.findall(r'Epoch time:\s*([\d.]+)\s*s', log)]
        assert len(times) == 200, name + ' incomplete epoch log'
        assert not any(v in (root / name / 'launcher.log').read_text(errors='replace').lower()
                       for v in ['out of memory', 'traceback', 'cuda error'])
        job = state['jobs'][name]
        assert job['status'] == 'complete'
        t, a = summary['Tumor'], summary['All']
        rows.append(dict(run=name, group=name.split('_s')[0], seed=manifest['seed'], epochs=200, val_cases=21,
                         tumor_cases=18, whole_liver_dice_all=a['liver_dice_mean'], tumor_dice=t['tumor_dice_mean'],
                         tumor_iou=t['tumor_iou_mean'], tumor_hd95_mm=t['tumor_hd95_mean'], tumor_nsd=t['tumor_nsd_mean'],
                         lesion_recall=t['lesion@0.1']['recall'], lesion_precision=t['lesion@0.1']['precision'],
                         lesion_f1=t['lesion@0.1']['f1'], fp_per_scan_tumor=t['lesion@0.1']['fp_per_scan'],
                         lesion_f1_all=a['lesion@0.1']['f1'], fp_per_scan_all=a['lesion@0.1']['fp_per_scan'],
                         epoch_mean_seconds=statistics.mean(times), occupied_wall_hours=hours(job),
                         gradient_skips=log.count('non-finite gradient'), nonfinite_losses=log.count('non-finite loss'),
                         started_at=job['started_at'], ended_at=job['ended_at'], git_commit=manifest['git_commit'],
                         checkpoint_final_sha256=actual_sha, checkpoint_sha256_verified=True, archived_outputs_match=True))
    return state, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=REPO.parent / 'nnUNet_data/nnUNet_results_v2')
    args = parser.parse_args()
    state, runs = audit(args.results_root)
    result = REPO / 'results'
    write_csv(result / 'stage0_6_runs.csv', runs)
    group_rows = []
    metrics = ['whole_liver_dice_all', 'tumor_dice', 'tumor_iou', 'tumor_hd95_mm', 'tumor_nsd', 'lesion_recall',
               'lesion_precision', 'lesion_f1', 'fp_per_scan_tumor', 'lesion_f1_all', 'fp_per_scan_all']
    for group in ['A0', 'A1', 'A6', 'A7']:
        selected = [r for r in runs if r['group'] == group]
        row = {'group': group, 'n_seeds': len(selected)}
        for metric in metrics:
            values = [r[metric] for r in selected]
            row[metric + '_mean'] = statistics.mean(values)
            row[metric + '_seed_sd_ddof1'] = statistics.stdev(values) if len(values) > 1 else ''
        group_rows.append(row)
    write_csv(result / 'stage0_6_groups.csv', group_rows)
    (result / 'server/stage0_6_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    shutil.copyfile(args.results_root / 'E16_cost/launcher.log', result / 'server/e16_cost.log')
    cost_text = (result / 'server/e16_cost.log').read_text()
    costs = []
    for match in re.finditer(r'^(A[016])\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)', cost_text, re.M):
        g, params, phaser, overhead, ms, gib = match.groups()
        costs.append(dict(group=g, params=int(params.replace(',', '')), phaser_params=int(phaser.replace(',', '')),
                          overhead_percent=float(overhead), forward_ms=float(ms), peak_allocated_gib=float(gib)))
    assert len(costs) == 3
    write_csv(result / 'stage0_6_cost.csv', costs)
    stats = {}
    for cohort, file in [('Tumor', 'stage6_stats_tumor.csv'), ('All', 'stage6_stats_all.csv'), ('A7', 'stage6_stats_a7_seed0.csv')]:
        with (result / file).open() as stream:
            stats[cohort] = list(csv.DictReader(stream))
    group_map = {r['group']: r for r in group_rows}
    def metric(group, key, percent=True):
        row = group_map[group]
        factor = 100 if percent else 1
        mean, sd = row[key + '_mean'] * factor, row[key + '_seed_sd_ddof1']
        return f'{mean:.2f} ± {sd * factor:.2f}' if sd != '' else f'{mean:.2f}（单 seed）'
    def stat(cohort, comparison, key):
        return next(r for r in stats[cohort] if r['comparison'] == comparison and r['metric'] == key)
    def comparison_rows(cohort, pairs):
        out = []
        for pair in pairs:
            for key, label, factor in [('tumor_dice', 'Tumor Dice（pp）', 100), ('lesion_f1@0.1', '病灶 F1（pp）', 100), ('fp_per_scan@0.1', 'FP/scan（个/例）', 1)]:
                r = stat(cohort, pair, key)
                out.append([pair, label, f"{float(r['delta']) * factor:+.3f}",
                            f"[{float(r['ci_lo']) * factor:+.3f}, {float(r['ci_hi']) * factor:+.3f}]",
                            f"{float(r['p_holm']):.4f}" if r['p_holm'] else '不适用'])
        return out
    gate = state['gate']
    a61 = stat('Tumor', 'A6 - A1', 'tumor_dice')
    a60 = stat('Tumor', 'A6 - A0', 'tumor_dice')
    occupied = sum(r['occupied_wall_hours'] for r in runs)
    stage4 = sum(r['occupied_wall_hours'] for r in runs if r['run'] in ['A0_s0', 'A1_s0', 'A6_s0'])
    a8 = state['jobs']['A8_s0']
    now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    sections = [f'''# 阶段 0–6 实验结果汇总

汇总时间：{now}（北京时间）。来源：本地归档原始验证结果、正式训练 manifest、服务器日志及已有阶段 0–3 证据。数据截至阶段 6 训练队列于 2026-10-10 20:23:28 收尾。

**已完成 10 个正式 run：A0/A1/A6 各 3 个 seed，加 A7 seed 0；均训练 200 epoch 并完成 21 例全卷验证。** 阶段 5 决策为分支 B；A8 按用户要求取消。阶段 6 的训练与 patch 成本测量完成，E10–E14 等分析尚未完成，阶段 7 一次性测试未启动。

三 seed 的含肿瘤病例 Dice：A0 **{metric('A0', 'tumor_dice')}%**，A1 **{metric('A1', 'tumor_dice')}%**，A6 **{metric('A6', 'tumor_dice')}%**（均值 ± seed 间样本 SD，3 seeds）。A6−A1 为 **{float(a61['delta']) * 100:+.3f} pp**，病例配对 95% CI **[{float(a61['ci_lo']) * 100:+.3f}, {float(a61['ci_hi']) * 100:+.3f}] pp**，Holm 校正 p={float(a61['p_holm']):.4f}，当前验证结果不能支持相位方法稳定优于梯度对照的结论。

## 1. 完成范围与阶段证据

| 阶段 | 实际状态 | 结果及来源 |
| --- | --- | --- |
| 0 环境准备 | 已完成 | 83 / 21 / 27 病例划分互不重叠，权重 MD5 与依赖核验；[阶段 0–3 记录](server/stage0.md) |
| 1 GPU 检查及回归 | 已完成 | 128³ 全精度 OOM 后按清单回退 96³ / 50 步，A0–A8 全部 PASS；Copy2 已存预测复算与独立重新推理均 4/4 PASS |
| 2 六组冒烟 | 已完成，保留记录缺项 | A0/A1/A5/A6/A7/A8 各 2 epoch、21 例验证、hash 核验通过、无非有限警告；历史逐组显存峰值未采集 |
| 3 旧机制脚本重跑 | 已完成 | s44 / s44b 已重跑并完成勘误，使用旧 checkpoint 的 legacy 路径 |
| 4 首批正式训练 | 已完成 | A0_s0 / A1_s0 / A6_s0 均 200 epoch + 21 例验证；[非有限核查日志](server/stage4_nonfinite_audit.log) |
| 5 决策门 | 已完成，分支 B | 冻结判据、病例比较和三个 run 结果已归档；[决策记录](gate_decision.json) |
| 6 第二批训练及成本 | 训练队列已完成，分析待完成 | 6 个补充 seed + A7 + E16 patch 成本；A8 已取消；E10–E14 尚未完成 |
| 7 冻结协议及一次性测试 | 未开始 | 无 PROTOCOL_FROZEN，无本轮 eval_test / eval_external 结果 |

阶段 0–3 的证据来自另一台服务器，记录提交 92e51f6 / 3773ed0；其冒烟运行的 manifest 为 8746bde。旧 Copy2 回归采用既有 24 例含肿瘤病例口径，不与本轮 18 例验证指标混合。旧回归包含已有旧模型测试预测，不能误写为本轮 V2 测试结果。GPU 实验代码与训练配置未在本次汇总中修改。

## 2. 数据与统一训练、评测口径

- Dataset102_LiTS，fold 0：训练 83 例、验证 21 例、留出测试 27 例；验证中含肿瘤 18 例、阴性 3 例。
- 3 × RTX 3080 Ti（12 GB）；patch 128³、batch 2、200 epoch、每 epoch 250 iterations、AMP。DA 进程 6，评测进程 4，CPU 数学线程 1。
- 使用 epoch 200 的 `checkpoint_final.pth`；全卷验证关闭 TTA，未按 pseudo Dice 或最佳 checkpoint 选模型。
- 主肿瘤指标来自 Tumor 队列 n=18；整肝 Dice 来自 All 队列 n=21，整肝为 label>0，不与肝实质 label=1 混用。病灶匹配 IoU>0.1，NSD 容差 2 mm。
- 以下百分比单位为 %；模型间百分比差使用 pp（百分点）。FP/scan 单位为个/例，HD95 单位为 mm。
- 跨 seed 描述表先计算每个 run 的指标，再取 3 seeds 均值和样本 SD（ddof=1）；每个 run 的原始 summary 中病例 SD 使用 ddof=0，两者不能混用。

## 3. 各 run 验证结果
''', table(['Run', '整肝 Dice All (%)', 'Tumor Dice (%)', 'Tumor IoU (%)', 'HD95 (mm)', 'NSD (%)', '病灶 F1 Tumor (%)', 'FP/scan Tumor'],
           [[r['run'], f"{r['whole_liver_dice_all'] * 100:.2f}", f"{r['tumor_dice'] * 100:.2f}", f"{r['tumor_iou'] * 100:.2f}", f"{r['tumor_hd95_mm']:.2f}", f"{r['tumor_nsd'] * 100:.2f}", f"{r['lesion_f1'] * 100:.2f}", f"{r['fp_per_scan_tumor']:.3f}"] for r in runs]),
'''原始数据按 run 保存在 `results/<run>/eval_val/`；详细机器可读汇总见 [run 汇总 CSV](stage0_6_runs.csv)。所有行均为无统一 E10 后处理的本轮原始验证结果。

## 4. 三 seed 描述统计
''', table(['组', 'Seeds', '整肝 Dice All (%)', 'Tumor Dice (%)', 'Tumor IoU (%)', '病灶 F1 Tumor (%)', 'FP/scan Tumor'],
           [[g, str(group_map[g]['n_seeds']), metric(g, 'whole_liver_dice_all'), metric(g, 'tumor_dice'), metric(g, 'tumor_iou'), metric(g, 'lesion_f1'), metric(g, 'fp_per_scan_tumor', False)] for g in ['A0', 'A1', 'A6', 'A7']]),
'''表中 ± 为 seed 间样本 SD；A7 只有 seed 0，不能估计 seed 波动。机器可读全指标见 [组汇总 CSV](stage0_6_groups.csv)。这里的病灶 F1 是三个 run 的 F1 均值；下一节配对统计使用平均病例 TP/FP/FN 后的 pooled F1，非线性比值会使两种结果略有差别。

## 5. 多 seed 配对比较

直接运行既有 `experiments/eval/stats.py`：先对同一病例跨 seeds 平均，再以病例为重采样单位做 10,000 次配对 bootstrap（随机种子 0）。n=18/21，不能把同一病例的三次预测当作 54/63 个独立病例。CI 条件于当前三个已训练 seed，不包含额外 seed 重采样，也不等同对概率或预测做 ensemble。

病例级指标做双侧 Wilcoxon，同一指标内对 A6−A0 / A6−A1 两个比较做 Holm 校正；病灶指标按平均病例计数重算 pooled ratio，无 Wilcoxon / Holm p 值。此处是验证集分析，非阶段 7 确认性测试。

### 5.1 含肿瘤验证病例（Tumor，n=18）
''', table(['比较', '指标', '差值', '病例配对 95% CI', 'p（Holm）'], comparison_rows('Tumor', ['A6 - A0', 'A6 - A1'])),
'''来源：[多 seed Tumor 统计](stage6_stats_tumor.csv)。A6 对 A0 的肿瘤 Dice CI 跨 0；含肿瘤病例的 pooled 病灶 F1 和 FP/scan 区间朝有利方向，但不应替代主要终点或写成外部验证结论。

### 5.2 全部验证病例（All，n=21，包含阴性病例）
''', table(['比较', '指标', '差值', '病例配对 95% CI', 'p（Holm）'], comparison_rows('All', ['A6 - A0', 'A6 - A1'])),
'''来源：[多 seed All 统计](stage6_stats_all.csv)。纳入阴性病例后，A6 对 A0 的病灶 F1 / FP/scan 区间均跨 0；A6 对 A1 的 F1 点估计更低、FP/scan 更高，两个区间也跨 0，不能只报告 Tumor 子集的有利变化。

### 5.3 A7 容量 / 空间置换对照（仅同 seed 0，Tumor n=18）
''', table(['比较', '指标', '差值', '病例配对 95% CI', 'p（Holm）'], comparison_rows('A7', ['A6 - A7'])),
'''来源：[A7 seed 0 配对统计](stage6_stats_a7_seed0.csv)。A6−A7 的 Dice CI 跨 0，当前单 seed 对照不足以证明收益来自相位空间信息；不把 A7 单 seed 与 A6 三 seed 均值当作对等稳定性比较。

## 6. 已冻结决策门与当前结论
''', f'''阶段 5 在 seed 0 验证完成后选择 **分支 {gate['branch']}**：A6−A1 tumor Dice {gate['delta_dice'] * 100:+.3f} pp，病灶 F1 {gate['delta_f1'] * 100:+.3f} pp，FP/scan {gate['delta_fp_per_scan']:+.3f}。Dice 为正但 F1 / FP 两项均未改善，因此依原冻结判据走 B。

补齐三 seed 后，A6−A1 Dice 缩小为 {float(a61['delta']) * 100:+.3f} pp；seed 2 出现 A6 低于 A1。A6−A0 Dice 的三 seed 均值差为 {float(a60['delta']) * 100:+.3f} pp，但其 CI 同样跨 0。现有证据支持继续按分析论文分支报告方法行为、稳定性和成本，尚不支持“相位稳定优于梯度”或“相位空间信息已被容量对照证实”的表述。

阶段 6 加入 A1 seeds、取消 A8 均属于执行计划调整，没有改变阶段 5 判据或历史决策。

## 7. 运行完整性、数值核查与实际耗时

本次逐 run 核验了 200 条 epoch 耗时、manifest 完成轮数 / seed、patch / batch、21 个唯一验证病例及其集合、Tumor 18 例计数，并从 cases.csv 复算 Dice/IoU 与病灶 TP/FP/FN。**10 / 10 个最终 checkpoint 的 SHA256 与原始 manifest 一致；归档 manifest 和四个 eval_val 文件与服务器原件逐文件一致。** 没有进程硬错误；非有限梯度跳过如实记录，按此前用户要求不视为失败。未重新运行训练或推理。
''', table(['Run', '平均 epoch (s)', '占卡墙钟 (h)', '梯度跳过', '非有限 loss', 'Checkpoint SHA256', '训练 git'],
           [[r['run'], f"{r['epoch_mean_seconds']:.2f}", f"{r['occupied_wall_hours']:.3f}", r['gradient_skips'], r['nonfinite_losses'], 'PASS', r['git_commit'][:7] + '-dirty' if '-dirty' in r['git_commit'] else r['git_commit'][:12]] for r in runs]),
f'''正式完整 run 合计占卡墙钟 **{occupied:.3f} GPU·h**：阶段 4 为 {stage4:.3f}，阶段 6 完整训练 run 为 {occupied - stage4:.3f}。A8 停止前另消耗 {hours(a8):.3f} GPU·h，E16 另消耗 {hours(state['jobs']['E16_cost']):.4f} GPU·h。此统计由调度记录起止时刻相加，包含数据加载、训练、验证和评测，属于占卡墙钟而非 GPU 内核执行时长；不含另一台服务器阶段 0–3 时间。

首批开始于 2026-10-09 21:21:04，本批队列收尾于 2026-10-10 20:23:28。阶段 4 日志中的非有限梯度分别为 A6=18、A0=17、A1=17；各 run 复核计数及精确 SHA256 见 [run 汇总 CSV](stage0_6_runs.csv)。最初三个 manifest 保留实际的 `f37d353-dirty`，启动时 tracked diff 为空，dirty 来源为当时未提交的审阅报告及病例清单；没有改写历史 provenance，也不宣称这三个历史 manifest 为 clean。

## 8. E16 成本测量
''', table(['组', '参数量', 'PhASER 参数', '参数增量 (%)', 'Forward (ms)', '峰值分配显存 (GiB)'],
           [[r['group'], f"{r['params']:,}", f"{r['phaser_params']:,}", f"{r['overhead_percent']:.3f}", f"{r['forward_ms']:.1f}", f"{r['peak_allocated_gib']:.2f}"] for r in costs]),
'''原始输出：[E16 日志](server/e16_cost.log)，机器可读：[成本 CSV](stage0_6_cost.csv)。`cost.py` 使用合成六层 plans、128³ 单 patch、batch 1、eval/no_grad/AMP、预热 3 次及重复 20 次，在 GPU0 RTX 3080 Ti 上测量；这是该构造下的参数量和 patch forward 成本，不是训练显存峰值、真实全病例总推理时间或训练 FLOPs。A6 的记录 forward 时间约为 A0 的 2.30 倍。E16 仅部分完成：FLOPs 和 27 例真实推理总耗时尚无本轮记录。

## 9. 旧机制重跑结果（阶段 3）

s44b 修正为平均秩 Spearman 后，LiTS_006 / 009 / 017 的 cap vs full PC Spearman 分别为 0.586934 / 0.606552 / 0.567990；非零子集为 0.470821 / 0.485598 / 0.491644。来源：`step2_mechanism/out/s44b_pc_cap.json`。

s44 的 Copy2 平均绝对选择性为 0.000301547，平均相对空间 SD 为 0.001809892，来源：`step2_mechanism/out/s44_routes.json`。这些都是旧 checkpoint 的勘误与复现结果，不能作为本轮 V2 已训练模型的 E13 后半证据。

## 10. 尚未完成、跳过及取消的项目

| 项目 | 状态 | 尚缺内容 |
| --- | --- | --- |
| E7 A5 正式训练 | 分支 B 跳过 | 仅有历史冒烟，不作正式消融 |
| E8 A8 正式训练 | 用户取消 | 无完整最终训练及验证结果；[取消记录](server/a8_cancellation.json)，部分输出保留在服务器 |
| E9 ResEnc | 分支 B 跳过 | 不写成已完成强基线 |
| E10 | 待完成 | 统一后处理的验证集选择与固定规则 |
| E11 | 待完成 | 推理期干预的批量验证结果 |
| E12 | 待完成 | 0.75 / 0.5 / 0.25 对比度退化结果 |
| E13 后半 | 待完成 | 本轮 V2 checkpoint 的路由 / 滤波器分析 |
| E14 | 待完成 | 外部数据集确定、独立性核查与格式转换 |
| E16 剩余 | 部分完成 | FLOPs、27 例真实总推理时间；patch 成本已测 |
| E17 | 待完成 | 预训练病例重叠核查；取消 A8 不取消此核查 |
| 阶段 7 / E15 | 未开始 | 固定主要终点 / 后处理 / 外部队列，冻结协议后一次性测试 |

## 11. 复算与文件索引

```bash
PY=/root/autodl-tmp/envs/mamba/bin/python
# 对 Tumor 和 All 分别运行；A0 / A1 / A6 都加载三个 seed
for COHORT in Tumor All; do
  OUT=$(printf '%s' "$COHORT" | tr '[:upper:]' '[:lower:]')
  $PY experiments/eval/stats.py --cohort "$COHORT" \\
    --method A0=results/A0_s0/eval_val,results/A0_s1/eval_val,results/A0_s2/eval_val \\
    --method A1=results/A1_s0/eval_val,results/A1_s1/eval_val,results/A1_s2/eval_val \\
    --method A6=results/A6_s0/eval_val,results/A6_s1/eval_val,results/A6_s2/eval_val \\
    --compare A6:A0 A6:A1 --n-boot 10000 --seed 0 --out results/stage6_stats_$OUT.csv
done
$PY experiments/eval/stats.py --cohort Tumor \\
  --method A6=results/A6_s0/eval_val --method A7=results/A7_s0/eval_val \\
  --compare A6:A7 --n-boot 10000 --seed 0 --out results/stage6_stats_a7_seed0.csv
python experiments/eval/summarize_results.py
```

本报告 `results/实验结果汇总.md`，run / group / cost CSV 与三个 stats CSV 为派生汇总；原始逐病例及逐病灶 CSV、summary、manifest 不变。最终调度状态归档为 [stage0_6_state.json](server/stage0_6_state.json)，对应 `training_complete_analysis_pending`。Checkpoint 和预测体积留在服务器，未提交到 Git。
''']
    document = '\n\n'.join(sections).rstrip() + '\n'
    temp = result / '实验结果汇总.tmp'
    temp.write_text(document)
    temp.replace(result / '实验结果汇总.md')
    print('Audited 10 complete runs and checkpoint hashes; wrote results/实验结果汇总.md and run/group/cost CSVs.')


if __name__ == '__main__':
    main()
