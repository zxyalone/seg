#!/usr/bin/env python3
"""Frozen test protocol (docs/测试集冻结协议_草案.md, confirmed 2026-10-12).

metrics   fixed 4-mm lesion/case tables from the saved masks of every run
          (lowcnr_pilot_metrics.metrics; raw detections and FP counts are
          asserted against the saved evaluation of the same split).
stats     primary endpoint, confirmatory comparisons, secondary endpoints,
          strata, leave-one-case-out sensitivity and the equal-FP comparison.

Dry run on validation (must reproduce §6c: A13b − A1 = +7.692 pp):
  python test_protocol.py metrics --split val  --out <dir> --jobs 16
  python test_protocol.py stats   --tables <dir> --sweep <validation threshold_sweep_per_run.csv> --out <dir>
Test (only after experiments/PROTOCOL_FROZEN is committed):
  python test_protocol.py metrics --split test --out <dir> --jobs 16
  python test_protocol.py stats   --tables <dir> --sweep <test threshold_sweep_per_run.csv> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lowcnr_pilot_metrics import metrics

REPO = Path(__file__).resolve().parents[2]
DATA = Path('/root/autodl-tmp/nnUNet_data')
GROUPS = ('A0', 'A1', 'A11', 'A13a', 'A13b')     # 3 seeds each
DESCRIPTIVE = ('A12_s0',)                         # literature baseline, no test
METHOD = 'A13b'                                   # plan §6c
COMPARISONS = (('A13b', 'A1'), ('A13b', 'A0'))    # primary, secondary; Holm
N_BOOT, BOOT_SEED, ALPHA = 10_000, 20261012, 0.05
SPLITS = {
    'val': dict(pred_subdir='validation', eval_subdir='eval_val',
                gt_dir=DATA / 'nnUNet_raw/Dataset102_LiTS/labelsTr',
                cases_file=REPO / 'experiments/eval/val_cases.txt'),
    'test': dict(pred_subdir='test_pred', eval_subdir='eval_test',
                 gt_dir=DATA / 'nnUNet_raw/Dataset101_LiTS/labelsTr',
                 cases_file=REPO / 'experiments/eval/test_cases.txt'),
}


def runs():
    return [f'{g}_s{s}' for g in GROUPS for s in range(3)] + list(DESCRIPTIVE)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _one(run, ns, cases):
    _, c, l = metrics(run, ns, cases)
    return c, l


def cmd_metrics(a):
    split = SPLITS[a.split]
    if a.split == 'test' and not (REPO / 'experiments/PROTOCOL_FROZEN').is_file():
        raise SystemExit('experiments/PROTOCOL_FROZEN missing: the test set stays locked')
    ns = argparse.Namespace(results_root=a.results_root, gt_dir=str(split['gt_dir']),
                            pred_subdir=split['pred_subdir'], eval_subdir=split['eval_subdir'])
    cases = Path(split['cases_file']).read_text().split()
    assert len(cases) == len(set(cases))
    rs = runs()
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            parts = list(ex.map(_one, rs, [ns] * len(rs), [cases] * len(rs)))
    else:
        parts = [_one(r, ns, cases) for r in rs]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.concat([c for c, _ in parts]).to_csv(out / 'cases_pp4mm.csv', index=False)
    pd.concat([l for _, l in parts]).to_csv(out / 'lesions_pp4mm.csv', index=False)
    (out / 'metrics_manifest.json').write_text(json.dumps(dict(
        split=a.split, runs=rs, n_cases=len(cases), cases=cases,
        pred_subdir=split['pred_subdir'], eval_subdir=split['eval_subdir'], gt_dir=str(split['gt_dir']),
        min_diameter_mm=4.0, connectivity=26, iou=0.1), indent=2) + '\n')
    print(f'tables written for {len(rs)} runs x {len(cases)} cases')


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def _gs(run):
    g, s = run.rsplit('_s', 1)
    return g, int(s)


def holm(p):
    order = np.argsort(p); m = len(p); adj = np.empty(m); running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj


def group_recall(det, n, run_index, group):
    """Seed-mean of pooled recall; det/n: [runs, ...] summed counts."""
    idx = run_index[group]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.mean(det[idx] / n[idx], axis=0)


def cmd_stats(a):
    tab = Path(a.tables); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = pd.read_csv(tab / 'lesions_pp4mm.csv'); C = pd.read_csv(tab / 'cases_pp4mm.csv')
    L['group'], L['seed'] = zip(*L.run.map(_gs)); C['group'], C['seed'] = zip(*C.run.map(_gs))
    cases = sorted(C.case.unique()); n_cases = len(cases)
    for g in GROUPS:
        assert sorted(C[C.group == g].seed.unique()) == [0, 1, 2], g
    rlist = [f'{g}_s{s}' for g in GROUPS for s in range(3)]
    run_index = {g: [rlist.index(f'{g}_s{s}') for s in range(3)] for g in GROUPS}

    low = L[L.cnr_stratum == 'low']
    ref = low[low.run == rlist[0]]
    n_low, n_low_cases = len(ref), ref.case.nunique()
    # per run x case counts of low-CNR lesions and detections
    K = np.zeros((len(rlist), n_cases)); N = np.zeros((len(rlist), n_cases))
    ci = {c: i for i, c in enumerate(cases)}
    for r, run in enumerate(rlist):
        d = low[low.run == run]
        for case, grp in d.groupby('case'):
            K[r, ci[case]] = grp.detected.sum(); N[r, ci[case]] = len(grp)

    point = {g: float(group_recall(K.sum(1), N.sum(1), run_index, g)) * 100 for g in GROUPS}

    # paired case bootstrap (same resampled cases for every run)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n_cases, size=(N_BOOT, n_cases))
    counts = np.stack([np.bincount(row, minlength=n_cases) for row in idx])      # [B, cases]
    Kb, Nb = K @ counts.T, N @ counts.T                                           # [runs, B]
    valid = Nb[0] > 0
    comps = []
    for m, b in COMPARISONS:
        diff = (group_recall(Kb, Nb, run_index, m) - group_recall(Kb, Nb, run_index, b))[valid] * 100
        p = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
        comps.append(dict(comparison=f'{m} vs {b}', role='primary' if b == 'A1' else 'secondary',
                          diff_pp=point[m] - point[b], ci95_low=float(np.percentile(diff, 2.5)),
                          ci95_high=float(np.percentile(diff, 97.5)), p_bootstrap=float(p)))
    for c, adj in zip(comps, holm(np.array([c['p_bootstrap'] for c in comps]))):
        c['p_holm'] = float(adj)
        c['significant'] = bool(adj < ALPHA and (c['ci95_low'] > 0 or c['ci95_high'] < 0))
        c['wording'] = ('提高' if c['diff_pp'] > 0 else '降低') if c['significant'] else '未观察到显著差异'

    # leave-one-case-out (primary comparison)
    m, b = COMPARISONS[0]
    loo = []
    for case in sorted(ref.case.unique()):
        keep = np.array([c != case for c in cases])
        d = (group_recall(K[:, keep].sum(1), N[:, keep].sum(1), run_index, m) -
             group_recall(K[:, keep].sum(1), N[:, keep].sum(1), run_index, b)) * 100
        loo.append(dict(case=case, n_low_cnr=int(N[0, ci[case]]),
                        diff_without_pp=float(d), contribution_pp=float(comps[0]['diff_pp'] - d)))
    loo = pd.DataFrame(loo).sort_values('contribution_pp', key=np.abs, ascending=False)
    loo.to_csv(out / 'loo_primary.csv', index=False)
    top = loo.iloc[0] if len(loo) else None

    # secondary endpoints (seed means; descriptive)
    sec = []
    for run in rlist + [r for r in DESCRIPTIVE if r in set(C.run)]:
        l, c = L[L.run == run], C[C.run == run]
        tp, fp = int(l.detected.sum()), int(c.fp.sum())
        sec.append(dict(run=run, group=_gs(run)[0], recall_low=100 * l[l.cnr_stratum == 'low'].detected.mean(),
                        recall_all=100 * l.detected.mean(), fp_per_scan=fp / len(c),
                        tumor_dice=100 * c[c.has_tumor].tumor_dice.mean(),
                        f1=100 * 2 * tp / max(len(l) + tp + fp, 1)))
    sec = pd.DataFrame(sec); sec.to_csv(out / 'secondary_per_run.csv', index=False)
    sec_g = sec.groupby('group', sort=False)[['recall_low', 'recall_all', 'fp_per_scan', 'tumor_dice', 'f1']].mean()
    sec_g.to_csv(out / 'secondary_per_group.csv')
    strata = (L[L.run.isin(rlist)].groupby(['group', 'seed', 'cnr_stratum', 'size_stratum']).detected
              .agg(['mean', 'size']).reset_index()
              .groupby(['group', 'cnr_stratum', 'size_stratum']).agg(recall=('mean', 'mean'), n=('size', 'first')))
    strata['recall'] *= 100; strata.to_csv(out / 'strata_recall.csv')

    # equal-FP comparison from the threshold sweep of the same split
    matched = None
    if a.sweep:
        S = pd.read_csv(a.sweep); S = S[S.rule.str.endswith('_pp')].copy(); S['recall_low'] *= 100
        budget = float(S[(S.group == METHOD) & (S.rule == 'argmax_pp')].fp_per_scan.mean())
        rows = []
        for g in GROUPS:
            vals = []
            for run, d in S[S.group == g].groupby('run'):
                ok = d[d.fp_per_scan <= budget + 1e-9]
                vals.append(ok.recall_low.max() if len(ok) else np.nan)
            rows.append(dict(group=g, budget_fp_per_scan=budget, best_recall_low=float(np.nanmean(vals)) if
                             not all(np.isnan(vals)) else float('nan'), seeds_reaching_budget=int(np.sum(~np.isnan(vals)))))
        matched = pd.DataFrame(rows); matched.to_csv(out / 'matched_fp.csv', index=False)

    res = dict(split=json.loads((tab / 'metrics_manifest.json').read_text())['split'], method=METHOD,
               n_cases=n_cases, n_low_cnr_lesions=int(n_low), n_cases_with_low_cnr=int(n_low_cases),
               bootstrap=dict(n=N_BOOT, seed=BOOT_SEED, valid=int(valid.sum())),
               primary_endpoint_pct=point, comparisons=comps,
               loo_top=None if top is None else dict(case=top.case, contribution_pp=float(top.contribution_pp),
                                                     diff_without_pp=float(top.diff_without_pp)))
    (out / 'protocol_results.json').write_text(json.dumps(res, indent=2, ensure_ascii=False) + '\n')

    md = [f"# 冻结协议结果（{res['split']}）", '',
          f"{n_cases} 例；低 CNR 病灶 {n_low} 个，分布在 {n_low_cases} 例中；4 mm 后处理；IoU > 0.1；每组 3 seed 取均值。", '',
          '## 主要终点：低 CNR 召回（%）', '', '| 组 | 召回 |', '|---|---:|']
    md += [f'| {g} | {point[g]:.2f} |' for g in GROUPS]
    md += ['', f'## 确认性比较（配对 bootstrap {N_BOOT} 次，种子 {BOOT_SEED}；有效 {int(valid.sum())} 次；Holm）', '',
           '| 比较 | 角色 | 差值（个百分点） | 95% CI | p | Holm p | 结论 |', '|---|---|---:|---|---:|---:|---|']
    md += [f"| {c['comparison']} | {c['role']} | {c['diff_pp']:+.2f} | [{c['ci95_low']:+.2f}, {c['ci95_high']:+.2f}] | "
           f"{c['p_bootstrap']:.4f} | {c['p_holm']:.4f} | {c['wording']} |" for c in comps]
    md += ['', '## 次要终点（描述）', '', sec_g.round(2).to_string()]
    if matched is not None:
        md += ['', f'## 同等 FP/scan（≤ {METHOD} argmax 点 {matched.budget_fp_per_scan.iloc[0]:.3f}）下的最高低 CNR 召回', '',
               matched.round(2).to_string(index=False)]
    md += ['', '## 留一病例敏感性（主要比较）', '',
           loo.round(2).to_string(index=False) if len(loo) else '（无低 CNR 病灶）']
    if top is not None:
        md += ['', f"贡献最大的病例 {top.case}：贡献 {top.contribution_pp:+.2f} 个百分点；去掉后差值 {top.diff_without_pp:+.2f}。"]
    (out / 'PROTOCOL_RESULTS.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('metrics')
    m.add_argument('--split', choices=SPLITS, required=True)
    m.add_argument('--results-root', default=str(DATA / 'nnUNet_results_v2'))
    m.add_argument('--out', required=True)
    m.add_argument('--jobs', type=int, default=1)
    s = sub.add_parser('stats')
    s.add_argument('--tables', required=True)
    s.add_argument('--sweep', default=None, help='threshold_sweep_per_run.csv of the same split')
    s.add_argument('--out', required=True)
    a = p.parse_args()
    cmd_metrics(a) if a.cmd == 'metrics' else cmd_stats(a)


if __name__ == '__main__':
    main()
