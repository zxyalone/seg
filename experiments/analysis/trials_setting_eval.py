#!/usr/bin/env python3
"""TriALS-setting experiment (plan §11, 2026-10-14; descriptive, added after the
test set was unlocked).

T0 (nnU-Net) and T13b (+ DCA + CFC) are trained like the TriALS reference
models: from scratch, stock nnU-Net optimiser (SGD, lr 1e-2, poly), 100 epochs,
same 83/21/27 split; seeds 0-2. Validation uses nnU-Net's final validation
(checkpoint_final); the test set is predicted with checkpoint_best (as TriALS,
primary for this block) and checkpoint_final (secondary), no TTA.

metrics  fixed 4-mm lesion/case tables with the frozen protocol's code
         (test_protocol.cmd_metrics) for 'val', 'test_best', 'test_final'.
stats    per-run / per-group endpoints, CNR x size strata, and paired case-
         bootstrap 95% CIs (10 000, seed 20261012) of T13b - T0, T13b - TriALS
         nnU-Net 3D and T0 - TriALS nnU-Net 3D; the TriALS reference rows
         (results/reference_baselines, checkpoint_best) and the archived
         200-epoch pretrained A0 / A13b rows are listed alongside.

    python trials_setting_eval.py metrics --split test_best --out <dir>/test_best_tables --jobs 3
    python trials_setting_eval.py stats --out <dir>

--groups BASE,METHOD selects the pair (default T0,T13b); the 200-epoch runs
(plan §11, 2026-10-15) use --groups T0L,T13bL.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import a14_eval as A
import test_protocol as TP

REPO = Path(__file__).resolve().parents[2]
GROUPS = ('T0', 'T13b')   # (baseline, method); --groups overrides
REF = REPO / 'results/reference_baselines'
REF_RUNS = ('nnUNet3D', 'nnUNet2D', 'SegResNet', 'UMambaBot', 'MedNeXt-L', 'SAMed-H', 'SwinUNETR', 'SAMed-B')
ARCHIVED = {'val': REPO / 'results/test_protocol_dryrun_val/tables', 'test': REPO / 'results/test_protocol/tables'}
TP.SPLITS['test_best'] = dict(TP.SPLITS['test'], pred_subdir='test_pred_best', eval_subdir='eval_test_best')
TP.SPLITS['test_final'] = dict(TP.SPLITS['test'])


def comparisons():
    base, method = GROUPS
    return ((method, base), (method, 'nnUNet3D'), (base, 'nnUNet3D'))


def group_of(run):
    return run if run in REF_RUNS else A.gs(run)[0]


def cmd_metrics(a):
    TP.GROUPS = GROUPS; TP.DESCRIPTIVE = ()
    TP.cmd_metrics(argparse.Namespace(split=a.split, out=a.out, jobs=a.jobs, results_root=a.results_root))


def cmd_stats(a):
    out = Path(a.out); md = [f'# TriALS 训练设置下的比较（{"/".join(GROUPS)}：从头训练、标准 nnU-Net 优化器；描述性）', '']
    result = {}
    for split in ('val', 'test_best', 'test_final'):
        L = pd.read_csv(out / f'{split}_tables/lesions_pp4mm.csv'); C = pd.read_csv(out / f'{split}_tables/cases_pp4mm.csv')
        arch = ARCHIVED['val' if split == 'val' else 'test']
        La, Ca = pd.read_csv(arch / 'lesions_pp4mm.csv'), pd.read_csv(arch / 'cases_pp4mm.csv')
        keep = lambda d: d[d.run.map(lambda r: A.gs(r)[0] in ('A0', 'A13b'))]
        parts_L, parts_C = [L, keep(La)], [C, keep(Ca)]
        if split != 'val':
            parts_L.append(pd.read_csv(REF / 'lesions_pp4mm.csv')); parts_C.append(pd.read_csv(REF / 'cases_pp4mm.csv'))
        L, C = pd.concat(parts_L, ignore_index=True), pd.concat(parts_C, ignore_index=True)
        L['group'] = L.run.map(group_of); C['group'] = C.run.map(group_of)
        rows = []
        for run in C.run.unique():
            l, c = L[L.run == run], C[C.run == run]
            tp, fp = int(l.detected.sum()), int(c.fp.sum())
            rows.append(dict(run=run, group=group_of(run), recall_low=100 * l[l.cnr_stratum == 'low'].detected.mean(),
                             recall_all=100 * l.detected.mean(), fp_per_scan=fp / len(c),
                             tumor_dice=100 * c[c.has_tumor].tumor_dice.mean(), f1=100 * 2 * tp / max(len(l) + tp + fp, 1)))
        S = pd.DataFrame(rows)
        S.to_csv(out / f'{split}_per_run.csv', index=False)
        order = [g for g in (*GROUPS, 'A0', 'A13b', *REF_RUNS) if g in set(S.group)]
        G = S.groupby('group')[['recall_low', 'recall_all', 'fp_per_scan', 'tumor_dice', 'f1']].mean().reindex(order)
        G['n_seeds'] = S.groupby('group').run.nunique().reindex(order)
        G.to_csv(out / f'{split}_per_group.csv')
        strata = (L.groupby(['group', 'cnr_stratum', 'size_stratum']).detected.mean().mul(100)
                    .unstack(['cnr_stratum', 'size_stratum']).reindex(order))
        strata.to_csv(out / f'{split}_strata.csv')
        cases = sorted(C.case.unique())
        comps = {f'{x} - {y}': A.boot(L, C, x, y, cases) for x, y in comparisons() if {x, y} <= set(S.group)}
        result[split] = {k: {m: [float(t) for t in v] for m, v in d.items()} for k, d in comps.items()}
        md += [f'## {split}', '', G.round(2).to_string(), '', '| 比较 | 低 CNR 召回 | 全部召回 | FP/scan | 肿瘤 Dice |', '|---|---|---|---|---|']
        for k, d in comps.items():
            md.append(f'| {k} | ' + ' | '.join(f'{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]' for v in
                                               (d['recall_low'], d['recall_all'], d['fp_per_scan'], d['tumor_dice'])) + ' |')
        md += ['', '高 CNR < 10 mm 召回（%）：' + ', '.join(f'{g} {strata.loc[g, ("high", "<10")]:.1f}' for g in order), '']
    (out / 'trials_setting_stats.json').write_text(json.dumps(result, indent=1) + '\n')
    (out / 'TRIALS_SETTING_RESULTS.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('metrics', 'stats'))
    ap.add_argument('--split', choices=('val', 'test_best', 'test_final'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--results-root', default='/root/autodl-tmp/nnUNet_data/nnUNet_results_v2')
    ap.add_argument('--groups', default='T0,T13b', help='baseline,method')
    a = ap.parse_args()
    global GROUPS
    GROUPS = tuple(a.groups.split(','))
    assert len(GROUPS) == 2
    (cmd_metrics if a.cmd == 'metrics' else cmd_stats)(a)


if __name__ == '__main__':
    main()
