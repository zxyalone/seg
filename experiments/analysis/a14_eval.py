#!/usr/bin/env python3
"""A14 (Pixel2Cancer) descriptive evaluation (plan §11, 2026-10-14).

metrics  fixed 4-mm lesion/case tables of A14_s{0,1,2} with the frozen
         protocol's code (test_protocol.cmd_metrics) for 'val' and 'test'.
stats    joins them with the archived tables of A0, A1, A11, A13a, A13b
         (results/test_protocol*/tables) and reports per-group secondary
         endpoints, CNR x size strata, and paired case-bootstrap 95% CIs
         (10 000, seed 20261012) of A13b - A14 and A14 - A0 / A1 for the
         low-CNR recall, overall recall, FP/scan and tumour Dice.
         Descriptive only: A14 was added after the test set was unlocked.

    python a14_eval.py metrics --split test --out <dir>/test_tables --jobs 3
    python a14_eval.py stats --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import test_protocol as TP

REPO = Path(__file__).resolve().parents[2]
ARCHIVED = {'val': REPO / 'results/test_protocol_dryrun_val/tables', 'test': REPO / 'results/test_protocol/tables'}
GROUPS = ('A0', 'A1', 'A11', 'A13a', 'A13b', 'A14')
COMPARISONS = (('A13b', 'A14'), ('A14', 'A0'), ('A14', 'A1'))
N_BOOT, SEED = 10_000, 20261012


def gs(run):
    g, s = run.rsplit('_s', 1)
    return g, int(s)


def summarise(L, C):
    rows = []
    for run in C.run.unique():
        l, c = L[L.run == run], C[C.run == run]
        tp, fp = int(l.detected.sum()), int(c.fp.sum())
        rows.append(dict(run=run, recall_low=100 * l[l.cnr_stratum == 'low'].detected.mean(),
                         recall_all=100 * l.detected.mean(), fp_per_scan=fp / len(c),
                         tumor_dice=100 * c[c.has_tumor].tumor_dice.mean(),
                         f1=100 * 2 * tp / max(len(l) + tp + fp, 1)))
    S = pd.DataFrame(rows)
    S['group'] = S.run.map(lambda r: gs(r)[0])
    return S


def boot(L, C, a, b, cases):
    """Paired case bootstrap of seed-mean differences a - b."""
    ci = {c: i for i, c in enumerate(cases)}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(cases), size=(N_BOOT, len(cases)))
    counts = np.stack([np.bincount(r, minlength=len(cases)) for r in idx])        # [B, cases]
    out = {}
    for name, frame, num, den in (('recall_low', L[L.cnr_stratum == 'low'], 'detected', None),
                                  ('recall_all', L, 'detected', None)):
        vals = {}
        for g in (a, b):
            runs = sorted(frame[frame.group == g].run.unique())
            K = np.zeros((len(runs), len(cases))); N = np.zeros_like(K)
            for r, run in enumerate(runs):
                d = frame[frame.run == run]
                for case, grp in d.groupby('case'):
                    K[r, ci[case]] = grp[num].sum(); N[r, ci[case]] = len(grp)
            vals[g] = ((K @ counts.T) / (N @ counts.T)).mean(0) * 100, (K.sum(1) / N.sum(1)).mean() * 100
        d = vals[a][0] - vals[b][0]
        out[name] = (vals[a][1] - vals[b][1], np.percentile(d, 2.5), np.percentile(d, 97.5))
    for name, col, filt in (('fp_per_scan', 'fp', None), ('tumor_dice', 'tumor_dice', 'has_tumor')):
        vals = {}
        for g in (a, b):
            c = C[C.group == g]
            if filt:
                c = c[c[filt]]
            m = c.groupby('case')[col].mean().reindex(cases)
            vals[g] = m
        diff = (vals[a] - vals[b]).values
        ok = ~np.isnan(diff)
        dd = diff[ok]
        bi = rng.integers(0, len(dd), size=(N_BOOT, len(dd)))
        scale = 100 if name == 'tumor_dice' else 1
        out[name] = (dd.mean() * scale, np.percentile(dd[bi].mean(1), 2.5) * scale, np.percentile(dd[bi].mean(1), 97.5) * scale)
    return out


def cmd_metrics(a):
    TP.GROUPS = ('A14',); TP.DESCRIPTIVE = ()
    ns = argparse.Namespace(split=a.split, out=a.out, jobs=a.jobs, results_root=a.results_root)
    TP.cmd_metrics(ns)


def cmd_stats(a):
    out = Path(a.out); md = ['# A14（Pixel2Cancer）描述性对比', '']
    result = {}
    for split in ('val', 'test'):
        L = pd.concat([pd.read_csv(ARCHIVED[split] / 'lesions_pp4mm.csv'), pd.read_csv(out / f'{split}_tables/lesions_pp4mm.csv')])
        C = pd.concat([pd.read_csv(ARCHIVED[split] / 'cases_pp4mm.csv'), pd.read_csv(out / f'{split}_tables/cases_pp4mm.csv')])
        L['group'] = L.run.map(lambda r: gs(r)[0]); C['group'] = C.run.map(lambda r: gs(r)[0])
        L, C = L[L.group.isin(GROUPS)], C[C.group.isin(GROUPS)]
        S = summarise(L, C)
        S.to_csv(out / f'{split}_per_run.csv', index=False)
        G = S.groupby('group')[['recall_low', 'recall_all', 'fp_per_scan', 'tumor_dice', 'f1']].mean().reindex(GROUPS)
        G.to_csv(out / f'{split}_per_group.csv')
        strata = (L.groupby(['group', 'cnr_stratum', 'size_stratum']).detected.mean().mul(100)
                    .unstack(['cnr_stratum', 'size_stratum']).reindex(GROUPS))
        strata.to_csv(out / f'{split}_strata.csv')
        cases = sorted(C.case.unique())
        comps = {f'{x} - {y}': boot(L, C, x, y, cases) for x, y in COMPARISONS}
        result[split] = {k: {m: [float(t) for t in v] for m, v in d.items()} for k, d in comps.items()}
        md += [f'## {split}', '', G.round(2).to_string(), '', '| 比较 | 低 CNR 召回 | 全部召回 | FP/scan | 肿瘤 Dice |', '|---|---|---|---|---|']
        for k, d in comps.items():
            md.append(f'| {k} | ' + ' | '.join(f'{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]' for v in
                                               (d['recall_low'], d['recall_all'], d['fp_per_scan'], d['tumor_dice'])) + ' |')
        md += ['', '高 CNR < 10 mm 召回（%）：' + ', '.join(f'{g} {strata.loc[g, ("high", "<10")]:.1f}' for g in GROUPS), '']
    (out / 'a14_stats.json').write_text(json.dumps(result, indent=1) + '\n')
    (out / 'A14_RESULTS.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('metrics', 'stats'))
    ap.add_argument('--split', choices=('val', 'test'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--results-root', default='/root/autodl-tmp/nnUNet_data/nnUNet_results_v2')
    a = ap.parse_args()
    (cmd_metrics if a.cmd == 'metrics' else cmd_stats)(a)


if __name__ == '__main__':
    main()
