#!/usr/bin/env python3
"""Architecture-transfer evaluation: SegResNet S0 vs S13b_bf16 (plan §11,
2026-10-13; descriptive, outside the confirmatory tests).

Same fixed-4-mm pipeline as the frozen test protocol (lowcnr_pilot_metrics.metrics,
26-connectivity, IoU > 0.1, frozen CNR tertiles), on validation and test.
S13b_bf16 - S0 differences use the protocol's paired case bootstrap (seed-mean
of pooled recall, 10000 resamples, seed 20261012) and are reported with 95% CIs
only; no test, no wording rule.

    python segresnet_transfer_eval.py metrics --split test --out <dir> --jobs 4
    python segresnet_transfer_eval.py stats --tables <dir> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from test_protocol import SPLITS, N_BOOT, BOOT_SEED, _one, _gs

DATA = Path('/root/autodl-tmp/nnUNet_data')
BASE, METHOD = 'S0', 'S13b_bf16'
SEEDS = (0, 1)
RUNS = [f'{g}_s{s}' for g in (BASE, METHOD) for s in SEEDS]


def cmd_metrics(a):
    split = SPLITS[a.split]
    ns = argparse.Namespace(results_root=a.results_root, gt_dir=str(split['gt_dir']),
                            pred_subdir=split['pred_subdir'], eval_subdir=split['eval_subdir'])
    cases = Path(split['cases_file']).read_text().split()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        parts = list(ex.map(_one, RUNS, [ns] * len(RUNS), [cases] * len(RUNS)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.concat([c for c, _ in parts]).to_csv(out / 'cases_pp4mm.csv', index=False)
    pd.concat([l for _, l in parts]).to_csv(out / 'lesions_pp4mm.csv', index=False)
    (out / 'metrics_manifest.json').write_text(json.dumps(dict(
        split=a.split, runs=RUNS, n_cases=len(cases), min_diameter_mm=4.0, connectivity=26, iou=0.1), indent=2) + '\n')
    print(f'tables written for {len(RUNS)} runs x {len(cases)} cases')


def per_run(L, C):
    rows = []
    for run in RUNS:
        l, c = L[L.run == run], C[C.run == run]
        tp, fp = int(l.detected.sum()), int(c.fp.sum())
        low = l[l.cnr_stratum == 'low']
        rows.append(dict(run=run, group=_gs(run)[0], seed=_gs(run)[1],
                         recall_low=100 * low.detected.mean(), n_low=len(low), det_low=int(low.detected.sum()),
                         recall_all=100 * l.detected.mean(), fp_per_scan=fp / len(c),
                         tumor_dice=100 * c[c.has_tumor].tumor_dice.mean(),
                         f1=100 * 2 * tp / max(len(l) + tp + fp, 1)))
    return pd.DataFrame(rows)


def cmd_stats(a):
    tab, out = Path(a.tables), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = pd.read_csv(tab / 'lesions_pp4mm.csv'); C = pd.read_csv(tab / 'cases_pp4mm.csv')
    R = per_run(L, C); R.round(4).to_csv(out / 'per_run.csv', index=False)
    G = R.groupby('group', sort=False).mean(numeric_only=True).drop(columns='seed')
    G.round(4).to_csv(out / 'per_group.csv')

    cases = sorted(C.case.unique()); ci = {c: i for i, c in enumerate(cases)}
    def counts(frame, value):
        M = np.zeros((len(RUNS), len(cases)))
        for r, run in enumerate(RUNS):
            for case, g in frame[frame.run == run].groupby('case'):
                M[r, ci[case]] = value(g)
        return M
    low = L[L.cnr_stratum == 'low']
    K, N = counts(low, lambda g: g.detected.sum()), counts(low, len)
    KA, NA = counts(L, lambda g: g.detected.sum()), counts(L, len)
    FP = counts(C, lambda g: g.fp.sum())
    bi = {g: [RUNS.index(f'{g}_s{s}') for s in SEEDS] for g in (BASE, METHOD)}

    def recall(k, n, g):
        return np.mean(k[bi[g]].sum(-1) / n[bi[g]].sum(-1), axis=0) * 100
    def fps(f, g):
        return np.mean(f[bi[g]].sum(-1), axis=0) / f.shape[-1]

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(cases), size=(N_BOOT, len(cases)))
    res = {}
    for name, k, n in (('recall_low', K, N), ('recall_all', KA, NA)):
        point = recall(k, n, METHOD) - recall(k, n, BASE)
        kb, nb = k[:, idx], n[:, idx]          # [runs, N_BOOT, cases]
        diff = recall(kb, nb, METHOD) - recall(kb, nb, BASE)
        res[name] = dict(point=float(point), ci95=[float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))])
    fpb = FP[:, idx]
    dfp = fps(fpb, METHOD) - fps(fpb, BASE)
    res['fp_per_scan'] = dict(point=float(fps(FP, METHOD) - fps(FP, BASE)),
                              ci95=[float(np.percentile(dfp, 2.5)), float(np.percentile(dfp, 97.5))])
    res['per_seed_recall_low_diff'] = {f's{s}': float(R.set_index('run').loc[f'{METHOD}_s{s}', 'recall_low']
                                                       - R.set_index('run').loc[f'{BASE}_s{s}', 'recall_low']) for s in SEEDS}
    res['n_low'] = int(N[0].sum()); res['n_cases'] = len(cases); res['n_boot'] = N_BOOT; res['boot_seed'] = BOOT_SEED
    (out / 'transfer_stats.json').write_text(json.dumps(res, indent=2) + '\n')
    strata = (L.assign(group=L.run.map(lambda r: _gs(r)[0]))
              .groupby(['group', 'cnr_stratum', 'size_stratum']).detected.agg(['mean', 'size']).reset_index())
    strata.to_csv(out / 'strata.csv', index=False)
    print(R.round(2).to_string(index=False)); print(G.round(2).to_string()); print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('metrics')
    m.add_argument('--split', choices=SPLITS, required=True)
    m.add_argument('--results-root', default=str(DATA / 'nnUNet_results_v2'))
    m.add_argument('--out', required=True)
    m.add_argument('--jobs', type=int, default=4)
    s = sub.add_parser('stats')
    s.add_argument('--tables', required=True)
    s.add_argument('--out', required=True)
    a = ap.parse_args()
    (cmd_metrics if a.cmd == 'metrics' else cmd_stats)(a)


if __name__ == '__main__':
    main()
