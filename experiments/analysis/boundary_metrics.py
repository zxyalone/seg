#!/usr/bin/env python3
"""Boundary and volume metrics (plan §11, 2026-10-14, registered before
computing; descriptive, outside the frozen protocol).

Per case: Dice, VOE, RVD, ASSD, HD95 and MSSD (maximum symmetric surface
distance) for the tumour (prediction after the fixed 4-mm post-processing)
and, on the LiTS test split only, for the whole liver (predicted liver +
tumour vs reference liver + tumour, no post-processing). Surfaces are
6-neighbourhood erosion borders; distances in mm. Conventions of
evaluate.py: a tumour-bearing case without predicted tumour gets the image
diagonal as surface distance (counted); tumour-free cases are excluded from
the tumour metrics.

    python boundary_metrics.py cases --split test --out <dir> --jobs 32
    python boundary_metrics.py summary --split test --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from missed_lesion_probability import E, _drop_small

REPO = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path('/root/autodl-tmp/nnUNet_data/nnUNet_results_v2')
TRAINER = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
SPLITS = {
    'test': dict(groups=('A0', 'A1', 'A11', 'A13a', 'A13b'), extra=('A12_s0',), pred='test_pred',
                 gt=Path('/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset101_LiTS/labelsTr'),
                 cases=REPO / 'experiments/eval/test_cases.txt', liver=True),
    'external_msd08': dict(groups=('A0', 'A1', 'A13b'), extra=(), pred='external_msd08_pred',
                           gt=Path('/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08/labelsTs'),
                           cases=Path('/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08/cases.txt'),
                           liver=False),
}
METRICS = ('dice', 'voe', 'rvd', 'assd', 'hd95', 'mssd')
N_BOOT, BOOT_SEED = 10_000, 20261012


def _surface(m):
    return m & ~ndi.binary_erosion(m, structure=ndi.generate_binary_structure(3, 1), border_value=0)


def region_metrics(p, g, sp, diag):
    tp, np_, ng = int((p & g).sum()), int(p.sum()), int(g.sum())
    out = dict(dice=100 * 2 * tp / (np_ + ng), voe=100 * (1 - tp / (np_ + ng - tp)),
               rvd=100 * (np_ - ng) / ng)
    if np_ == 0:
        return dict(out, assd=diag, hd95=diag, mssd=diag, pred_empty=True)
    idx = np.argwhere(p | g)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(np.maximum(idx.min(0) - 2, 0),
                                                       np.minimum(idx.max(0) + 3, p.shape)))
    sp_, sg = _surface(p[sl]), _surface(g[sl])
    d_pg = ndi.distance_transform_edt(~sg, sampling=sp)[sp_]
    d_gp = ndi.distance_transform_edt(~sp_, sampling=sp)[sg]
    d = np.concatenate([d_pg, d_gp])
    return dict(out, assd=float(d.mean()), hd95=float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))),
                mssd=float(d.max()), pred_empty=False)


def runs_of(split):
    c = SPLITS[split]
    return [f'{g}_s{s}' for g in c['groups'] for s in range(3)] + list(c['extra'])


def one(args):
    split, run, case = args
    c = SPLITS[split]
    gimg, gt = E.read(c['gt'] / f'{case}.nii.gz')
    _, pred = E.read(RESULTS_ROOT / run / TRAINER / c['pred'] / f'{case}.nii.gz')
    assert pred.shape == gt.shape, (run, case)
    sp = tuple(float(s) for s in gimg.GetSpacing()[::-1])
    diag = float(np.linalg.norm(np.array(gt.shape) * np.array(sp)))
    rows = []
    if (gt == 2).any():
        pt = _drop_small(pred == 2, float(np.prod(sp)), 4.0)
        rows.append(dict(run=run, case=case, region='tumour', **region_metrics(pt, gt == 2, sp, diag)))
    if c['liver']:
        rows.append(dict(run=run, case=case, region='liver', **region_metrics(pred > 0, gt > 0, sp, diag)))
    return rows


def cmd_cases(a):
    from concurrent.futures import ProcessPoolExecutor
    cases = Path(SPLITS[a.split]['cases']).read_text().split()
    jobs = [(a.split, r, c) for r in runs_of(a.split) for c in cases]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = [r for rows in ex.map(one, jobs, chunksize=2) for r in rows]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_csv(out / 'boundary_cases.csv', index=False)
    print(f'{len(res)} rows for {len(runs_of(a.split))} runs x {len(cases)} cases')


def cmd_summary(a):
    out = Path(a.out)
    D = pd.read_csv(out / 'boundary_cases.csv')
    groups = SPLITS[a.split]['groups']
    D['group'] = D.run.str.rsplit('_s', n=1).str[0]
    rows, comps = [], []
    for region, R in D.groupby('region'):
        per_run = R.groupby(['group', 'run'])[list(METRICS)].mean().reset_index()
        for g, G in per_run.groupby('group'):
            rows.append(dict(region=region, group=g, n_seeds=len(G), n_cases=int(R[R.group == g].case.nunique()),
                             pred_empty_cases=int(R[R.group == g].pred_empty.sum()),
                             **{m: float(G[m].mean()) for m in METRICS}))
        # paired case bootstrap of the seed-mean per case
        cm = R[R.group.isin(groups)].groupby(['group', 'case'])[list(METRICS)].mean()
        cases = sorted(R.case.unique())
        ix = np.random.default_rng(BOOT_SEED).integers(0, len(cases), size=(N_BOOT, len(cases)))
        for b in ('A1', 'A0'):
            if 'A13b' not in groups:
                continue
            for m in METRICS:
                d = (cm.loc['A13b'].loc[cases, m] - cm.loc[b].loc[cases, m]).values
                boot = d[ix].mean(1)
                comps.append(dict(region=region, comparison=f'A13b - {b}', metric=m, diff=float(d.mean()),
                                  ci95_low=float(np.percentile(boot, 2.5)), ci95_high=float(np.percentile(boot, 97.5))))
    S, C = pd.DataFrame(rows), pd.DataFrame(comps)
    S.to_csv(out / 'boundary_summary.csv', index=False)
    C.to_csv(out / 'boundary_comparisons.csv', index=False)
    (out / 'boundary_summary.json').write_text(json.dumps(dict(split=a.split, groups=rows, comparisons=comps),
                                                          indent=1) + '\n')
    md = [f'# 边界与体积指标（{a.split}；描述性）', '',
          '肿瘤为 4 mm 后处理后的预测；Dice/VOE/RVD 为 %，ASSD/HD95/MSSD 为 mm；每个运行取病例均值，组内种子取均值。', '',
          S.round(2).to_string(index=False), '', '## A13b 与对照的差（配对病例 bootstrap 95% CI，只描述）', '',
          C.round(2).to_string(index=False), '']
    (out / 'BOUNDARY.md').write_text('\n'.join(md))
    print('\n'.join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('cases', 'summary'))
    ap.add_argument('--split', choices=tuple(SPLITS), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=32)
    a = ap.parse_args()
    (cmd_cases if a.cmd == 'cases' else cmd_summary)(a)


if __name__ == '__main__':
    main()
