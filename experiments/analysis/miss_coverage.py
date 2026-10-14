#!/usr/bin/env python3
"""Coverage of protocol-missed lesions on the test set (plan §11, 2026-10-14,
registered before computing; descriptive, outside the frozen protocol).

For every reference lesion that the frozen protocol counts as missed (4-mm
post-processing, IoU > 0.1, one-to-one matching), the fraction of its voxels
covered by the post-processed tumour prediction. Covered >= 50 % is split into
'merged' (the covering component also overlaps another reference lesion) and
'oversized' (it does not). Detection itself is unchanged.

    python miss_coverage.py --out <dir> --jobs 27
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from missed_lesion_probability import E, _drop_small

REPO = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path('/root/autodl-tmp/nnUNet_data/nnUNet_results_v2')
TRAINER = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
GT = Path('/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset101_LiTS/labelsTr')
RUNS = [f'{g}_s{s}' for g in ('A0', 'A1', 'A13b') for s in range(3)]


def one(args):
    run, case = args
    img, gt = E.read(GT / f'{case}.nii.gz')
    _, pred = E.read(RESULTS_ROOT / run / TRAINER / 'test_pred' / f'{case}.nii.gz')
    gl, ng = E.components(gt == 2, 26)
    pp = _drop_small(pred == 2, float(np.prod(img.GetSpacing())), 4.0)
    pl, npr = E.components(pp, 26)
    match, _ = E.match_lesions(gl, ng, pl, npr, .1)
    rows = []
    for i in range(1, ng + 1):
        m = gl == i
        cov = float(pp[m].mean())
        kind = 'detected' if match[i] is not None else 'missed_uncovered'
        if match[i] is None and cov >= .5:
            ids, cnt = np.unique(pl[m], return_counts=True)
            k = ids[ids > 0][np.argmax(cnt[ids > 0])]
            other = np.unique(gl[pl == k])
            kind = 'missed_merged' if len(set(other) - {0, i}) else 'missed_oversized'
        rows.append(dict(run=run, case=case, lesion=i, coverage=cov, kind=kind))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=27)
    a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    cases = (REPO / 'experiments/eval/test_cases.txt').read_text().split()
    with ProcessPoolExecutor(a.jobs) as ex:
        R = pd.DataFrame([r for rows in ex.map(one, [(r, c) for r in RUNS for c in cases]) for r in rows])
    T = pd.read_csv(REPO / 'results/test_protocol/tables/lesions_pp4mm.csv')
    R = R.merge(T, on=['run', 'case', 'lesion'], validate='one_to_one')
    assert ((R.kind == 'detected') == R.detected).all(), 'detection differs from the protocol tables'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / 'miss_coverage_lesions.csv', index=False)
    R['group'] = R.run.str.rsplit('_s', n=1).str[0]
    S = (R.groupby(['group', 'cnr_stratum', 'size_stratum', 'kind']).size().unstack(fill_value=0) / 3).reset_index()
    S.to_csv(out / 'miss_coverage_strata.csv', index=False)
    A = (R.groupby(['group', 'kind']).size().unstack(fill_value=0) / 3)
    md = ['# 测试集漏检病灶的覆盖率（3 种子均值，病灶数）', '', A.round(2).to_string(), '',
          '## 按 CNR × 大小', '', S.round(2).to_string(index=False), '']
    (out / 'MISS_COVERAGE.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
