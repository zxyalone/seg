#!/usr/bin/env python3
"""Post-hoc descriptive analysis of the false positives on MSD Task08 (plan
§11, 2026-10-14, registered before running; not part of the registered
external validation, no tests).

components  every unmatched predicted tumour component after the fixed 4-mm
            post-processing (IoU > 0.1, 26-connectivity), per run and case,
            with a mutually exclusive category in fixed priority:
              tumour_fragment  overlaps Task08 tumour voxels
              outside_liver    < 50 % of voxels inside the reference liver
              vessel           >= 30 % of voxels in the Task08 vessel label
                               dilated by one voxel
              liver_other      everything else
            plus diameter (equal-volume sphere) and native-grid contrast
            against a 2-6 mm reference-liver ring (tumour and other
            predictions excluded, >= 10 ring voxels; positive = hypodense).
            FP counts are asserted against tables/cases_pp4mm.csv.
summary     FP/scan per category (3-seed mean per group), A13b - A1 and
            A13b - A0 with paired case bootstrap 95 % CI (descriptive), strata
            by diameter and by annotated components per case.
montage     16 random (seed 20261014) A13b_s0 liver_other FPs that no A1 seed
            predicts, axial slice through the largest cross-section.

    python msd08_fp_analysis.py components --out <dir> --jobs 32
    python msd08_fp_analysis.py summary --out <dir>
    python msd08_fp_analysis.py montage --out <dir>
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
EXT = Path('/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08')
RAW = Path('/root/autodl-tmp/ext_data/msd08/raw/Task08_HepaticVessel')
TABLES = REPO / 'results/external_msd08/tables'
GROUPS = ('A0', 'A1', 'A13b')
RUNS = [f'{g}_s{s}' for g in GROUPS for s in range(3)]
CATS = ('tumour_fragment', 'outside_liver', 'vessel', 'liver_other')
N_BOOT, BOOT_SEED, MONTAGE_SEED = 10_000, 20261012, 20261014


def load_case(case):
    img, ct = E.read(EXT / 'imagesTs' / f'{case}_0000.nii.gz')
    _, ref = E.read(EXT / 'labelsTs' / f'{case}.nii.gz')
    _, raw = E.read(RAW / 'labelsTr' / f'{case}.nii.gz')
    return img, ct.astype(np.float32), ref, raw


def pp_components(run, case, img):
    _, pred = E.read(RESULTS_ROOT / run / TRAINER / 'external_msd08_pred' / f'{case}.nii.gz')
    pp = _drop_small(pred == 2, float(np.prod(img.GetSpacing())), 4.0)
    return E.components(pp, 26)


def one(args):
    run, case = args
    img, ct, ref, raw = load_case(case)
    sp = np.array(img.GetSpacing()[::-1])                      # z, y, x
    gt = ref == 2
    gl, ng = E.components(gt, 26)
    pl, npred = pp_components(run, case, img)
    match, _ = E.match_lesions(gl, ng, pl, npred, .1)
    matched = {v for v in match.values() if v is not None}
    liver, vessel = ref > 0, ndi.binary_dilation(raw == 1)
    margin = np.ceil(8 / sp).astype(int)
    rows = []
    for k, sl in enumerate(ndi.find_objects(pl), start=1):
        if sl is None or k in matched:
            continue
        sl = tuple(slice(max(s.start - m, 0), min(s.stop + m, n)) for s, m, n in zip(sl, margin, pl.shape))
        m = pl[sl] == k
        vox = int(m.sum())
        f_liver = float(liver[sl][m].mean())
        f_vessel = float(vessel[sl][m].mean())
        touches = bool(gt[sl][m].any())
        cat = ('tumour_fragment' if touches else 'outside_liver' if f_liver < .5
               else 'vessel' if f_vessel >= .3 else 'liver_other')
        d = ndi.distance_transform_edt(~m, sampling=sp)
        ring = (d >= 2) & (d <= 6) & liver[sl] & ~gt[sl] & (pl[sl] == 0)
        c = ct[sl]
        contrast = (float((c[ring].mean() - c[m].mean()) / (c[ring].std() + 1e-8))
                    if ring.sum() >= 10 else np.nan)
        z_area = m.sum((1, 2))
        zc = int(np.argmax(z_area)) + sl[0].start
        yx = np.argwhere(m[zc - sl[0].start]).mean(0) + [sl[1].start, sl[2].start]
        rows.append(dict(run=run, case=case, comp=k, category=cat, n_gt=ng,
                         diameter_mm=float((6 * vox * np.prod(sp) / np.pi) ** (1 / 3)),
                         frac_liver=f_liver, frac_vessel=f_vessel, mean_hu=float(c[m].mean()),
                         contrast=contrast, z=zc, y=int(yx[0]), x=int(yx[1])))
    return run, case, npred - len(matched), ng, rows


def cmd_components(a):
    from concurrent.futures import ProcessPoolExecutor
    cases = (EXT / 'cases.txt').read_text().split()
    jobs = [(r, c) for r in RUNS for c in cases]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, jobs, chunksize=4))
    C = pd.read_csv(TABLES / 'cases_pp4mm.csv').set_index(['run', 'case'])
    for run, case, nfp, _, rows in res:
        assert nfp == len(rows) == C.loc[(run, case), 'fp'], (run, case, nfp, len(rows))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for *_, rows in res for r in rows]).to_csv(out / 'fp_components.csv', index=False)
    pd.DataFrame([dict(case=c, n_gt=ng) for r, c, _, ng, _ in res if r == RUNS[0]]).to_csv(
        out / 'case_n_gt.csv', index=False)
    print(f'{sum(len(x[-1]) for x in res)} FP components; counts match cases_pp4mm.csv')


def cmd_summary(a):
    out = Path(a.out)
    F = pd.read_csv(out / 'fp_components.csv')
    G = pd.read_csv(out / 'case_n_gt.csv')
    cases = sorted(G.case); ci = {c: i for i, c in enumerate(cases)}
    F['group'] = F.run.str.rsplit('_s', n=1).str[0]
    F['size'] = np.where(F.diameter_mm < 10, '<10', '>=10')

    def counts(mask):
        """[run, case] FP counts for the rows selected by mask."""
        M = np.zeros((len(RUNS), len(cases)))
        for (run, case), n in F[mask].groupby(['run', 'case']).size().items():
            M[RUNS.index(run), ci[case]] = n
        return M

    gi = {g: [RUNS.index(f'{g}_s{s}') for s in range(3)] for g in GROUPS}

    def describe(name, mask, case_sel=None):
        M = counts(mask)
        sel = np.ones(len(cases), bool) if case_sel is None else case_sel
        Ms = M[:, sel]
        ix = np.random.default_rng(BOOT_SEED).integers(0, sel.sum(), size=(N_BOOT, sel.sum()))
        per = {g: Ms[gi[g]].mean(0) for g in GROUPS}                # seed mean, per case
        row = dict(subset=name, n_cases=int(sel.sum()))
        for g in GROUPS:
            row[f'{g}_fp_per_scan'] = float(per[g].mean())
        for b in ('A1', 'A0'):
            d = per['A13b'] - per[b]
            boot = d[ix].mean(1)
            row[f'A13b-{b}'] = float(d.mean())
            row[f'A13b-{b}_ci'] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        return row

    allm = np.ones(len(F), bool)
    rows = [describe('all', allm)]
    rows += [describe(c, (F.category == c).values) for c in CATS]
    rows += [describe(f'diameter {s} mm', (F['size'] == s).values) for s in ('<10', '>=10')]
    rows += [describe(f'liver_other, diameter {s} mm', ((F.category == 'liver_other') & (F['size'] == s)).values)
             for s in ('<10', '>=10')]
    one_gt = G.set_index('case').loc[cases, 'n_gt'].values == 1
    rows += [describe('cases with 1 annotated component', allm, one_gt),
             describe('cases with >= 2 annotated components', allm, ~one_gt)]
    S = pd.DataFrame(rows)
    S.to_csv(out / 'fp_summary.csv', index=False)
    shape = (F.groupby(['group', 'category'])
               .agg(n=('comp', 'size'), diameter_median=('diameter_mm', 'median'),
                    contrast_median=('contrast', 'median'), mean_hu_median=('mean_hu', 'median'))
               .reset_index())
    shape['n_per_scan'] = shape.n / (3 * len(cases))
    shape.to_csv(out / 'fp_shape.csv', index=False)
    (out / 'fp_summary.json').write_text(json.dumps(dict(n_cases=len(cases), rows=rows), indent=1) + '\n')

    fmt = lambda v: f'{v:+.2f}'
    ci_ = lambda v: f'[{v[0]:+.2f}, {v[1]:+.2f}]'
    md = ['# MSD Task08 假阳性的事后描述性分析', '',
          f'{len(cases)} 例；4 mm 后处理后未匹配（IoU > 0.1）的预测连通域；3 seed 均值；差值的 95% CI 为配对病例 bootstrap（10 000 次，种子 {BOOT_SEED}），只描述。', '',
          '| 子集 | 病例 | A0 | A1 | A13b | A13b − A1 | 95% CI | A13b − A0 | 95% CI |',
          '|---|---:|---:|---:|---:|---:|---|---:|---|']
    for r in rows:
        md.append(f"| {r['subset']} | {r['n_cases']} | {r['A0_fp_per_scan']:.2f} | {r['A1_fp_per_scan']:.2f} | "
                  f"{r['A13b_fp_per_scan']:.2f} | {fmt(r['A13b-A1'])} | {ci_(r['A13b-A1_ci'])} | "
                  f"{fmt(r['A13b-A0'])} | {ci_(r['A13b-A0_ci'])} |")
    md += ['', '## 各类假阳性的形态（全部 seed 汇总）', '', shape.round(2).to_string(index=False), '']
    (out / 'FP_ANALYSIS.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


def cmd_montage(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(a.out)
    F = pd.read_csv(out / 'fp_components.csv')
    cand = F[(F.run == 'A13b_s0') & (F.category == 'liver_other')]
    keep = []
    for case, grp in cand.groupby('case'):
        img, ct, ref, raw = load_case(case)
        a1 = np.zeros(ref.shape, bool)
        for s in range(3):
            lab, _ = pp_components(f'A1_s{s}', case, img); a1 |= lab > 0
        lab, _ = pp_components('A13b_s0', case, img)
        for _, r in grp.iterrows():
            if not a1[lab == r.comp].any():
                keep.append(r)
    K = pd.DataFrame(keep)
    rng = np.random.default_rng(MONTAGE_SEED)
    pick = K.iloc[np.sort(rng.choice(len(K), size=min(16, len(K)), replace=False))]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12.6))
    for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
        img, ct, ref, raw = load_case(r.case)
        lab, _ = pp_components('A13b_s0', r.case, img)
        sp = img.GetSpacing()[::-1]
        h = int(round(40 / sp[1]))
        y0, x0 = max(r.y - h, 0), max(r.x - h, 0)
        win = lambda arr: arr[r.z, y0:r.y + h, x0:r.x + h]
        ax.imshow(np.clip((win(ct) + 120) / 360, 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.contour(win(lab) == r.comp, levels=[.5], colors='#d62728', linewidths=1.2)
        if (win(ref) == 2).any():
            ax.contour(win(ref) == 2, levels=[.5], colors='#2ca02c', linewidths=1.0)
        if (win(raw) == 1).any():
            ax.contour(win(raw) == 1, levels=[.5], colors='#17becf', linewidths=.6)
        ax.set_title(f'{r.case.replace("hepaticvessel_", "case ")}  d={r.diameter_mm:.1f} mm  '
                     f'c={r.contrast:.2f}', fontsize=9)
        ax.axis('off')
    for ax in axes.ravel()[len(pick):]:
        ax.axis('off')
    fig.suptitle('A13b_s0 liver_other FPs not predicted by any A1 seed (random 16 of '
                 f'{len(K)}; red FP, green Task08 tumour, cyan vessel; 80 mm crop, W360/L60)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'fp_montage_a13b_only.png', dpi=150)
    pick.to_csv(out / 'fp_montage_cases.csv', index=False)
    print(f'{len(K)} A13b_s0-only liver_other FPs; montage of {len(pick)} written')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('components', 'summary', 'montage'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=32)
    a = ap.parse_args()
    dict(components=cmd_components, summary=cmd_summary, montage=cmd_montage)[a.cmd](a)


if __name__ == '__main__':
    main()
