#!/usr/bin/env python3
"""Server-side crops for the additional figures (selection rules registered in
plan §11, 2026-10-14, before plotting).

  multi    test set: 6 low-CNR lesions (diameter >= 10 mm, not in Fig. 5, at
           most one per case) drawn at random (seed 20261014); CT, reference
           and seed-0 predictions of A0, A1, A12, A14, A13b (4-mm post-processed)
  msd08    MSD Task08: the 3 largest low-CNR lesions (distinct cases) detected by
           all A13b seeds and missed by all A1 seeds, plus one A13b_s0-only
           false positive drawn (seed 20261014) from the false-positive montage
  p2c      one Pixel2Cancer training variant drawn (seed 20261014) from the 830

Images are reoriented to LPS (rows anterior -> posterior, columns patient
right -> left). Each crop is the axial slice through the largest cross-section
of the target (64 mm, or 1.3 x its extent if larger).

    python extract_figs_extra.py --out figs_extra.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'experiments/analysis'))
from missed_lesion_probability import E, _drop_small  # noqa: E402

DATA = Path('/root/autodl-tmp/nnUNet_data')
RES = DATA / 'nnUNet_results_v2'
FOLD = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
LITS = DATA / 'nnUNet_raw/Dataset101_LiTS'
MSD = Path('/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08')
P2C = DATA / 'p2c_synth/Dataset102_fold0'
PRE = DATA / 'nnUNet_preprocessed/Dataset102_LiTS'
SEED = 20261014
FIG5 = {('LiTS_122', 17), ('LiTS_117', 74), ('LiTS_109', 26), ('LiTS_097', 10)}
MULTI_RUNS = ('A0_s0', 'A1_s0', 'A12_s0', 'A14_s0', 'A13b_s0')


def lps(img):
    return sitk.DICOMOrient(img, 'LPS')


def arr(img):
    return sitk.GetArrayFromImage(img)


def label_lps(raw_mask_arr, ref_img):
    im = sitk.GetImageFromArray(raw_mask_arr.astype(np.int32)); im.CopyInformation(ref_img)
    return arr(lps(im))


def crop_box(target, spacing_yx, size_mm=64):
    """Axial slice of the largest cross-section; square crop of max(64 mm, 1.3 x
    the in-plane extent of the target) so that large lesions stay in view."""
    z = int(np.argmax(target.sum((1, 2))))
    yx = np.argwhere(target[z]); cy, cx = yx.mean(0)
    extent = (yx.max(0) - yx.min(0) + 1).max() * spacing_yx
    h = int(round(max(size_mm, 1.3 * extent) / 2 / spacing_yx))
    return z, int(round(cy)) - h, int(round(cx)) - h, h


def cut(v, z, y0, x0, h, fill=0):
    pad = np.pad(v[z], h + 1, constant_values=fill)
    return pad[y0 + h + 1:y0 + 3 * h + 1, x0 + h + 1:x0 + 3 * h + 1]


def lesion_crops(prefix, out, img_path, gt_path, lesion, pred_paths, extra=None):
    img = sitk.ReadImage(str(img_path)); gt_img = sitk.ReadImage(str(gt_path))
    gt_raw = arr(gt_img)
    sp = img.GetSpacing()
    lab, _ = E.components(gt_raw == 2, 26)
    if lesion is not None:
        target = label_lps(lab == lesion, gt_img)
    else:
        target = label_lps(extra, gt_img)
    ct = arr(lps(img)).astype(np.float32)
    gt = arr(lps(gt_img))
    z, y0, x0, h = crop_box(target > 0, sp[0])
    out[f'{prefix}_ct'] = cut(ct, z, y0, x0, h, -1000).astype(np.float32)
    out[f'{prefix}_gt'] = cut(gt == 2, z, y0, x0, h)
    out[f'{prefix}_target'] = cut(target > 0, z, y0, x0, h)
    for name, p in pred_paths.items():
        pr = arr(sitk.ReadImage(str(p)))
        pp = _drop_small(pr == 2, float(np.prod(sp)), 4.0)
        out[f'{prefix}_{name}'] = cut(label_lps(pp, gt_img) > 0, z, y0, x0, h)
    out[f'{prefix}_px'] = np.float32(sp[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figs_extra.npz')
    a = ap.parse_args()
    out, meta = {}, {}
    rng = np.random.default_rng(SEED)

    # multi-method test lesions
    T = pd.concat([pd.read_csv(REPO / 'results/test_protocol/tables/lesions_pp4mm.csv'),
                   pd.read_csv(REPO / 'results/a14_pixel2cancer/test_tables/lesions_pp4mm.csv')])
    ref = T[(T.run == 'A0_s0') & (T.cnr_stratum == 'low') & (T.diameter_mm >= 10)]
    ref = ref[[(c, l) not in FIG5 for c, l in zip(ref.case, ref.lesion)]].sort_values(['case', 'lesion'])
    order = rng.permutation(len(ref)); picked, seen = [], set()
    for i in order:
        r = ref.iloc[i]
        if r.case in seen:
            continue
        seen.add(r.case); picked.append((r.case, int(r.lesion)))
        if len(picked) == 6:
            break
    meta['multi'] = []
    for k, (case, lesion) in enumerate(picked):
        rows = T[(T.case == case) & (T.lesion == lesion)].set_index('run')
        lesion_crops(f'multi{k}', out, LITS / 'imagesTr' / f'{case}_0000.nii.gz', LITS / 'labelsTr' / f'{case}.nii.gz', lesion,
                     {r: RES / r / FOLD / 'test_pred' / f'{case}.nii.gz' for r in MULTI_RUNS})
        meta['multi'].append(dict(case=case, lesion=lesion, cnr=float(rows.cnr.iloc[0]), diameter=float(rows.diameter_mm.iloc[0]),
                                  detected={r: bool(rows.loc[r, 'detected']) for r in MULTI_RUNS}))
        print('multi', case, lesion, flush=True)

    # MSD08 gains + one false positive
    M = pd.read_csv(REPO / 'results/external_msd08/tables/lesions_pp4mm.csv')
    M['g'] = M.run.str.rsplit('_s', n=1).str[0]
    low = M[M.cnr_stratum == 'low']
    P = low[low.g.isin(['A1', 'A13b'])].pivot_table(index=['case', 'lesion', 'diameter_mm', 'cnr'], columns='g',
                                                   values='detected', aggfunc='sum').reset_index()
    gains = P[(P.A13b == 3) & (P.A1 == 0)].sort_values('diameter_mm', ascending=False)
    chosen, seen = [], set()
    for _, r in gains.iterrows():
        if r.case not in seen:
            seen.add(r.case); chosen.append(r)
        if len(chosen) == 3:
            break
    meta['msd08'] = []
    runs = {'A1': RES / 'A1_s0' / FOLD / 'external_msd08_pred', 'A13b': RES / 'A13b_s0' / FOLD / 'external_msd08_pred'}
    for k, r in enumerate(chosen):
        lesion_crops(f'msd{k}', out, MSD / 'imagesTs' / f'{r.case}_0000.nii.gz', MSD / 'labelsTs' / f'{r.case}.nii.gz',
                     int(r.lesion), {g: p / f'{r.case}.nii.gz' for g, p in runs.items()})
        meta['msd08'].append(dict(kind='gain', case=r.case, lesion=int(r.lesion), cnr=float(r.cnr), diameter=float(r.diameter_mm)))
        print('msd gain', r.case, int(r.lesion), flush=True)
    F = pd.read_csv(REPO / 'results/external_msd08/fp_analysis/fp_montage_cases.csv')
    f = F.iloc[int(rng.integers(len(F)))]
    img = sitk.ReadImage(str(MSD / 'imagesTs' / f'{f.case}_0000.nii.gz'))
    pr = arr(sitk.ReadImage(str(runs['A13b'] / f'{f.case}.nii.gz')))
    pp = _drop_small(pr == 2, float(np.prod(img.GetSpacing())), 4.0)
    pl, _ = E.components(pp, 26)
    lesion_crops('msd3', out, MSD / 'imagesTs' / f'{f.case}_0000.nii.gz', MSD / 'labelsTs' / f'{f.case}.nii.gz', None,
                 {g: p / f'{f.case}.nii.gz' for g, p in runs.items()}, extra=pl == int(f.comp))
    meta['msd08'].append(dict(kind='fp', case=f.case, comp=int(f.comp), diameter=float(f.diameter_mm), contrast=float(f.contrast)))
    print('msd fp', f.case, int(f.comp), flush=True)

    # Pixel2Cancer variant
    index = json.load(open(P2C / 'index.json'))
    files = sorted(fn for fs in index['variants'].values() for fn in fs)
    fn = files[int(rng.integers(len(files)))]
    v = np.load(P2C / fn); case = fn[:8]
    plans = json.load(open(PRE / 'nnUNetPlans.json'))['foreground_intensity_properties_per_channel']['0']
    data = np.load(PRE / 'nnUNetPlans_3d_fullres' / f'{case}.npy', mmap_mode='r')[0]
    seg = np.load(PRE / 'nnUNetPlans_3d_fullres' / f'{case}_seg.npy', mmap_mode='r')[0]
    props = pd.read_pickle(PRE / 'nnUNetPlans_3d_fullres' / f'{case}.pkl')
    lo = v['lo']; box = tuple(slice(l, l + n) for l, n in zip(lo, v['seg'].shape))
    syn = np.array(data); syn[box] = v['image'].astype(np.float32)
    syn_seg = np.array(seg); syn_seg[box] = v['seg']
    new = (syn_seg == 2) & (np.asarray(seg) != 2)
    direction = props['sitk_stuff']['direction']

    def orient(x):
        im = sitk.GetImageFromArray(x.astype(np.float32)); im.SetDirection(direction)
        return arr(lps(im))
    hu0, hu1 = orient(np.asarray(data) * plans['std'] + plans['mean']), orient(syn * plans['std'] + plans['mean'])
    new_o, liver_o = orient(new) > 0.5, orient(np.asarray(seg) > 0) > 0.5
    z = int(np.argmax(new_o.sum((1, 2))))
    yy, xx = np.where(liver_o[z])
    out['p2c_orig'] = hu0[z, yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    out['p2c_syn'] = hu1[z, yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    out['p2c_new'] = new_o[z, yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    meta['p2c'] = dict(file=fn, case=case)
    print('p2c', fn, flush=True)

    out['meta'] = np.array(json.dumps(meta))
    np.savez_compressed(a.out, **out)


if __name__ == '__main__':
    main()
