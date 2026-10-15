#!/usr/bin/env python3
"""Server-side: cut the illustration cube for Fig. 2 (method overview) from a
validation case (never a test case) and apply DCA, Eq. (4), in 3D.

    python extract_method_case.py --case LiTS_006 --out method_case.npz

Picks the tumour component of the case whose CNR (Eq. 2, ring = liver voxels
2-6 voxels outside the lesion) is closest to 1.25 among lesions with an
equivalent diameter of 12-35 mm, resamples a 1-mm isotropic cube of
--size mm around it, attenuates it to --target CNR with the paper's formula,
and stores HU volumes, labels, the attenuated volume and the A0 / A13b
(seed 0) validation predictions on the same grid.
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

DATA = Path('/root/autodl-tmp/nnUNet_data')
FOLD = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'


def read(p):
    img = sitk.ReadImage(str(p))
    return img, sitk.GetArrayFromImage(img)


def ring_of(m, liver, others):
    st = np.ones((3, 3, 3), bool)
    d6 = ndi.binary_dilation(m, st, iterations=6); d1 = ndi.binary_dilation(m, st, iterations=1)
    return d6 & ~d1 & liver & ~ndi.binary_dilation(others, st, iterations=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='LiTS_006')
    ap.add_argument('--target', type=float, default=0.6)
    ap.add_argument('--size', type=int, default=72)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    img, ct = read(DATA / f'nnUNet_raw/Dataset102_LiTS/imagesTr/{a.case}_0000.nii.gz')
    _, gt = read(DATA / f'nnUNet_raw/Dataset102_LiTS/labelsTr/{a.case}.nii.gz')
    # resample everything to 1 mm isotropic (z, y, x order in numpy)
    sp = np.array(img.GetSpacing())[::-1]
    zoom = sp / 1.0
    ct1 = ndi.zoom(ct.astype(np.float32), zoom, order=1)
    gt1 = ndi.zoom(gt, zoom, order=0)
    preds = {}
    for run in ('A0_s0', 'A13b_s0'):
        _, p = read(DATA / f'nnUNet_results_v2/{run}/{FOLD}/validation/{a.case}.nii.gz')
        preds[run] = ndi.zoom(p, zoom, order=0)
    tumor = gt1 == 2; liver = gt1 == 1
    lab, n = ndi.label(tumor, np.ones((3, 3, 3)))
    best = None
    for k in range(1, n + 1):
        m = lab == k
        vol = m.sum()
        diam = (6 * vol / np.pi) ** (1 / 3)
        if not 12 <= diam <= 35:
            continue
        r = ring_of(m, liver, tumor & ~m)
        if r.sum() < 50:
            continue
        cnr = abs(ct1[m].mean() - ct1[r].mean()) / ct1[r].std()
        print(f'lesion {k}: diam {diam:.1f} mm, CNR {cnr:.2f}')
        if best is None or abs(cnr - 1.25) < abs(best[1] - 1.25):
            best = (k, cnr, diam)
    k, cnr0, diam = best
    m = lab == k; r = ring_of(m, liver, tumor & ~m)
    mu, nu, s = ct1[m].mean(), ct1[r].mean(), ct1[r].std()
    alpha = min(1.0, a.target / cnr0)
    w = np.maximum(ndi.gaussian_filter(m.astype(np.float32), 1.0), m) * (gt1 > 0) * ~(tumor & ~m)
    att = ct1 - (1 - alpha) * (mu - nu) * w
    cnr1 = abs(att[m].mean() - att[r].mean()) / att[r].std()
    c = np.round(ndi.center_of_mass(m)).astype(int)
    h = a.size // 2
    sl = tuple(slice(max(0, ci - h), max(0, ci - h) + a.size) for ci in c)
    omega = ndi.binary_dilation(m, np.ones((3, 3, 3)), iterations=6)
    np.savez_compressed(a.out, ct=ct1[sl].astype(np.float32), att=att[sl].astype(np.float32), gt=gt1[sl].astype(np.uint8),
                        lesion=m[sl], ring=r[sl], omega=omega[sl], pred_a0=preds['A0_s0'][sl].astype(np.uint8),
                        pred_a13b=preds['A13b_s0'][sl].astype(np.uint8), cnr0=cnr0, cnr1=cnr1, alpha=alpha, diam=diam,
                        center=c - np.array([s.start for s in sl]), case=a.case)
    print(f'chosen lesion {k}: diam {diam:.1f} mm, CNR {cnr0:.2f} -> {cnr1:.2f} (alpha {alpha:.2f}); cube {[s for s in sl]}')


if __name__ == '__main__':
    main()
