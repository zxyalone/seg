#!/usr/bin/env python3
"""Offline Pixel2Cancer tumour synthesis for the A14 comparator (plan §11,
2026-10-14, registered before implementation).

Pixel2Cancer (Lai et al., MICCAI 2024; github.com/MrGiovanni/Pixel2Cancer,
Apache-2.0) grows tumours with a cellular automaton (CUDA) on whole volumes in
HU at 1 mm isotropic spacing, before intensity scaling, inside the MONAI
transform TumorSysthesis(prob=0.9). nnU-Net loads data in worker processes
without CUDA, so the variants are generated offline here with the original,
unmodified functions (generate_tumor, get_predefined_texture; liver
parameters Organ_HU [100, 160], steps 150, kernel 3^3, threshold 10, ten
predefined textures sigma_a in {3,6,9,12,15} x sigma_b in {4,7}):

  1. preprocessed case (1.0 x 0.78 x 0.78 mm, CTNormalization) -> HU with the
     dataset's mean/std (values were clipped to [p0.5, p99.5] = [-18, 196]);
  2. liver box (labels 1, 2; -1 = outside the nonzero crop counts as
     background) resampled to 1 mm isotropic, axes ordered (x, y, z) as after
     MONAI's Orientationd so that the per-slice blurs of map_to_CT_value act
     on the same planes as in the original;
  3. generate_tumor; the HU change and the new tumour voxels are resampled
     back to the preprocessed grid and only the changed region is stored
     (|change| > 2 HU or new tumour; generate_tumor also leaves +-1 HU
     rounding changes across the whole liver box, which are dropped).

The trainer (PHASER_P2C_DIR) swaps a training case for one of its variants
with probability PHASER_P2C_P (0.9, as in the original) and adds the
synthetic tumour voxels to the foreground-sampling locations.

    CUDA_VISIBLE_DEVICES=0 python p2c_synthesize.py --out <dir> --k 10 --part 0 --parts 3
    python p2c_synthesize.py --out <dir> --index      # after all parts
"""
import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

P2C = Path('/root/autodl-tmp/repos/Pixel2Cancer')
P2C_BUILD = P2C / 'Tumor_Synthesis/cellular/build/lib.linux-x86_64-cpython-310'
PRE = Path('/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset102_LiTS')
CFG = 'nnUNetPlans_3d_fullres'
SEED = 20261014
# TumorSysthesis (Tumor_Synthesis/tumor_synthesis.py), organ = liver
STEPS, KERNEL, ORGAN_STD, HU_LOW, HU_OUT, THRESHOLD = 150, (3, 3, 3), 0, 100, 160, 10
SIGMA_AS, SIGMA_BS, TEXTURE_SHAPE = (3, 6, 9, 12, 15), (4, 7), (500, 500, 500)


def zoom_to(a, shape, order):
    out = ndi.zoom(a, [s / t for s, t in zip(shape, a.shape)], order=order)
    pad = [(0, max(0, s - o)) for s, o in zip(shape, out.shape)]
    return np.pad(out, pad, mode='edge')[tuple(slice(0, s) for s in shape)]


def synth_one(case, k, texture, norm, spacing):
    data = np.load(PRE / CFG / f'{case}.npy', mmap_mode='r')[0]
    seg = np.load(PRE / CFG / f'{case}_seg.npy', mmap_mode='r')[0]
    liver = seg > 0
    zz, yy, xx = np.where(liver)
    lo = np.array([zz.min(), yy.min(), xx.min()]); hi = np.array([zz.max(), yy.max(), xx.max()]) + 1
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    hu = np.asarray(data[box], np.float32) * norm['std'] + norm['mean']
    lab = np.asarray(seg[box]).astype(np.int16); lab[lab < 0] = 0
    iso = tuple(int(round(n * s)) for n, s in zip(hu.shape, spacing))           # 1 mm grid (z, y, x)
    img1 = zoom_to(hu, iso, 1).transpose(2, 1, 0).copy()                          # (x, y, z)
    lab1 = zoom_to(lab, iso, 0).transpose(2, 1, 0).copy()
    if max(lab1.shape) >= TEXTURE_SHAPE[0] - 1:
        return None
    from Tumor_Synthesis.utils import generate_tumor
    args = argparse.Namespace(gpu=0, organ='liver')
    img_out, lab_out = generate_tumor(img1.copy(), lab1.copy(), texture, STEPS, KERNEL, ORGAN_STD, HU_LOW,
                                      HU_OUT, THRESHOLD, 'liver', args)
    new_t = ((lab_out == 2) & (lab1 != 2)).transpose(2, 1, 0)
    diff = (np.asarray(img_out, np.float32) - img1).transpose(2, 1, 0)
    if not new_t.any():
        return None
    diff_b = zoom_to(diff, hu.shape, 1)
    new_b = zoom_to(new_t.astype(np.uint8), hu.shape, 0).astype(bool) & (lab > 0)
    changed = (np.abs(diff_b) > 2.0) | new_b       # generate_tumor leaves +-1 HU rounding changes everywhere
    if not new_b.any():
        return None
    cz, cy, cx = np.where(changed)
    clo = np.array([cz.min(), cy.min(), cx.min()]); chi = np.array([cz.max(), cy.max(), cx.max()]) + 1
    cbox = tuple(slice(a, b) for a, b in zip(clo, chi))
    hu_new = hu[cbox] + np.where(changed[cbox], diff_b[cbox], 0)
    hu_new = np.clip(hu_new, norm['percentile_00_5'], norm['percentile_99_5'])
    z_new = ((hu_new - norm['mean']) / norm['std']).astype(np.float16)
    seg_new = np.asarray(seg[box][cbox]).astype(np.int8).copy()
    seg_new[new_b[cbox]] = 2
    locs = np.argwhere(new_b) + lo
    if len(locs) > 20000:
        locs = locs[np.random.choice(len(locs), 20000, replace=False)]
    locs = np.concatenate([np.zeros((len(locs), 1), np.int64), locs], 1)            # (c, z, y, x)
    vox_ml = float(np.prod(spacing)) / 1000
    comps = ndi.label(new_b, np.ones((3, 3, 3), bool))[1]
    stats = dict(case=case, k=k, new_tumour_ml=float(new_b.sum() * vox_ml), n_components=int(comps),
                 hu_tumour=float(hu_new[new_b[cbox]].mean()),
                 hu_liver=float(hu[(lab == 1) & ~new_b].mean()),
                 box_lo=(lo + clo).tolist(), box_shape=list(z_new.shape))
    return dict(lo=lo + clo, image=z_new, seg=seg_new, tumour_locs=locs), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--cases', nargs='*')
    ap.add_argument('--index', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    split = json.load(open(PRE / 'splits_final.json'))[0]
    if a.index:
        stats = [json.loads(l) for f in sorted(out.glob('stats_part*.jsonl')) for l in open(f)]
        variants = {}
        for s in stats:
            variants.setdefault(s['case'], []).append(f"{s['case']}_{s['k']:02d}.npz")
        assert not set(variants) & set(split['val']), 'variants of validation cases'
        assert set(variants) <= set(split['train'])
        json.dump(dict(source='Pixel2Cancer (MrGiovanni/Pixel2Cancer, unmodified Tumor_Synthesis)', seed=SEED, k=a.k, p_original=0.9,
                       variants={c: sorted(v) for c, v in sorted(variants.items())},
                       n_variants=len(stats), cases=len(variants)), open(out / 'index.json', 'w'), indent=1)
        print(f'{len(stats)} variants for {len(variants)} cases')
        return
    sys.path.insert(0, str(P2C_BUILD)); sys.path.insert(0, str(P2C))
    import torch
    from Tumor_Synthesis.utils import get_predefined_texture
    plans = json.load(open(PRE / 'nnUNetPlans.json'))
    norm = plans['foreground_intensity_properties_per_channel']['0']
    spacing = plans['configurations']['3d_fullres']['spacing']
    cases = a.cases or sorted(split['train'])
    cases = cases[a.part::a.parts]
    random.seed(SEED + a.part); np.random.seed(SEED + a.part); torch.manual_seed(SEED + a.part)
    t0 = time.time()
    textures = [get_predefined_texture(TEXTURE_SHAPE, sa, sb) for sa in SIGMA_AS for sb in SIGMA_BS]
    print(f'part {a.part}: {len(textures)} textures in {time.time() - t0:.0f} s; {len(cases)} cases', flush=True)
    with open(out / f'stats_part{a.part}.jsonl', 'a') as log:
        for case in cases:
            for k in range(a.k):
                f = out / f'{case}_{k:02d}.npz'
                if f.exists():
                    continue
                res = None
                for _ in range(3):
                    # generate_tumor draws from the global random / np.random streams seeded above
                    res = synth_one(case, k, textures[random.randrange(len(textures))], norm, spacing)
                    if res is not None:
                        break
                if res is None:
                    print(f'{case} {k}: no tumour after 3 tries', flush=True)
                    continue
                v, s = res
                np.savez_compressed(f, **v)
                log.write(json.dumps(s) + '\n'); log.flush()
                print(f"{case} {k}: {s['new_tumour_ml']:.1f} ml, {s['n_components']} comp, "
                      f"HU {s['hu_tumour']:.0f} vs liver {s['hu_liver']:.0f}, {time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
