#!/usr/bin/env python3
"""Checks for the A14 Pixel2Cancer training dataset (plan §11, 2026-10-14).

1. _PatchedArray: every slice of a patched case equals slicing a fully
   materialised patched copy (random boxes, including boxes that cross the
   case border and boxes that miss the variant).
2. P2CDataset with p=1 / p=0: variants are / are never used; class
   locations of label 2 gain the synthetic tumour voxels; the original arrays
   on disk are unchanged.
3. nnUNetDataLoader3D batches from P2CDataset: shapes, label set, and the
   fraction of batches whose seg contains synthetic-only tumour voxels.

    PYTHONPATH=<repo>/nnUNet python check_p2c_loader.py --synth <dir>
"""
import argparse
import json
import os

import numpy as np

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_phaser import P2CDataset, _PatchedArray
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

PRE = '/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset102_LiTS'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', required=True)
    a = ap.parse_args()
    split = json.load(open(os.path.join(PRE, 'splits_final.json')))[0]
    folder = os.path.join(PRE, 'nnUNetPlans_3d_fullres')
    base = nnUNetDataset(folder, sorted(split['train']), num_images_properties_loading_threshold=0)
    rng = np.random.default_rng(0)

    # 1. patched slicing == materialised copy
    ds1 = P2CDataset(base, a.synth, 1.0)
    case = sorted(ds1.variants)[0]
    v = np.load(ds1.variants[case][0])
    data, seg, _ = base.load_case(case)
    full = np.array(data); lo = v['lo']; sh = v['image'].shape
    full[(slice(None),) + tuple(slice(l, l + n) for l, n in zip(lo, sh))] = v['image'][None].astype(full.dtype)
    pa = _PatchedArray(data, lo, v['image'][None])
    for _ in range(200):
        lb = [int(rng.integers(-40, s)) for s in data.shape[1:]]
        sl = (slice(0, 1),) + tuple(slice(max(0, l), min(s, l + 128)) for l, s in zip(lb, data.shape[1:]))
        assert np.array_equal(pa[sl], full[sl])
    print('1. patched slicing matches a materialised copy (200 random boxes)')

    # 2. p = 1 / p = 0
    d1, s1, p1 = ds1.load_case(case)
    assert isinstance(d1, _PatchedArray) and len(p1['class_locations'][2]) > len(base[case]['properties']['class_locations'][2])
    ds0 = P2CDataset(base, a.synth, 0.0)
    d0, s0, p0 = ds0.load_case(case)
    assert not isinstance(d0, _PatchedArray)
    assert np.array_equal(np.asarray(base.load_case(case)[0][(slice(None),) + tuple(slice(l, l + n) for l, n in zip(lo, sh))]),
                          np.asarray(data[(slice(None),) + tuple(slice(l, l + n) for l, n in zip(lo, sh))]))
    print('2. p=1 uses variants with extra label-2 locations, p=0 never does; arrays on disk unchanged')

    # 2b. a case without real tumour: label-2 locations stay integer after merging
    no_tumour = [c for c in sorted(ds1.variants) if len(base[c]['properties']['class_locations'].get(2, [])) == 0]
    if no_tumour:
        _, _, pn = ds1.load_case(no_tumour[0])
        assert np.issubdtype(np.asarray(pn['class_locations'][2]).dtype, np.integer)
        print(f'2b. {no_tumour[0]} (no real tumour): merged locations are integer')

    # 3. loader batches (all cases, p = 1 so every sample uses a variant)
    np.random.seed(0)
    ds = P2CDataset(base, a.synth, 1.0)
    lm = PlansManager(json.load(open(os.path.join(PRE, 'nnUNetPlans.json')))).get_label_manager(
        json.load(open(os.path.join(PRE, 'dataset.json'))))
    dl = nnUNetDataLoader3D(ds, 2, (128, 128, 128), (128, 128, 128), lm, 0.66, None, None)
    n_syn = n_tum = 0
    for _ in range(150):
        b = dl.generate_train_batch()
        assert b['data'].shape == (2, 1, 128, 128, 128) and set(np.unique(b['seg'])) <= {-1, 0, 1, 2}
        for j, k in enumerate(b['keys']):
            n_tum += bool((b['seg'][j] == 2).any())
        n_syn += sum(len(p['class_locations'][2]) > len(base[k]['properties']['class_locations'][2])
                     for p, k in zip(b['properties'], b['keys']))
    print(f'3. 150 batches with p=1: {n_syn}/300 samples from variants, '
          f'{n_tum}/300 patches contain tumour')


if __name__ == '__main__':
    main()
