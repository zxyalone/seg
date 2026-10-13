#!/usr/bin/env python3
"""MSD Task08 as an external liver-tumour test set (plan §11, 2026-10-13,
registered before inference).

    python -I prepare_msd08.py stage --raw <Task08_HepaticVessel> --out <dataset dir>
    python -I prepare_msd08.py reference --raw <Task08_HepaticVessel> --out <dataset dir> \
        --a0 <A0_s0 external_pred> <A0_s1 ...> <A0_s2 ...>

stage      imagesTs/hepaticvessel_XXX_0000.nii.gz (symlinks to the 303 labelled
           CTs; macOS '._' files are skipped) and cases.txt.
reference  labelsTs/hepaticvessel_XXX.nii.gz for evaluation. Task08 labels
           vessel (1) and tumour (2) but not the liver, which the CNR ring
           needs. Fixed rule: liver = voxels predicted as liver or tumour by at
           least 2 of the 3 baseline (A0) models; tumour = Task08 tumour label
           (always label 2, also outside the predicted liver); everything else
           background. The same reference is used for every compared model.
           Writes reference_manifest.json (liver volume per case, fraction of
           tumour voxels inside the reference liver).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def cases_of(raw):
    return sorted(p.name.replace('.nii.gz', '') for p in (Path(raw) / 'labelsTr').glob('hepaticvessel_*.nii.gz'))


def stage(a):
    raw, out = Path(a.raw), Path(a.out)
    (out / 'imagesTs').mkdir(parents=True, exist_ok=True)
    cases = cases_of(raw)
    for c in cases:
        src = (raw / 'imagesTr' / f'{c}.nii.gz').resolve()
        dst = out / 'imagesTs' / f'{c}_0000.nii.gz'
        if not dst.exists():
            dst.symlink_to(src)
    (out / 'cases.txt').write_text('\n'.join(cases) + '\n')
    print(f'{len(cases)} cases staged in {out}')


def one(args):
    import SimpleITK as sitk
    case, raw, out, a0 = args
    gt_img = sitk.ReadImage(str(Path(raw) / 'labelsTr' / f'{case}.nii.gz'))
    gt = sitk.GetArrayFromImage(gt_img)
    votes = np.zeros(gt.shape, np.uint8)
    for d in a0:
        p = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(d) / f'{case}.nii.gz')))
        assert p.shape == gt.shape, (case, p.shape, gt.shape)
        votes += (p > 0)
    liver = votes >= 2
    tumour = gt == 2
    ref = np.zeros(gt.shape, np.uint8)
    ref[liver] = 1
    ref[tumour] = 2
    img = sitk.GetImageFromArray(ref); img.CopyInformation(gt_img)
    sitk.WriteImage(img, str(Path(out) / 'labelsTs' / f'{case}.nii.gz'), useCompression=True)
    vox = float(np.prod(gt_img.GetSpacing())) / 1000.0
    return dict(case=case, liver_ml=float(liver.sum() * vox), tumour_ml=float(tumour.sum() * vox),
                tumour_in_liver_frac=float((tumour & liver).sum() / max(tumour.sum(), 1)))


def reference(a):
    raw, out = Path(a.raw), Path(a.out)
    (out / 'labelsTs').mkdir(parents=True, exist_ok=True)
    cases = (out / 'cases.txt').read_text().split()
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(one, [(c, str(raw), str(out), a.a0) for c in cases]))
    frac = np.array([r['tumour_in_liver_frac'] for r in rows])
    man = dict(rule='liver = predicted liver or tumour in >= 2 of 3 A0 models; tumour = Task08 label 2',
               a0_dirs=a.a0, n_cases=len(rows),
               tumour_in_liver_frac=dict(median=float(np.median(frac)), min=float(frac.min()),
                                         cases_below_0_5=int((frac < 0.5).sum())),
               per_case=rows)
    (out / 'reference_manifest.json').write_text(json.dumps(man, indent=1) + '\n')
    print(json.dumps({k: v for k, v in man.items() if k != 'per_case'}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stage'); s.add_argument('--raw', required=True); s.add_argument('--out', required=True)
    r = sub.add_parser('reference'); r.add_argument('--raw', required=True); r.add_argument('--out', required=True)
    r.add_argument('--a0', nargs=3, required=True); r.add_argument('--jobs', type=int, default=16)
    a = ap.parse_args()
    (stage if a.cmd == 'stage' else reference)(a)


if __name__ == '__main__':
    main()
