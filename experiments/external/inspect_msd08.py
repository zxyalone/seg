#!/usr/bin/env python3
"""MSD Task08 (hepatic vessel and tumour, Memorial Sloan Kettering) as an
external liver-tumour test set: safe extraction and label inspection.

    python -I experiments/external/inspect_msd08.py extract --tar <Task08_HepaticVessel.tar> --dest <empty dir>
    python -I experiments/external/inspect_msd08.py inspect --root <dest>/Task08_HepaticVessel --out <dir> --jobs 16

The archive is untrusted data: every member is checked to stay inside --dest
and only regular files and directories are extracted; nothing in it is run.
'inspect' writes per-case geometry and tumour statistics (label 2; label 1 is
vessel) so that the evaluation protocol can be fixed before any inference:
how many tumour components each case has (complete vs single-lesion
annotation) and their sizes.
"""
import argparse
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def extract(a):
    dest = Path(a.dest).resolve(); dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.tar) as t:
        members = t.getmembers()
        for m in members:
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest) + '/') and target != dest:
                raise SystemExit(f'unsafe member {m.name}')
            if not (m.isfile() or m.isdir()):
                raise SystemExit(f'non-regular member {m.name}')
        t.extractall(dest, members=members)
    print(f'{len(members)} members extracted to {dest}')


def one(path):
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    img = sitk.ReadImage(str(path))
    lab = sitk.GetArrayFromImage(img)
    sp = img.GetSpacing()
    tumor = lab == 2
    comp, n = ndi.label(tumor, structure=np.ones((3, 3, 3)))
    vox_ml = float(np.prod(sp)) / 1000.0
    sizes = np.bincount(comp.ravel())[1:] if n else np.array([])
    diam = [float((6 * s * vox_ml * 1000 / np.pi) ** (1 / 3)) for s in sizes]  # equivalent diameter, mm
    return dict(case=path.name.replace('.nii.gz', ''), size=list(img.GetSize()), spacing=[round(s, 3) for s in sp],
                labels=sorted(int(v) for v in np.unique(lab)), n_tumor=int(n), tumor_ml=float(tumor.sum() * vox_ml),
                tumor_diam_mm=sorted(diam, reverse=True), vessel_ml=float((lab == 1).sum() * vox_ml))


def inspect(a):
    import pandas as pd
    root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    labels = sorted((root / 'labelsTr').glob('hepaticvessel_*.nii.gz'))
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(one, labels))
    T = pd.DataFrame(rows); T.to_json(out / 'cases.json', orient='records', indent=1)
    ds = json.loads((root / 'dataset.json').read_text())
    n_img_tr = len(list((root / 'imagesTr').glob('hepaticvessel_*.nii.gz')))
    n_img_ts = len(list((root / 'imagesTs').glob('hepaticvessel_*.nii.gz')))
    alld = [d for r in rows for d in r['tumor_diam_mm']]
    summary = dict(labels_meta=ds.get('labels'), modality=ds.get('modality'), n_labelled=len(rows),
                   n_images_tr=n_img_tr, n_images_ts=n_img_ts,
                   cases_without_tumour=int((T.n_tumor == 0).sum()),
                   tumour_components_per_case=T.n_tumor.value_counts().sort_index().to_dict(),
                   n_tumour_components=int(T.n_tumor.sum()),
                   tumour_diam_mm_quartiles=np.percentile(alld, [0, 25, 50, 75, 100]).round(1).tolist() if alld else None,
                   slice_thickness_mm=T.spacing.map(lambda s: s[2]).describe().round(2).to_dict(),
                   inplane_mm=T.spacing.map(lambda s: s[0]).describe().round(3).to_dict())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract'); e.add_argument('--tar', required=True); e.add_argument('--dest', required=True)
    i = sub.add_parser('inspect'); i.add_argument('--root', required=True); i.add_argument('--out', required=True)
    i.add_argument('--jobs', type=int, default=16)
    a = ap.parse_args()
    (extract if a.cmd == 'extract' else inspect)(a)


if __name__ == '__main__':
    main()
