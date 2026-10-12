#!/usr/bin/env python3
"""Independence check: is any 3D-IRCADb scan also in our LiTS data?

    python -I experiments/external/check_lits_overlap.py --ircadb <dataset dir>/imagesTs \
        --lits <nnUNet_raw>/Dataset101_LiTS/imagesTr <nnUNet_raw>/Dataset101_LiTS/imagesTs \
        --out <dir> --jobs 16

Two independent criteria, reported for every IRCADb scan against every LiTS scan:
  1  header: identical voxel grid (sizes equal up to axis order) and spacing
     equal within 0.01 mm;
  2  content: orientation-free intensity fingerprint, the HU values at the
     0.5, 1, 2, ..., 99, 99.5 percentiles of the whole volume; distance =
     mean absolute difference (HU). A re-saved or re-oriented copy of the
     same scan has distance ~0.
A pair is flagged as a probable duplicate when the header matches, or when
the fingerprint distance is below 2 HU. Writes overlap_pairs.csv (the nearest
LiTS scans for each IRCADb scan) and overlap.json (verdict).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

Q = np.r_[0.5, np.arange(1, 100), 99.5]


def describe(path):
    img = sitk.ReadImage(str(path))
    a = sitk.GetArrayFromImage(img).astype(np.float32).ravel()
    return dict(path=str(path), name=Path(path).name.replace('_0000.nii.gz', ''), size=sorted(img.GetSize()),
                spacing=sorted(round(s, 3) for s in img.GetSpacing()), fp=np.percentile(a, Q))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ircadb', required=True)
    ap.add_argument('--lits', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    irc = sorted(Path(a.ircadb).glob('*.nii.gz'))
    lits = sorted(p for d in a.lits for p in Path(d).glob('*.nii.gz'))
    with ProcessPoolExecutor(a.jobs) as ex:
        I = list(ex.map(describe, irc)); L = list(ex.map(describe, lits))
    rows = []
    for i in I:
        cand = []
        for l in L:
            header = i['size'] == l['size'] and np.allclose(i['spacing'], l['spacing'], atol=0.01)
            dist = float(np.abs(i['fp'] - l['fp']).mean())
            cand.append(dict(ircadb=i['name'], lits=l['name'], header_match=header, fingerprint_mae_hu=dist,
                             ircadb_size=i['size'], lits_size=l['size'], ircadb_spacing=i['spacing'], lits_spacing=l['spacing']))
        cand.sort(key=lambda r: (not r['header_match'], r['fingerprint_mae_hu']))
        rows += cand[:3]
    T = pd.DataFrame(rows)
    T['flag'] = T.header_match | (T.fingerprint_mae_hu < 2.0)
    T.to_csv(out / 'overlap_pairs.csv', index=False)
    flagged = T[T.flag]
    verdict = dict(n_ircadb=len(I), n_lits=len(L), lits_dirs=a.lits, flagged_pairs=flagged[['ircadb', 'lits']].values.tolist(),
                   min_fingerprint_mae_hu=float(T.fingerprint_mae_hu.min()), independent=bool(flagged.empty))
    (out / 'overlap.json').write_text(json.dumps(verdict, indent=2) + '\n')
    print(T.groupby('ircadb').head(1).to_string(index=False))
    print('INDEPENDENT' if verdict['independent'] else f'OVERLAP SUSPECTED: {verdict["flagged_pairs"]}')


if __name__ == '__main__':
    main()
