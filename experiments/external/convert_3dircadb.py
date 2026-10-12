#!/usr/bin/env python3
"""Convert 3D-IRCADb-01 (20 patient zips from IRCAD) to nnU-Net-style NIfTI.

    python -I experiments/external/convert_3dircadb.py --zips <download dir> \
        --work <empty scratch dir> --out <dataset dir>

Input: 3Dircadb1.<i>.zip, each holding PATIENT_DICOM.zip and MASKS_DICOM.zip
(one sub-folder of DICOM slices per structure). The archives are treated as
untrusted data: members are extracted with a path check and nothing in them
is executed.

Labels (fixed before any inference; plan §11): 1 = liver ("liver" mask),
2 = tumor (union of every mask folder whose name starts with "livertumor").
Cysts ("liverkyst", "livercyst") and all other structures are background
unless they lie inside the liver mask, where they stay liver. Patients
without a livertumor folder are kept as tumor-free cases.

Writes <out>/imagesTs/IRCADb_<ii>_0000.nii.gz (HU, int16),
<out>/labelsTs/IRCADb_<ii>.nii.gz (uint8), <out>/cases.txt and manifest.json.
"""
import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

TUMOR = re.compile(r'^livertumor', re.I)


def safe_extract(zip_path: Path, dest: Path, keep=None):
    """Extract members (optionally only those for which keep(name) is true)
    after checking that none escapes dest."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if keep is None or keep(m.filename)]
        for info in members:
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest)):
                raise RuntimeError(f'unsafe member {info.filename} in {zip_path}')
        z.extractall(dest, members=members)


def mask_names(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        return sorted({n.split('/')[1] for n in z.namelist() if n.count('/') >= 2 and n.split('/')[1]})


def slice_files(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and re.fullmatch(r'image_\d+', p.name)]
    return sorted(files, key=lambda p: int(p.name.split('_')[1]))


def read_series(files):
    r = sitk.ImageSeriesReader()
    r.SetFileNames([str(f) for f in files])
    return r.Execute()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zips', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    zips, work, out = Path(a.zips), Path(a.work), Path(a.out)
    (out / 'imagesTs').mkdir(parents=True, exist_ok=True); (out / 'labelsTs').mkdir(parents=True, exist_ok=True)
    manifest, cases = {}, []
    for i in range(1, 21):
        zp = zips / f'3Dircadb1.{i}.zip'
        case = f'IRCADb_{i:02d}'
        if (out / 'labelsTs' / f'{case}.nii.gz').exists() and (out / 'imagesTs' / f'{case}_0000.nii.gz').exists():
            print(f'{case}: already converted, re-reading manifest entry', flush=True)
        pdir = work / f'3Dircadb1.{i}'
        # Only the CT and the liver / liver-tumor masks are extracted (the other
        # ~20 structures would need ~0.5 GB per patient as uncompressed DICOM).
        safe_extract(zp, work, keep=lambda n: n.endswith(('PATIENT_DICOM.zip', 'MASKS_DICOM.zip')))
        safe_extract(pdir / 'PATIENT_DICOM.zip', pdir)
        structures = mask_names(pdir / 'MASKS_DICOM.zip')
        tumor_dirs = [s for s in structures if TUMOR.match(s)]
        wanted = {'liver', *tumor_dirs}
        safe_extract(pdir / 'MASKS_DICOM.zip', pdir, keep=lambda n: n.count('/') >= 1 and n.split('/')[1] in wanted)
        ct_files = slice_files(pdir / 'PATIENT_DICOM')
        ct = read_series(ct_files)
        names = [f.name for f in ct_files]
        size = ct.GetSize()
        lab = np.zeros(sitk.GetArrayFromImage(ct).shape, np.uint8)

        def mask(name):
            files = [pdir / 'MASKS_DICOM' / name / n for n in names]
            if not all(f.exists() for f in files):
                raise RuntimeError(f'{case}: mask {name} does not cover every CT slice')
            m = sitk.GetArrayFromImage(read_series(files)) > 0
            if m.shape != lab.shape:
                raise RuntimeError(f'{case}: mask {name} shape {m.shape} vs CT {lab.shape}')
            return m

        lab[mask('liver')] = 1
        for t in tumor_dirs:
            lab[mask(t)] = 2
        img = sitk.Cast(ct, sitk.sitkInt16)
        seg = sitk.GetImageFromArray(lab); seg.CopyInformation(img)
        sitk.WriteImage(img, str(out / 'imagesTs' / f'{case}_0000.nii.gz'), useCompression=True)
        sitk.WriteImage(seg, str(out / 'labelsTs' / f'{case}.nii.gz'), useCompression=True)
        manifest[case] = dict(source=zp.name, size=list(size), spacing=[round(s, 4) for s in ct.GetSpacing()],
                              tumor_folders=tumor_dirs,
                              other_lesion_folders=[s for s in structures if re.search('kyst|cyst|tumor', s, re.I)
                                                    and s not in tumor_dirs],
                              liver_voxels=int((lab > 0).sum()), tumor_voxels=int((lab == 2).sum()))
        cases.append(case)
        shutil.rmtree(pdir)
        print(f'{case}: size {size}, spacing {manifest[case]["spacing"]}, tumor folders {len(tumor_dirs)}, '
              f'tumor voxels {manifest[case]["tumor_voxels"]}', flush=True)
    (out / 'cases.txt').write_text('\n'.join(cases) + '\n')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{len(cases)} cases, {sum(bool(m["tumor_voxels"]) for m in manifest.values())} with tumor')


if __name__ == '__main__':
    main()
