#!/usr/bin/env python3
"""Server-side: cut the axial crops for Fig. 5 (qualitative comparison) from
the test set. Lesions follow the selection rule registered in plan §11
(2026-10-14; cost row amended the same day, see §11) and are passed explicitly; numbering is that of the frozen
protocol tables (26-connected components of the reference tumour label).

    python extract_qualitative.py --out qualitative.npz

Images are reoriented to LPS so that rows run anterior -> posterior and
columns patient right -> left (radiological display). For each lesion the
axial slice with its largest cross-section is cropped (--size mm, centred on
the lesion in that slice) from the CT, the reference tumour label, the target
lesion, and the 4-mm post-processed tumour predictions of A1_s0 and A13b_s0.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'experiments/analysis'))
from missed_lesion_probability import E, _drop_small  # noqa: E402

DATA = Path('/root/autodl-tmp/nnUNet_data')
RAW = DATA / 'nnUNet_raw/Dataset101_LiTS'
FOLD = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
LESIONS = [('gain', 'LiTS_122', 17), ('gain', 'LiTS_117', 74), ('cost', 'LiTS_109', 26), ('miss', 'LiTS_097', 10)]
RUNS = ('A1_s0', 'A13b_s0')


def read_lps(p):
    img = sitk.DICOMOrient(sitk.ReadImage(str(p)), 'LPS')
    return img, sitk.GetArrayFromImage(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='qualitative.npz')
    ap.add_argument('--size', type=float, default=64.0)
    a = ap.parse_args()
    out = {}
    for k, (kind, case, lesion) in enumerate(LESIONS):
        cimg, ct = read_lps(RAW / 'imagesTr' / f'{case}_0000.nii.gz')
        _, gt = read_lps(RAW / 'labelsTr' / f'{case}.nii.gz')
        # lesion numbering on the original grid, as in the protocol tables
        _, gt_raw = E.read(RAW / 'labelsTr' / f'{case}.nii.gz')
        lab_raw, _ = E.components(gt_raw == 2, 26)
        lab_img = sitk.GetImageFromArray(lab_raw.astype(np.int32))
        lab_img.CopyInformation(sitk.ReadImage(str(RAW / 'labelsTr' / f'{case}.nii.gz')))
        lab = sitk.GetArrayFromImage(sitk.DICOMOrient(lab_img, 'LPS'))
        target = lab == lesion
        sp = np.array(cimg.GetSpacing()[::-1])                   # z, y, x
        z = int(np.argmax(target.sum((1, 2))))
        cy, cx = np.argwhere(target[z]).mean(0)
        h = int(round(a.size / 2 / sp[1]))
        y0, x0 = int(round(cy)) - h, int(round(cx)) - h

        def crop(v, fill=0):
            pad = np.pad(v[z], h + 1, constant_values=fill)
            return pad[y0 + h + 1:y0 + 3 * h + 1, x0 + h + 1:x0 + 3 * h + 1]

        out[f'{k}_ct'] = crop(ct.astype(np.float32), -1000).astype(np.float32)
        out[f'{k}_gt'] = crop(gt == 2)
        out[f'{k}_target'] = crop(target)
        for run in RUNS:
            pimg, pred = read_lps(DATA / 'nnUNet_results_v2' / run / FOLD / 'test_pred' / f'{case}.nii.gz')
            assert pred.shape == ct.shape
            pp = _drop_small(pred == 2, float(np.prod(sp)), 4.0)
            out[f'{k}_{run}'] = crop(pp)
        out[f'{k}_meta'] = np.array([kind, case, str(lesion), str(z), f'{sp[1]:.4f}'])
        print(kind, case, lesion, 'slice', z, 'crop', out[f'{k}_ct'].shape, flush=True)
    np.savez_compressed(a.out, **out)


if __name__ == '__main__':
    main()
