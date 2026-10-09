#!/usr/bin/env python3
"""Self-test of evaluate.py and stats.py on synthetic volumes with known answers.

    python experiments/eval/test_eval.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
SHAPE = (40, 64, 64)  # z, y, x
SPACING_XYZ = (0.7791720032691956, 0.7791720032691956, 1.0)  # plan spacing: no resampling


def ball(center, r):
    z, y, x = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    return (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= r ** 2


def write(arr, path):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(SPACING_XYZ)
    sitk.WriteImage(img, str(path))


def make_cases(root: Path):
    for d in ("gt", "pred", "img"):
        (root / d).mkdir()
    liver = ball((20, 32, 32), 18)
    rng = np.random.default_rng(0)

    # case A: lesions t1 (found exactly), t2 (missed); one FP blob
    gt = liver.astype(np.uint8)
    t1, t2 = ball((20, 24, 24), 4), ball((20, 40, 40), 3)
    gt[t1 | t2] = 2
    pr = liver.astype(np.uint8)
    pr[t1] = 2
    pr[ball((12, 40, 24), 2)] = 2
    ct = np.where(liver, 100.0, -100.0) + rng.normal(0, 10, SHAPE)
    ct[t1 | t2] = 80.0 + rng.normal(0, 10, (t1 | t2).sum())
    write(gt, root / "gt/A.nii.gz"); write(pr, root / "pred/A.nii.gz")
    write(ct.astype(np.float32), root / "img/A_0000.nii.gz")

    # case B: one lesion predicted as two halves -> 1 TP + 1 FP at IoU>0.1
    gt = liver.astype(np.uint8)
    t = ball((20, 32, 32), 6)
    gt[t] = 2
    pr = liver.astype(np.uint8)
    z, y, x = np.nonzero(t)
    left = np.zeros(SHAPE, bool); left[z[x < 31], y[x < 31], x[x < 31]] = True
    right = np.zeros(SHAPE, bool); right[z[x > 33], y[x > 33], x[x > 33]] = True
    pr[left | right] = 2
    write(gt, root / "gt/B.nii.gz"); write(pr, root / "pred/B.nii.gz")

    # case C: tumor-free, empty prediction -> Dice 1
    write(liver.astype(np.uint8), root / "gt/C.nii.gz")
    write(liver.astype(np.uint8), root / "pred/C.nii.gz")

    # case D: tumor-free, false-positive prediction -> Dice 0
    pr = liver.astype(np.uint8); pr[ball((20, 32, 32), 2)] = 2
    write(liver.astype(np.uint8), root / "gt/D.nii.gz"); write(pr, root / "pred/D.nii.gz")

    # case E: prediction = reference shifted by one voxel along x
    gt = liver.astype(np.uint8); t = ball((20, 32, 32), 6); gt[t] = 2
    pr = liver.astype(np.uint8); pr[np.roll(t, 1, axis=2)] = 2
    write(gt, root / "gt/E.nii.gz"); write(pr, root / "pred/E.nii.gz")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_cases(root)
        cmd = [sys.executable, str(HERE / "evaluate.py"), "--pred-dir", str(root / "pred"),
               "--gt-dir", str(root / "gt"), "--image-dir", str(root / "img"),
               "--out-dir", str(root / "out"), "--workers", "1"]
        subprocess.run(cmd, check=True)
        c = pd.read_csv(root / "out/cases.csv").set_index("case")
        les = pd.read_csv(root / "out/lesions.csv")
        s = json.loads((root / "out/summary.json").read_text())

        a = c.loc["A"]
        assert (a["les_tp@0.1"], a["les_fn@0.1"], a["les_fp@0.1"]) == (1, 1, 1), a
        t1, t2, fp = ball((20, 24, 24), 4).sum(), ball((20, 40, 40), 3).sum(), ball((12, 40, 24), 2).sum()
        assert abs(a.tumor_dice - 2 * t1 / (2 * t1 + fp + t2)) < 1e-9
        b = c.loc["B"]
        assert (b["les_tp@0.1"], b["les_fp@0.1"], b["les_fn@0.1"]) == (1, 1, 0), b
        assert b["les_tp@0.5"] == 0 and b["les_fn@0.5"] == 1
        assert c.loc["C"].tumor_dice == 1.0 and c.loc["D"].tumor_dice == 0.0
        e = c.loc["E"]
        assert abs(e.tumor_hd95 - SPACING_XYZ[0]) < 1e-6, e.tumor_hd95
        assert abs(e.tumor_nsd - 1.0) < 1e-9
        # CNR of the two lesions in case A: |80-100|/10 = 2 (noise-limited estimate)
        cnr = les[les.case == "A"].cnr.to_numpy()
        assert np.all(np.abs(cnr - 2.0) < 0.3), cnr
        from evaluate import cnr_stratum  # noqa: E402
        assert all(cnr_stratum(v) == st for v, st in zip(les.cnr, les.cnr_stratum) if np.isfinite(v))
        assert s["Tumor"]["n_cases"] == 3 and s["All"]["n_cases"] == 5
        assert abs(s["All"]["lesion@0.1"]["fp_per_scan"] - 3 / 5) < 1e-9  # A, B, D

        # stats.py: identical method vs itself -> zero deltas; Holm runs
        out = root / "stats.csv"
        subprocess.run([sys.executable, str(HERE / "stats.py"), "--cohort", "All",
                        "--method", f"X={root / 'out'}", "--method", f"Y={root / 'out'}",
                        "--compare", "Y:X", "--n-boot", "200", "--out", str(out)],
                       check=True, stdout=subprocess.DEVNULL)
        st = pd.read_csv(out)
        assert np.allclose(st.delta.fillna(0), 0) and np.allclose(st[["ci_lo", "ci_hi"]].fillna(0), 0)
    from stats import holm, rank_biserial  # noqa: E402
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    print("test_eval: all checks passed")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
