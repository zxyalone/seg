#!/usr/bin/env python3
"""
Full evaluation of one prediction folder (manuscript Section 4.3, E1).

Per case (raw image grid, as written by nnUNetv2_predict):
  * Dice / IoU for liver parenchyma (label 1), whole liver (label > 0) and tumor
    (label 2), plus TP/FP/FN voxel counts so that global Dice can be pooled.
  * Tumor HD95 (mm) and NSD at tolerance tau (mm).
  * Lesion-wise detection: 3D connected components (26-connectivity by default)
    of reference and prediction, candidate pairs with IoU > thr, one-to-one
    Hungarian assignment maximising summed IoU. TP / FN / FP per case, for
    thr = 0.1 (main) and 0.5 (supplement).
Per reference lesion: volume, equivalent-sphere diameter, size stratum
(<10, 10-20, >20 mm), CNR and CNR tertile, detected at each threshold, IoU of
its match.

CNR follows step2_mechanism/scripts/s42_real_pc.py exactly, because the tertile
thresholds (1.2785 / 1.9208) were frozen on the 518 training lesions with that
code: CT and labels are resampled to the plan spacing (1.0 x 0.779 x 0.779 mm),
lesions are 6-connected components of label 2, the ring is liver (label 1)
voxels at Euclidean voxel distance 2..6 outside the lesion that belong to no
other lesion (fallback 1..6 if < 20 voxels; CNR undefined if < 10), and
CNR = |mean_lesion - mean_ring| / std_ring in HU. Each raw-grid lesion takes
the CNR of the plan-grid lesion that covers most of its voxels.

Conventions (fixed before any test evaluation; also written to summary.json):
  * empty reference and empty prediction -> tumor Dice/IoU = 1 (option
    --empty-empty nan excludes such cases instead);
  * empty reference, non-empty prediction -> tumor Dice/IoU = 0;
  * tumor-bearing case without predicted tumor -> HD95 = image diagonal (mm),
    NSD = 0; the number of such cases is reported.

Usage:
  python experiments/eval/evaluate.py --pred-dir P --gt-dir G --image-dir I \
      --cases-file experiments/eval/test_cases.txt --out-dir results/A6_s0/test
Writes cases.csv, lesions.csv, pred_lesions.csv, summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

PLAN_SPACING_SITK = (0.7791720032691956, 0.7791720032691956, 1.0)  # x, y, z
CNR_TERTILES = (1.2785, 1.9208)  # frozen on the 518 training lesions
SIZE_EDGES_MM = (10.0, 20.0)
IOU_THRESHOLDS = (0.1, 0.5)
TUMOR_FREE = {"LiTS_032", "LiTS_106", "LiTS_115"}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read(path: Path):
    img = sitk.ReadImage(str(path))
    return img, sitk.GetArrayFromImage(img)


def find_image(image_dir: Path, case: str) -> Path | None:
    for name in (f"{case}_0000.nii.gz", f"{case}.nii.gz"):
        p = image_dir / name
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def overlap(pred: np.ndarray, gt: np.ndarray, empty_empty: float):
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    if tp + fp + fn == 0:
        return empty_empty, empty_empty, tp, fp, fn
    return 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn), tp, fp, fn


# ---------------------------------------------------------------------------
# Surface metrics
# ---------------------------------------------------------------------------

def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1),
                                      border_value=0)


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing_zyx, tau_mm: float):
    """HD95 = max of the two directed 95th percentiles; NSD = surface Dice at tau."""
    union = pred | gt
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, np.array(union.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    p, g = pred[sl], gt[sl]
    sp, sg = _surface(p), _surface(g)
    dist_to_g = ndi.distance_transform_edt(~sg, sampling=spacing_zyx)
    dist_to_p = ndi.distance_transform_edt(~sp, sampling=spacing_zyx)
    d_pg = dist_to_g[sp]
    d_gp = dist_to_p[sg]
    hd95 = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    nsd = float((np.count_nonzero(d_pg <= tau_mm) + np.count_nonzero(d_gp <= tau_mm))
                / (len(d_pg) + len(d_gp)))
    return hd95, nsd


# ---------------------------------------------------------------------------
# Lesion matching
# ---------------------------------------------------------------------------

def components(mask: np.ndarray, connectivity: int):
    structure = ndi.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    lab, n = ndi.label(mask, structure=structure)
    return lab, n


def match_lesions(gt_lab, n_gt, pr_lab, n_pr, thr: float):
    """One-to-one Hungarian matching on IoU > thr. Returns (gt->pred, gt IoU)."""
    match = {i: None for i in range(1, n_gt + 1)}
    iou_of = {i: 0.0 for i in range(1, n_gt + 1)}
    if n_gt == 0 or n_pr == 0:
        return match, iou_of
    both = (gt_lab > 0) & (pr_lab > 0)
    pairs, inter = np.unique(np.stack([gt_lab[both], pr_lab[both]]), axis=1, return_counts=True)
    gt_sz = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
    pr_sz = np.bincount(pr_lab.ravel(), minlength=n_pr + 1)
    iou = np.zeros((n_gt, n_pr))
    for (gi, pi), it in zip(pairs.T, inter):
        iou[gi - 1, pi - 1] = it / (gt_sz[gi] + pr_sz[pi] - it)
    cand = np.where(iou > thr, iou, 0.0)
    rows, cols = linear_sum_assignment(-cand)
    for r, c in zip(rows, cols):
        if cand[r, c] > 0:
            match[r + 1] = c + 1
            iou_of[r + 1] = float(iou[r, c])
    return match, iou_of


# ---------------------------------------------------------------------------
# CNR (identical rule to s42_real_pc.py)
# ---------------------------------------------------------------------------

def _resample(img: sitk.Image, ref_size, interp):
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(PLAN_SPACING_SITK)
    f.SetSize(ref_size)
    f.SetOutputDirection(img.GetDirection())
    f.SetOutputOrigin(img.GetOrigin())
    f.SetInterpolator(interp)
    f.SetDefaultPixelValue(0)
    return f.Execute(img)


def plan_grid_cnr(ct_img: sitk.Image, gt_img: sitk.Image):
    """Return (plan-grid lesion label image, {label: cnr})."""
    sz, osp = ct_img.GetSize(), ct_img.GetSpacing()
    new = [int(round(sz[i] * osp[i] / PLAN_SPACING_SITK[i])) for i in range(3)]
    ct = sitk.GetArrayFromImage(_resample(ct_img, new, sitk.sitkLinear)).astype(np.float32)
    seg_img = _resample(gt_img, new, sitk.sitkNearestNeighbor)
    seg = sitk.GetArrayFromImage(seg_img)
    liver, tumor = seg == 1, seg == 2
    lab, n = ndi.label(tumor)  # 6-connectivity, as in s42
    cnr = {}
    for i in range(1, n + 1):
        m_full = lab == i
        idx = np.argwhere(m_full)
        lo = np.maximum(idx.min(0) - 10, 0)
        hi = np.minimum(idx.max(0) + 11, np.array(m_full.shape))
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        m, liv, oth = m_full[sl], liver[sl], (tumor & ~m_full)[sl]
        d_out = ndi.distance_transform_edt(~m)
        peri = (~m) & liv & (d_out >= 2) & (d_out <= 6) & ~oth
        if peri.sum() < 20:
            peri = (~m) & liv & (d_out >= 1) & (d_out <= 6) & ~oth
        if peri.sum() >= 10:
            im = ct[sl]
            cnr[i] = abs(float(im[m].mean()) - float(im[peri].mean())) / (float(im[peri].std()) + 1e-8)
    lab_img = sitk.GetImageFromArray(lab.astype(np.int32))
    lab_img.CopyInformation(seg_img)
    return lab_img, cnr


def raw_lesion_cnr(gt_lab: np.ndarray, n_gt: int, gt_img: sitk.Image, ct_img: sitk.Image):
    plan_lab_img, cnr = plan_grid_cnr(ct_img, gt_img)
    back = sitk.Resample(plan_lab_img, gt_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    back = sitk.GetArrayFromImage(back)
    out = {}
    for i in range(1, n_gt + 1):
        labels = back[gt_lab == i]
        labels = labels[labels > 0]
        if labels.size == 0:
            out[i] = np.nan
            continue
        major = int(np.bincount(labels).argmax())
        out[i] = cnr.get(major, np.nan)
    return out


def cnr_stratum(c: float) -> str:
    if not np.isfinite(c):
        return "na"
    return "low" if c < CNR_TERTILES[0] else ("mid" if c < CNR_TERTILES[1] else "high")


def size_stratum(d: float) -> str:
    return "<10" if d < SIZE_EDGES_MM[0] else ("10-20" if d <= SIZE_EDGES_MM[1] else ">20")


# ---------------------------------------------------------------------------
# Per case
# ---------------------------------------------------------------------------

def evaluate_case(case: str, args) -> dict:
    gt_img, gt = read(Path(args.gt_dir) / f"{case}.nii.gz")
    pr_img, pr = read(Path(args.pred_dir) / f"{case}.nii.gz")
    if gt.shape != pr.shape or not np.allclose(gt_img.GetSpacing(), pr_img.GetSpacing(), atol=1e-4):
        raise RuntimeError(f"{case}: prediction and reference grids differ")
    spacing_zyx = tuple(float(s) for s in gt_img.GetSpacing()[::-1])
    voxel_mm3 = float(np.prod(spacing_zyx))

    row = {"case": case}
    for name, pm, gm in (("liver1", pr == 1, gt == 1), ("liver", pr > 0, gt > 0),
                         ("tumor", pr == 2, gt == 2)):
        ee = args.empty_empty if name == "tumor" else np.nan
        d, j, tp, fp, fn = overlap(pm, gm, ee)
        row.update({f"{name}_dice": d, f"{name}_iou": j,
                    f"{name}_tp": tp, f"{name}_fp": fp, f"{name}_fn": fn})

    pt, gtt = pr == 2, gt == 2
    row["has_tumor"] = bool(gtt.any())
    row["pred_empty"] = not pt.any()
    if gtt.any() and pt.any():
        row["tumor_hd95"], row["tumor_nsd"] = surface_metrics(pt, gtt, spacing_zyx, args.tau)
        row["tumor_surface_missing"] = False
    elif gtt.any():
        row["tumor_hd95"] = float(np.linalg.norm(np.array(gt.shape) * np.array(spacing_zyx)))
        row["tumor_nsd"] = 0.0
        row["tumor_surface_missing"] = True
    else:
        row["tumor_hd95"] = row["tumor_nsd"] = np.nan
        row["tumor_surface_missing"] = False

    gt_lab, n_gt = components(gtt, args.connectivity)
    pr_lab, n_pr = components(pt, args.connectivity)
    gt_vox = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
    pr_vox = np.bincount(pr_lab.ravel(), minlength=n_pr + 1)

    cnr = {i: np.nan for i in range(1, n_gt + 1)}
    if args.image_dir and n_gt:
        ip = find_image(Path(args.image_dir), case)
        if ip is not None:
            cnr = raw_lesion_cnr(gt_lab, n_gt, gt_img, sitk.ReadImage(str(ip)))

    lesions = []
    pred_rows = []
    for thr in IOU_THRESHOLDS:
        match, iou_of = match_lesions(gt_lab, n_gt, pr_lab, n_pr, thr)
        tag = f"{thr:g}"
        matched_pred = {p for p in match.values() if p is not None}
        row[f"les_tp@{tag}"] = len(matched_pred)
        row[f"les_fn@{tag}"] = n_gt - len(matched_pred)
        row[f"les_fp@{tag}"] = n_pr - len(matched_pred)
        for i in range(1, n_gt + 1):
            if thr == IOU_THRESHOLDS[0]:
                vol = gt_vox[i] * voxel_mm3
                diam = (6 * vol / np.pi) ** (1 / 3)
                lesions.append({"case": case, "lesion": i, "voxels": int(gt_vox[i]),
                                "volume_mm3": vol, "diameter_mm": diam,
                                "size_stratum": size_stratum(diam), "cnr": cnr[i],
                                "cnr_stratum": cnr_stratum(cnr[i])})
            lesions[i - 1][f"detected@{tag}"] = match[i] is not None
            lesions[i - 1][f"iou@{tag}"] = iou_of[i]
        for p in range(1, n_pr + 1):
            if thr == IOU_THRESHOLDS[0]:
                vol = pr_vox[p] * voxel_mm3
                pred_rows.append({"case": case, "pred_lesion": p, "voxels": int(pr_vox[p]),
                                  "volume_mm3": vol,
                                  "diameter_mm": (6 * vol / np.pi) ** (1 / 3)})
            pred_rows[p - 1][f"fp@{tag}"] = p not in matched_pred
    row["n_gt_lesions"], row["n_pred_lesions"] = n_gt, n_pr
    return {"case": row, "lesions": lesions, "pred": pred_rows}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(cases: pd.DataFrame, lesions: pd.DataFrame, args) -> dict:
    out = {"conventions": {
        "empty_empty_tumor_dice": args.empty_empty, "connectivity": args.connectivity,
        "nsd_tau_mm": args.tau, "iou_thresholds": list(IOU_THRESHOLDS),
        "cnr_tertiles": list(CNR_TERTILES), "size_edges_mm": list(SIZE_EDGES_MM),
        "aggregation": "mean over cases; std with ddof=0; global Dice pools voxel counts"}}
    cohorts = {"All": cases, "Tumor": cases[cases.has_tumor]}
    for cname, df in cohorts.items():
        res = {"n_cases": int(len(df))}
        for k in ("liver1", "liver", "tumor"):
            res[f"{k}_dice_mean"] = float(np.nanmean(df[f"{k}_dice"]))
            res[f"{k}_dice_std"] = float(np.nanstd(df[f"{k}_dice"]))
            res[f"{k}_iou_mean"] = float(np.nanmean(df[f"{k}_iou"]))
            tp, fp, fn = (df[f"{k}_{x}"].sum() for x in ("tp", "fp", "fn"))
            res[f"{k}_global_dice"] = float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else np.nan
        tumor = df[df.has_tumor]
        res["tumor_hd95_mean"] = float(np.nanmean(tumor.tumor_hd95)) if len(tumor) else np.nan
        res["tumor_hd95_median"] = float(np.nanmedian(tumor.tumor_hd95)) if len(tumor) else np.nan
        res["tumor_nsd_mean"] = float(np.nanmean(tumor.tumor_nsd)) if len(tumor) else np.nan
        res["n_tumor_cases_without_prediction"] = int(tumor.tumor_surface_missing.sum())
        for thr in IOU_THRESHOLDS:
            tag = f"{thr:g}"
            tp, fp, fn = (int(df[f"les_{x}@{tag}"].sum()) for x in ("tp", "fp", "fn"))
            rec = tp / (tp + fn) if tp + fn else np.nan
            prec = tp / (tp + fp) if tp + fp else np.nan
            res[f"lesion@{tag}"] = {
                "tp": tp, "fp": fp, "fn": fn, "recall": rec, "precision": prec,
                "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
                "fp_per_scan": fp / len(df)}
        out[cname] = res
    strata = {}
    for col in ("size_stratum", "cnr_stratum"):
        for s, g in lesions.groupby(col):
            strata[f"{col}={s}"] = {"n_lesions": int(len(g)),
                                    **{f"recall@{t:g}": float(g[f'detected@{t:g}'].mean())
                                       for t in IOU_THRESHOLDS}}
    out["lesion_strata"] = strata
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--image-dir", default=None, help="raw CT (case_0000.nii.gz) for lesion CNR")
    ap.add_argument("--cases-file", default=None, help="one case id per line; default: all predictions")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tau", type=float, default=2.0, help="NSD tolerance in mm")
    ap.add_argument("--connectivity", type=int, default=26, choices=(6, 18, 26))
    ap.add_argument("--empty-empty", default="1", choices=("1", "nan"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.empty_empty = 1.0 if args.empty_empty == "1" else np.nan

    if args.cases_file:
        cases = [c.strip() for c in Path(args.cases_file).read_text().split() if c.strip()]
    else:
        cases = sorted(p.name[:-7] for p in Path(args.pred_dir).glob("*.nii.gz"))
    missing = [c for c in cases if not (Path(args.pred_dir) / f"{c}.nii.gz").is_file()]
    if missing:
        sys.exit(f"missing predictions: {missing}")

    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(evaluate_case, cases, [args] * len(cases)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cdf = pd.DataFrame([r["case"] for r in results])
    ldf = pd.DataFrame([l for r in results for l in r["lesions"]])
    pdf = pd.DataFrame([p for r in results for p in r["pred"]])
    if ldf.empty:
        ldf = pd.DataFrame(columns=["case", "lesion", "size_stratum", "cnr_stratum"]
                           + [f"detected@{t:g}" for t in IOU_THRESHOLDS])
    cdf.to_csv(out / "cases.csv", index=False)
    ldf.to_csv(out / "lesions.csv", index=False)
    pdf.to_csv(out / "pred_lesions.csv", index=False)
    summary = summarize(cdf, ldf, args)
    summary["pred_dir"], summary["gt_dir"] = str(args.pred_dir), str(args.gt_dir)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    t = summary["Tumor"]
    print(f"{len(cases)} cases -> {out}")
    print(f"Tumor cohort (n={t['n_cases']}): tumor Dice {t['tumor_dice_mean']:.4f} ± {t['tumor_dice_std']:.4f}"
          f" | global {t['tumor_global_dice']:.4f} | HD95 {t['tumor_hd95_mean']:.2f} mm"
          f" | NSD@{args.tau:g} {t['tumor_nsd_mean']:.4f}")
    a = summary["All"]["lesion@0.1"]
    print(f"All cases lesion@0.1: recall {a['recall']:.3f} precision {a['precision']:.3f}"
          f" F1 {a['f1']:.3f} FP/scan {a['fp_per_scan']:.2f}")


if __name__ == "__main__":
    main()
