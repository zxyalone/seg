#!/usr/bin/env python3
"""
Paired comparisons between methods (manuscript Section 4.3, "Statistical analysis").

Each method is given as one or more evaluate.py output folders (one per seed).
Per-case values are averaged over seeds before testing. For every metric:
  * mean difference B - A with a 95% paired-bootstrap CI over cases;
  * two-sided Wilcoxon signed-rank p-value and matched-pairs rank-biserial r;
  * Holm correction over all pre-specified comparisons in the call.
Lesion-level quantities (recall, precision, F1, FP/scan) are pooled ratios, so
the bootstrap resamples cases and recomputes the pooled ratio; there is no
Wilcoxon test for them (one value per cohort, not per case).

Usage (A6 against three references, Tumor cohort, IoU>0.1 lesion matching):
  python experiments/eval/stats.py --cohort Tumor \
     --method A0=results/A0_s0/test,results/A0_s1/test,results/A0_s2/test \
     --method A1=results/A1_s0/test \
     --method A6=results/A6_s0/test,results/A6_s1/test,results/A6_s2/test \
     --compare A6:A0 A6:A1 --out results/stats_test.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CASE_METRICS = ("tumor_dice", "tumor_iou", "liver_dice", "liver1_dice", "tumor_hd95", "tumor_nsd")
LESION_METRICS = ("lesion_recall", "lesion_precision", "lesion_f1", "fp_per_scan")
COUNT_COLS = ("les_tp", "les_fp", "les_fn")


def load_method(dirs: list[str], cohort: str, thr: str) -> pd.DataFrame:
    frames = []
    for d in dirs:
        df = pd.read_csv(Path(d) / "cases.csv")
        if cohort == "Tumor":
            df = df[df.has_tumor]
        keep = ["case", *CASE_METRICS, *(f"{c}@{thr}" for c in COUNT_COLS)]
        frames.append(df[keep].set_index("case"))
    cases = sorted(set.intersection(*(set(f.index) for f in frames)))
    if any(len(f) != len(cases) for f in frames):
        raise ValueError(f"seed runs of {dirs} cover different cases")
    stacked = pd.concat([f.loc[cases] for f in frames])
    return stacked.groupby(level=0).mean().loc[cases]


def pooled(df: pd.DataFrame, thr: str) -> dict:
    tp, fp, fn = (df[f"{c}@{thr}"].sum() for c in COUNT_COLS)
    return {
        "lesion_recall": tp / (tp + fn) if tp + fn else np.nan,
        "lesion_precision": tp / (tp + fp) if tp + fp else np.nan,
        "lesion_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
        "fp_per_scan": fp / len(df),
    }


def rank_biserial(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    order = np.argsort(p[ok])
    q = p[ok][order] * (ok.sum() - np.arange(ok.sum()))
    q = np.minimum(np.maximum.accumulate(q), 1.0)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[ok] = tmp
    return out.tolist()


def compare(a: pd.DataFrame, b: pd.DataFrame, thr: str, n_boot: int, rng) -> list[dict]:
    cases = a.index.intersection(b.index)
    a, b = a.loc[cases], b.loc[cases]
    n = len(cases)
    idx = rng.integers(0, n, size=(n_boot, n))
    rows = []
    for m in CASE_METRICS:
        da, db = a[m].to_numpy(float), b[m].to_numpy(float)
        ok = np.isfinite(da) & np.isfinite(db)
        diff = db - da
        boots = np.nanmean(np.where(ok[idx], diff[idx], np.nan), axis=1)
        p = wilcoxon(diff[ok]).pvalue if ok.sum() > 0 and np.any(diff[ok] != 0) else np.nan
        rows.append({"metric": m, "n": int(ok.sum()), "mean_a": float(np.mean(da[ok])),
                     "mean_b": float(np.mean(db[ok])), "delta": float(np.mean(diff[ok])),
                     "ci_lo": float(np.nanpercentile(boots, 2.5)),
                     "ci_hi": float(np.nanpercentile(boots, 97.5)),
                     "p_wilcoxon": float(p), "rank_biserial": rank_biserial(diff[ok])})
    pa, pb = pooled(a, thr), pooled(b, thr)
    counts_a = a[[f"{c}@{thr}" for c in COUNT_COLS]].to_numpy()
    counts_b = b[[f"{c}@{thr}" for c in COUNT_COLS]].to_numpy()
    boot = {m: [] for m in LESION_METRICS}
    for row in idx:
        ra = pooled(pd.DataFrame(counts_a[row], columns=[f"{c}@{thr}" for c in COUNT_COLS]), thr)
        rb = pooled(pd.DataFrame(counts_b[row], columns=[f"{c}@{thr}" for c in COUNT_COLS]), thr)
        for m in LESION_METRICS:
            boot[m].append(rb[m] - ra[m])
    for m in LESION_METRICS:
        rows.append({"metric": f"{m}@{thr}", "n": n, "mean_a": pa[m], "mean_b": pb[m],
                     "delta": pb[m] - pa[m],
                     "ci_lo": float(np.nanpercentile(boot[m], 2.5)),
                     "ci_hi": float(np.nanpercentile(boot[m], 97.5)),
                     "p_wilcoxon": np.nan, "rank_biserial": np.nan})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", action="append", required=True,
                    help="NAME=dir[,dir...] (one evaluate.py folder per seed)")
    ap.add_argument("--compare", nargs="+", required=True, help="B:A pairs; delta = B - A")
    ap.add_argument("--cohort", default="Tumor", choices=("Tumor", "All"))
    ap.add_argument("--iou-thr", default="0.1")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    methods = {}
    for spec in args.method:
        name, dirs = spec.split("=", 1)
        methods[name] = load_method(dirs.split(","), args.cohort, args.iou_thr)

    rng = np.random.default_rng(args.seed)
    rows = []
    for pair in args.compare:
        b, a = pair.split(":")
        for r in compare(methods[a], methods[b], args.iou_thr, args.n_boot, rng):
            rows.append({"comparison": f"{b} - {a}", **r})
    df = pd.DataFrame(rows)
    # Holm over the pre-specified comparisons, separately for each metric.
    df["p_holm"] = np.nan
    for m, g in df.groupby("metric"):
        df.loc[g.index, "p_holm"] = holm(g.p_wilcoxon.tolist())
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
