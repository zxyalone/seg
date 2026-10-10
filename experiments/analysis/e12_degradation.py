#!/usr/bin/env python3
"""
E12 / Fig. 6: controlled lesion-contrast degradation (manuscript Section 4.9).

For each reference lesion j (26-connected component of label 2, the same
components evaluate.py matches), intensities are moved toward the lesion's
liver-ring mean, in HU, before nnU-Net normalization:

    x'_v = x_v - (1 - alpha) * (mu_lesion,j - mu_ring,j) * w_j(v)

w_j = 1 inside the lesion and decays outside it as a Gaussian-smoothed mask
(sigma = 1 voxel), so there is no step at the border while the lesion's mean
shift is exact. The lesion's internal texture and the ring statistics are
unchanged, so the lesion CNR becomes alpha * CNR. The ring is liver (label 1)
at Euclidean voxel distance 2..6 outside the lesion, excluding other lesions
(fallback 1..6 if < 20 voxels; lesions with < 10 ring voxels are left
unchanged and flagged). Labels are not changed.

All models are loaded once; each case and alpha is degraded and preprocessed
once and then run through every model.
  OUT/pred/<run>/a<alpha>/<case>.nii.gz
  OUT/lesions/<case>.json   per-lesion mu, sigma, CNR before / after (raw grid)

  CUDA_VISIBLE_DEVICES=0 python experiments/analysis/e12_degradation.py run --shard 0/3 &
  ... (shards 1/3, 2/3) ...; wait
  python experiments/analysis/e12_degradation.py summarize
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ALPHAS = (1.0, 0.75, 0.5, 0.25)
DEFAULT_RUNS = ["A0:0,1,2", "A1:0,1,2", "A6:0,1,2", "A7:0"]


def atag(a: float) -> str:
    return f"a{a:g}"


def lesion_stats(img: np.ndarray, gt: np.ndarray):
    """Per-lesion box, mask, mean, ring mean / std on the raw grid."""
    tumor, liver = gt == 2, gt == 1
    lab, n = ndi.label(tumor, structure=ndi.generate_binary_structure(3, 3))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        lo = [max(s.start - 10, 0) for s in sl]
        hi = [min(s.stop + 10, d) for s, d in zip(sl, lab.shape)]
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        m = lab[box] == i
        oth = tumor[box] & ~m
        d_out = ndi.distance_transform_edt(~m)
        ring = (~m) & liver[box] & (d_out >= 2) & (d_out <= 6) & ~oth
        if ring.sum() < 20:
            ring = (~m) & liver[box] & (d_out >= 1) & (d_out <= 6) & ~oth
        rec = dict(lesion=i, voxels=int(m.sum()), box=[(b.start, b.stop) for b in box],
                   ring_voxels=int(ring.sum()), valid=bool(ring.sum() >= 10))
        im = img[box]
        rec["mu_lesion"] = float(im[m].mean())
        if rec["valid"]:
            rec["mu_ring"], rec["sd_ring"] = float(im[ring].mean()), float(im[ring].std())
            rec["cnr"] = abs(rec["mu_lesion"] - rec["mu_ring"]) / (rec["sd_ring"] + 1e-8)
        out.append((rec, m, ring))
    return out


def degrade(img: np.ndarray, lesions, alpha: float) -> np.ndarray:
    if alpha == 1.0:
        return img
    x = img.astype(np.float32, copy=True)
    for rec, m, _ in lesions:
        if not rec["valid"]:
            continue
        box = tuple(slice(a, b) for a, b in rec["box"])
        w = np.maximum(ndi.gaussian_filter(m.astype(np.float32), sigma=1.0), m.astype(np.float32))
        x[box] -= (1.0 - alpha) * (rec["mu_lesion"] - rec["mu_ring"]) * w
    return x


def cmd_run(args):
    device = torch.device(args.device)
    image_dir, label_dir, out = Path(args.image_dir), Path(args.label_dir), Path(args.out)
    models = [C.load_model(g, s, device, args.results_root) for g, s in C.parse_runs(args.runs)]
    ref = models[0].predictor
    cases = C.shard(C.read_cases(args.cases_file, image_dir), args.shard)
    for ci, case in enumerate(cases):
        todo = [(m, a) for a in args.alphas for m in models
                if not (out / "pred" / m.name / atag(a) / f"{case}.nii.gz").is_file()]
        if not todo:
            print(f"[{ci + 1}/{len(cases)}] {case}: done, skipped", flush=True)
            continue
        t0 = time.time()
        raw, props, seg_raw = C.read_raw(ref, case, image_dir, label_dir)
        img, gt = raw[0], seg_raw[0]
        lesions = lesion_stats(img, gt)
        log = {"case": case, "lesions": []}
        for rec, m, ring in lesions:
            entry = {k: v for k, v in rec.items() if k != "box"}
            log["lesions"].append(entry)
        for a in args.alphas:
            x = degrade(img, lesions, a)
            # realised CNR after degradation, same masks
            for entry, (rec, m, ring) in zip(log["lesions"], lesions):
                if rec["valid"]:
                    box = tuple(slice(p, q) for p, q in rec["box"])
                    xb = x[box]
                    entry[f"cnr_{atag(a)}"] = abs(float(xb[m].mean()) - float(xb[ring].mean())) / (
                        float(xb[ring].std()) + 1e-8)
            pending = [mm for mm in models if not (out / "pred" / mm.name / atag(a) / f"{case}.nii.gz").is_file()]
            if not pending:
                continue
            data, _, props_pp = C.preprocess(ref, x[None], props)
            for mm in pending:
                logits = C.predict_logits(mm, data)
                C.export_seg(mm, logits, props_pp, out / "pred" / mm.name / atag(a) / f"{case}.nii.gz")
        C.write_json(log, out / "lesions" / f"{case}.json")
        print(f"[{ci + 1}/{len(cases)}] {case}: {len(lesions)} lesions, {len(todo)} predictions, "
              f"{time.time() - t0:.0f}s", flush=True)


def _recall_by_case(df: pd.DataFrame) -> pd.DataFrame:
    """Per case: detected (seed-averaged) and number of lesions."""
    return df.groupby("case").agg(det=("det", "sum"), n=("det", "size"))


def cmd_summarize(args):
    out = Path(args.out)
    label_dir, image_dir = Path(args.label_dir), Path(args.image_dir)
    runs = [(g, s) for g, s in C.parse_runs(args.runs)]
    for g, s in runs:
        for a in args.alphas:
            ev = out / "eval" / f"{g}_s{s}" / atag(a)
            if not (ev / "summary.json").is_file():
                print(f"evaluate {g}_s{s} {atag(a)}", flush=True)
                C.evaluate(out / "pred" / f"{g}_s{s}" / atag(a), ev, label_dir,
                           image_dir if a == 1.0 else None, args.cases_file, args.workers)

    # lesion table: baseline CNR (raw grid, this script's rule) and realised CNR per alpha
    lrows = []
    for f in sorted((out / "lesions").glob("*.json")):
        u = json.loads(f.read_text())
        for l in u["lesions"]:
            lrows.append({"case": u["case"], "lesion": l["lesion"], "voxels": l["voxels"],
                          "valid": l["valid"], "cnr": l.get("cnr", np.nan),
                          **{f"cnr_{atag(a)}": l.get(f"cnr_{atag(a)}", np.nan) for a in args.alphas}})
    les = pd.DataFrame(lrows)
    les["stratum"] = [C.cnr_stratum(c) for c in les.cnr]

    # detection per lesion, per run and alpha (lesion ids = 26-connected components in evaluate.py)
    det = []
    for g, s in runs:
        for a in args.alphas:
            d = pd.read_csv(out / "eval" / f"{g}_s{s}" / atag(a) / "lesions.csv")
            d = d[["case", "lesion", "detected@0.1"]].rename(columns={"detected@0.1": "det"})
            d["group"], d["seed"], d["alpha"] = g, s, a
            det.append(d)
    det = pd.concat(det).merge(les, on=["case", "lesion"], how="left")
    det = det[det.valid.fillna(False) & (det.voxels >= args.min_lesion_voxels)]
    det["det"] = det["det"].astype(float)
    det.to_csv(out / "e12_lesion_detection.csv", index=False)

    # seed-averaged detection per lesion
    lv = det.groupby(["group", "alpha", "case", "lesion", "stratum", "cnr"], as_index=False).det.mean()
    lv["cnr_degraded"] = lv.cnr * lv.alpha
    curve = lv.groupby(["group", "alpha"]).det.agg(recall="mean", n="size").reset_index()
    by_stratum = lv.groupby(["group", "alpha", "stratum"]).det.mean().unstack("stratum").reset_index()
    curve.to_csv(out / "e12_recall_curve.csv", index=False)
    by_stratum.to_csv(out / "e12_recall_by_stratum.csv", index=False)

    # tumor Dice per alpha (seed-averaged per case, Tumor cohort)
    dice = []
    for g, s in runs:
        for a in args.alphas:
            c = pd.read_csv(out / "eval" / f"{g}_s{s}" / atag(a) / "cases.csv")
            c = c[c.has_tumor][["case", "tumor_dice"]]
            c["group"], c["alpha"] = g, a
            dice.append(c)
    dice = pd.concat(dice).groupby(["group", "alpha", "case"], as_index=False).tumor_dice.mean()
    dice.groupby(["group", "alpha"]).tumor_dice.mean().unstack("alpha").to_csv(out / "e12_tumor_dice.csv")

    # interaction: [recall_B(a) - recall_A(a)] - [recall_B(1) - recall_A(1)], case-level bootstrap
    rng = np.random.default_rng(args.seed)
    rows = []
    groups = sorted(lv.group.unique())
    for pair in args.compare:
        b, a_ = pair.split(":")
        if b not in groups or a_ not in groups:
            continue
        cases = sorted(lv.case.unique())
        idx = rng.integers(0, len(cases), size=(args.n_boot, len(cases)))

        def counts(g, al):
            t = lv[(lv.group == g) & (lv.alpha == al)].groupby("case").det.agg(["sum", "size"])
            t = t.reindex(cases).fillna(0.0)
            return t["sum"].to_numpy(), t["size"].to_numpy()

        base_b, base_a = counts(b, 1.0), counts(a_, 1.0)
        for al in args.alphas:
            if al == 1.0:
                continue
            cb, ca = counts(b, al), counts(a_, al)

            def stat(ix):
                rb = cb[0][ix].sum() / cb[1][ix].sum() - base_b[0][ix].sum() / base_b[1][ix].sum()
                ra = ca[0][ix].sum() / ca[1][ix].sum() - base_a[0][ix].sum() / base_a[1][ix].sum()
                return rb - ra

            full = np.arange(len(cases))
            boots = np.array([stat(ix) for ix in idx])
            rows.append({"comparison": f"{b} - {a_}", "alpha": al,
                         "recall_drop_B": cb[0].sum() / cb[1].sum() - base_b[0].sum() / base_b[1].sum(),
                         "recall_drop_A": ca[0].sum() / ca[1].sum() - base_a[0].sum() / base_a[1].sum(),
                         "interaction": stat(full), "ci_lo": float(np.nanpercentile(boots, 2.5)),
                         "ci_hi": float(np.nanpercentile(boots, 97.5))})
    inter = pd.DataFrame(rows)
    inter.to_csv(out / "e12_interaction.csv", index=False)

    # realised CNR check: should equal alpha * CNR inside valid lesions
    chk = {atag(a): float(np.nanmedian(les[f"cnr_{atag(a)}"] / les.cnr)) for a in args.alphas}
    (out / "e12_cnr_check.json").write_text(json.dumps(chk, indent=2))

    print("Recall (IoU>0.1, lesions ≥ %d voxels, seed-averaged):" % args.min_lesion_voxels)
    print(curve.pivot(index="group", columns="alpha", values="recall").round(4).to_string())
    print("\nInteraction (positive = B loses recall more slowly than A):")
    print(inter.round(4).to_string(index=False) if not inter.empty else "(none)")
    print("\nMedian realised CNR / baseline CNR:", chk)

    try:
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for g in groups:
            c = curve[curve.group == g].sort_values("alpha")
            axes[0].plot(c.alpha, c.recall, marker="o", label=g)
        axes[0].set_xlabel("contrast factor α"); axes[0].set_ylabel("lesion recall (IoU > 0.1)")
        axes[0].invert_xaxis(); axes[0].grid(alpha=.3); axes[0].legend()
        bins = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0])
        lv["bin"] = pd.cut(lv.cnr_degraded, bins)
        for g in groups:
            t = lv[lv.group == g].groupby("bin", observed=True).det.mean()
            axes[1].plot([iv.mid for iv in t.index], t.values, marker="o", label=g)
        axes[1].set_xlabel("degraded lesion CNR (α·CNR)"); axes[1].set_ylabel("lesion recall")
        axes[1].grid(alpha=.3); axes[1].legend()
        fig.tight_layout()
        fig.savefig(out / "fig6_degradation.png", dpi=150)
    except Exception as e:  # figures are optional
        print(f"figure skipped: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
        p.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
        p.add_argument("--out", default=str(C.RESULTS_ROOT / "_analysis" / "E12"))
        p.add_argument("--results-root", default=None)
        p.add_argument("--image-dir", default=str(C.RAW_DATASET / "imagesTr"))
        p.add_argument("--label-dir", default=str(C.RAW_DATASET / "labelsTr"))
        p.add_argument("--cases-file", default=str(C.REPO / "experiments" / "eval" / "val_cases.txt"))
    run = sub.choices["run"]
    run.add_argument("--shard", default=None)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    summ = sub.choices["summarize"]
    summ.add_argument("--compare", nargs="+", default=["A6:A1", "A6:A0", "A1:A0"])
    summ.add_argument("--min-lesion-voxels", type=int, default=1)
    summ.add_argument("--n-boot", type=int, default=10_000)
    summ.add_argument("--seed", type=int, default=0)
    summ.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if 1.0 not in args.alphas:
        sys.exit("alphas must include 1.0 (the undegraded reference)")
    {"run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
