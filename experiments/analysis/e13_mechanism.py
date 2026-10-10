#!/usr/bin/env python3
"""
E13 (second half) / Fig. 3 trained panel and Fig. 5: what training changes in
the V2 PFCR models (manuscript Section 4.8, items 1-4).

1. Learned filters per skip level: center wavelength 1/rho_s, bandwidth sigma_s
   and noise multiplier k against their initial values (3.0 / 6.3 / 13.2 voxels,
   0.55, 2.0); read from checkpoint_final.pth, no data needed.
2. PC on features: for every validation lesion (>= --min-voxels voxels in
   preprocessed space) a 128^3 patch centered on it is passed through the
   network; at each skip level the PC map the router sees is compared between
   the lesion boundary shell (3x3x3 dilation minus erosion, as in L_rank) and
   the liver ring at voxel distance 2..6: boundary-to-ring ratio, AUC
   (boundary > ring), fraction of shell voxels with PC > 0. Reported per CNR
   tertile (frozen thresholds).
3. Route maps: mean route (low, mid, high) in tumor / liver / background,
   per-band selectivity |mu(tumor) - mu(background)| (absolute value per band;
   the signed values sum to zero), coefficient of variation over the patch, L1
   distance to the budgets. Validated on synthetic routes (self_test) first.
4. Residual scale tanh(gamma) per level; prior head mean inside tumor vs
   background.

  python experiments/analysis/e13_mechanism.py --runs A6:0,1,2 A7:0
Outputs in OUT: filters.csv, lesion_level_stats.csv, summary.json, summary.md,
fig5_mechanism.png.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

BAND = ("low", "mid", "high")
BUDGET_IN = np.array([0.20, 0.40, 0.40])
BUDGET_OUT = np.array([0.70, 0.22, 0.08])
INIT_LAMBDA = [3.0 * 2.1 ** s for s in range(3)]


# ---------------------------------------------------------------------------
# Statistics on one level
# ---------------------------------------------------------------------------

def route_stats(route: np.ndarray, seg: np.ndarray) -> dict:
    """route [3, *sp]; seg [*sp] with labels 0/1/2 (-1 treated as background)."""
    t, l = seg == 2, seg == 1
    b = ~(t | l)
    rec = {}
    for c, name in enumerate(BAND):
        r = route[c]
        mu = float(r.mean())
        rec[f"{name}_tumor"] = float(r[t].mean()) if t.any() else np.nan
        rec[f"{name}_liver"] = float(r[l].mean()) if l.any() else np.nan
        rec[f"{name}_bg"] = float(r[b].mean()) if b.any() else np.nan
        rec[f"{name}_cv"] = float(r.std() / max(mu, 1e-12))
        rec[f"{name}_sel"] = abs(rec[f"{name}_tumor"] - rec[f"{name}_bg"])
    sel = [rec[f"{n}_sel"] for n in BAND]
    rec["sel_mean_abs"] = float(np.mean(sel))
    rec["sel_l1"] = float(np.sum(sel))
    rec["cv_mean"] = float(np.mean([rec[f"{n}_cv"] for n in BAND]))
    tin = np.array([rec[f"{n}_tumor"] for n in BAND])
    tout = np.array([float(route[c][~t].mean()) for c in range(3)])
    rec["dist_budget_in"] = float(np.abs(tin - BUDGET_IN).sum())
    rec["dist_budget_out"] = float(np.abs(tout - BUDGET_OUT).sum())
    return rec


def pc_boundary_stats(pc: np.ndarray, lesion: np.ndarray, liver: np.ndarray, other: np.ndarray) -> dict:
    """Boundary shell (dilate3 - erode3) vs liver ring at voxel distance 2..6."""
    st = np.ones((3, 3, 3), bool)
    shell = ndi.binary_dilation(lesion, st) & ~ndi.binary_erosion(lesion, st, border_value=0)
    d_out = ndi.distance_transform_edt(~lesion)
    ring = (~lesion) & liver & (d_out >= 2) & (d_out <= 6) & ~other
    if shell.sum() < 5 or ring.sum() < 5:
        return {}
    pb, pr = pc[shell], pc[ring]
    if np.all(pb == pb[0]) and np.all(pr == pb[0]):
        auc = 0.5
    else:
        auc = float(mannwhitneyu(pb, pr, alternative="two-sided").statistic / (len(pb) * len(pr)))
    return {"pc_shell": float(pb.mean()), "pc_ring": float(pr.mean()),
            "ratio": float(pb.mean() / pr.mean()) if pr.mean() > 0 else np.nan,
            "auc": auc, "shell_nonzero": float((pb > 0).mean()), "ring_nonzero": float((pr > 0).mean()),
            "n_shell": int(shell.sum()), "n_ring": int(ring.sum())}


def self_test():
    seg = np.zeros((16, 16, 16), np.int8)
    seg[2:14, 2:14, 2:14] = 1
    seg[5:11, 5:11, 5:11] = 2
    route = np.empty((3,) + seg.shape, np.float32)
    for c in range(3):
        route[c] = np.where(seg == 2, BUDGET_IN[c], BUDGET_OUT[c])
    r = route_stats(route, seg)
    want = np.abs(BUDGET_IN - BUDGET_OUT)
    assert np.allclose([r[f"{n}_sel"] for n in BAND], want, atol=1e-6)
    assert abs(r["dist_budget_in"]) < 1e-6 and abs(r["dist_budget_out"]) < 1e-6
    assert abs(np.sum([r[f"{n}_tumor"] - r[f"{n}_bg"] for n in BAND])) < 1e-6  # signed sum is 0
    lesion = seg == 2
    pc = np.zeros(seg.shape, np.float32)
    st = np.ones((3, 3, 3), bool)
    pc[ndi.binary_dilation(lesion, st) & ~ndi.binary_erosion(lesion, st)] = 1.0
    p = pc_boundary_stats(pc, lesion, seg == 1, np.zeros_like(lesion))
    assert p["auc"] == 1.0 and p["shell_nonzero"] == 1.0, p
    print("[self-test] route / PC statistics recover known values")


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def filter_table(model: C.Model) -> list[dict]:
    rows = []
    for lvl, m in enumerate(model.network.pfcr_modules):
        rho = (0.5 * torch.sigmoid(m.pc.rho_raw)).detach().cpu().numpy()
        sig = (0.05 + 0.9 * torch.sigmoid(m.pc.sigma_raw)).detach().cpu().numpy()
        k = float(F.softplus(m.pc.noise_k).detach())
        g = float(torch.tanh(m.residual_gate).detach())
        for s in range(len(rho)):
            lam = 1.0 / float(rho[s])
            rows.append({"run": model.name, "level": lvl, "scale": s, "lambda": lam,
                         "lambda_init": INIT_LAMBDA[s], "rel_shift": abs(lam - INIT_LAMBDA[s]) / INIT_LAMBDA[s],
                         "sigma": float(sig[s]), "k": k, "tanh_gamma": g})
    return rows


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------

def centered_patch(arr: np.ndarray, center, size: int, fill):
    """arr [..., Z, Y, X] -> patch of size^3 around center (padded with fill)."""
    sp = arr.shape[-3:]
    out = np.full(arr.shape[:-3] + (size,) * 3, fill, dtype=arr.dtype)
    src, dst = [], []
    for c, n in zip(center, sp):
        lo = int(c) - size // 2
        a, b = max(lo, 0), min(lo + size, n)
        src.append(slice(a, b))
        dst.append(slice(a - lo, b - lo))
    out[(..., *dst)] = arr[(..., *src)]
    return out


@torch.no_grad()
def forward_patch(model: C.Model, x: np.ndarray, device):
    net = model.network
    t = torch.from_numpy(x)[None].to(device).float()
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        net(t)
    pcs = [m._last_pc[0, 0].float().cpu().numpy() for m in net.pfcr_modules]
    routes = [m._last_route[0].float().cpu().numpy() for m in net.pfcr_modules]
    prior = net._last_tumor_prior
    prior = prior[0, 0].float().cpu().numpy() if prior is not None else None
    return pcs, routes, prior


def down(mask: np.ndarray, shape) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(t, size=tuple(shape), mode="nearest")[0, 0].numpy()


def md(df: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["A6:0,1,2", "A7:0"])
    ap.add_argument("--out", default=str(C.RESULTS_ROOT / "_analysis" / "E13"))
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--image-dir", default=str(C.RAW_DATASET / "imagesTr"))
    ap.add_argument("--label-dir", default=str(C.RAW_DATASET / "labelsTr"))
    ap.add_argument("--cases-file", default=str(C.REPO / "experiments" / "eval" / "val_cases.txt"))
    ap.add_argument("--min-voxels", type=int, default=50)
    ap.add_argument("--patch", type=int, default=None, help="default: plan patch size")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    self_test()

    device = torch.device(args.device)
    out = Path(args.out)
    models = [C.load_model(g, s, device, args.results_root) for g, s in C.parse_runs(args.runs)]
    models = [m for m in models if m.has_pfcr]
    if not models:
        sys.exit("no PFCR runs given")
    ref = models[0].predictor
    patch = args.patch or int(ref.configuration_manager.patch_size[0])

    filt = pd.DataFrame([r for m in models for r in filter_table(m)])
    out.mkdir(parents=True, exist_ok=True)
    filt.to_csv(out / "filters.csv", index=False)

    rows = []
    cases = C.read_cases(args.cases_file, Path(args.image_dir))
    for ci, case in enumerate(cases):
        t0 = time.time()
        raw, props, seg_raw = C.read_raw(ref, case, Path(args.image_dir), Path(args.label_dir))
        data, seg, _ = C.preprocess(ref, raw, props, seg_raw)
        lab, lesions = C.lesion_cnr_preprocessed(data, seg, min_voxels=args.min_voxels)
        s0 = seg[0]
        fill = float(data.min())
        for les in lesions:
            center = [int((a + b) // 2) for a, b in les["bbox"]]
            x = centered_patch(data, center, patch, fill)
            sp = centered_patch(s0, center, patch, -1)
            lp = centered_patch(lab, center, patch, 0) == les["label"]
            for m in models:
                pcs, routes, prior = forward_patch(m, x, device)
                base = {"run": m.name, "case": case, "lesion": les["label"], "voxels": les["voxels"],
                        "cnr": les["cnr"], "stratum": les["stratum"]}
                if prior is not None:
                    pr = down(sp == 2, prior.shape) > 0.5
                    base["prior_tumor"] = float(prior[pr].mean()) if pr.any() else np.nan
                    base["prior_bg"] = float(prior[~pr].mean())
                for lvl, (pc, route) in enumerate(zip(pcs, routes)):
                    shp = pc.shape
                    seg_l = down(sp, shp).astype(np.int8)
                    les_l = down(lp, shp) > 0.5
                    if not les_l.any():
                        continue
                    other = (seg_l == 2) & ~les_l
                    rec = dict(base, level=lvl, **pc_boundary_stats(pc, les_l, seg_l == 1, other))
                    r_seg = down(sp, route.shape[1:]).astype(np.int8)
                    rec.update(route_stats(route, r_seg))
                    rows.append(rec)
        print(f"[{ci + 1}/{len(cases)}] {case}: {len(lesions)} lesions, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "lesion_level_stats.csv", index=False)

    # aggregate: per run x level x stratum, then mean over runs of the same group
    df["group"] = df.run.str.split("_s").str[0]
    pc_tab = (df.groupby(["group", "run", "level", "stratum"])
              .agg(ratio=("ratio", "median"), auc=("auc", "median"), shell_nonzero=("shell_nonzero", "median"),
                   n=("lesion", "size")).reset_index()
              .groupby(["group", "level", "stratum"]).agg(ratio=("ratio", "mean"), auc=("auc", "mean"),
                                                          shell_nonzero=("shell_nonzero", "mean"),
                                                          n=("n", "mean")).reset_index())
    route_cols = ["sel_mean_abs", "sel_l1", "cv_mean", "dist_budget_in", "dist_budget_out"] + \
                 [f"{b}_{r}" for r in ("tumor", "liver", "bg") for b in BAND]
    route_tab = df.groupby(["group", "level"])[route_cols].mean().reset_index()
    filt_tab = filt.assign(group=filt.run.str.split("_s").str[0]).groupby(["group", "level", "scale"]).agg(
        lambda_mean=("lambda", "mean"), lambda_sd=("lambda", "std"), lambda_init=("lambda_init", "first"),
        rel_shift=("rel_shift", "mean"), sigma=("sigma", "mean"), k=("k", "mean"),
        tanh_gamma=("tanh_gamma", "mean")).reset_index()
    prior_tab = df.drop_duplicates(["run", "case", "lesion"]).groupby("group")[
        ["prior_tumor", "prior_bg"]].mean() if "prior_tumor" in df else pd.DataFrame()
    pc_tab.to_csv(out / "pc_by_level_stratum.csv", index=False)
    route_tab.to_csv(out / "route_by_level.csv", index=False)
    filt_tab.to_csv(out / "filters_by_level.csv", index=False)
    C.write_json({"runs": [m.name for m in models], "patch": patch, "min_voxels": args.min_voxels,
                  "n_lesion_patches": int(df.drop_duplicates(["case", "lesion"]).shape[0]),
                  "prior": prior_tab.to_dict() if not prior_tab.empty else None}, out / "summary.json")
    with open(out / "summary.md", "w") as f:
        f.write("# E13 mechanism analysis (V2 models)\n\n## Learned filters (mean over seeds)\n\n")
        f.write(md(filt_tab))
        f.write("\n\n## PC on features: boundary vs liver ring, per level and CNR tertile\n\n")
        f.write(md(pc_tab))
        f.write("\n\n## Route statistics per level\n\n")
        f.write(md(route_tab))
        if not prior_tab.empty:
            f.write("\n\n## Prior head\n\n" + md(prior_tab.reset_index()))
        f.write("\n")
    print((out / "summary.md").read_text())

    try:
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for s in range(3):
            t = filt[filt.scale == s]
            axes[0].scatter(t.level + 0.1 * s, t["lambda"], label=f"scale {s} (init {INIT_LAMBDA[s]:.1f})")
            axes[0].axhline(INIT_LAMBDA[s], ls="--", lw=.7, color=f"C{s}")
        axes[0].set_xlabel("skip level"); axes[0].set_ylabel("center wavelength (voxels)"); axes[0].legend(fontsize=7)
        t = filt.drop_duplicates(["run", "level"])
        axes[1].scatter(t.level, t.tanh_gamma, c="C3", label="tanh(γ)")
        axes[1].scatter(t.level, t.k / 10, c="C4", marker="x", label="k / 10")
        axes[1].set_xlabel("skip level"); axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
        for st, mk in zip(("low", "mid", "high"), "ov^"):
            t = pc_tab[(pc_tab.stratum == st)]
            for g in t.group.unique():
                tt = t[t.group == g]
                axes[2].plot(tt.level, tt.auc, marker=mk, label=f"{g} {st}")
        axes[2].axhline(0.5, ls="--", color="k", lw=.7)
        axes[2].set_xlabel("skip level"); axes[2].set_ylabel("AUC boundary vs ring (PC)"); axes[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fig5_mechanism.png", dpi=150)
    except Exception as e:
        print(f"figure skipped: {e}")


if __name__ == "__main__":
    main()
