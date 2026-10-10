#!/usr/bin/env python3
"""
E11 / Table 6: inference-time interventions on the trained A6 models.

Conditions (manuscript Section 4.9; skip level 0 = 128^3 = "level 1" in the paper):
  none            unmodified model (also checks that this reproduces eval_val)
  pc_zero@L       PC entering the router at skip level L set to 0, L = 0..4
  pc_zero@all     PC set to 0 at every level
  route_uniform   route fixed to (1/3, 1/3, 1/3) at every voxel and level
  prior_zero      tumor prior entering every router set to 0
  gate_zero       PFCR off: routed residual removed (out = x) at every level

For each case the image is preprocessed once and every model / condition is
run on the cached array. Outputs per model and condition:
  OUT/pred/<run>/<cond>/<case>.nii.gz     segmentation (raw grid)
  OUT/utility/<run>/<case>.json           per-lesion soft-Dice loss L_j per condition
U_j = L_j(intervened) - L_j(none); L_j is the soft-Dice loss of the tumor
probability against the tumor label inside the lesion's bounding box enlarged
by --box-margin voxels, in preprocessed space. Positive U_j: the removed signal
helped lesion j. Lesion CNR uses the frozen tertiles (common.lesion_cnr_preprocessed).

  # run (one shard per GPU), then summarize once all shards are done
  CUDA_VISIBLE_DEVICES=0 python experiments/analysis/e11_interventions.py run --shard 0/3 &
  CUDA_VISIBLE_DEVICES=1 python experiments/analysis/e11_interventions.py run --shard 1/3 &
  CUDA_VISIBLE_DEVICES=2 python experiments/analysis/e11_interventions.py run --shard 2/3 &
  wait; python experiments/analysis/e11_interventions.py summarize
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

CONDITIONS = {
    "none": ("none", None),
    **{f"pc_zero@{l}": ("pc_zero", [l]) for l in range(5)},
    "pc_zero@all": ("pc_zero", None),
    "route_uniform": ("route_uniform", None),
    "prior_zero": ("prior_zero", None),
    "gate_zero": ("gate_zero", None),
}
LABELS = {"none": "None (A6)", **{f"pc_zero@{l}": f"Zero PC, level {l + 1}" for l in range(5)},
          "pc_zero@all": "Zero PC, all levels", "route_uniform": "Uniform route (1/3, 1/3, 1/3)",
          "prior_zero": "Zero prior", "gate_zero": "PFCR off (out = x)"}


def soft_dice_loss_in_box(prob: np.ndarray, gt: np.ndarray, bbox, margin: int) -> float:
    sl = tuple(slice(max(a - margin, 0), min(b + margin, n)) for (a, b), n in zip(bbox, prob.shape))
    p, g = prob[sl].astype(np.float64), gt[sl].astype(np.float64)
    return float(1.0 - 2.0 * (p * g).sum() / (p.sum() + g.sum() + 1e-8))


def cmd_run(args):
    device = torch.device(args.device)
    image_dir, label_dir = Path(args.image_dir), Path(args.label_dir)
    out = Path(args.out)
    runs = C.parse_runs(args.runs)
    models = [C.load_model(g, s, device, args.results_root) for g, s in runs]
    for m in models:
        if not m.has_pfcr:
            sys.exit(f"{m.name} has no PFCR modules; E11 applies to PFCR groups only")
    conds = args.conditions or list(CONDITIONS)
    cases = C.shard(C.read_cases(args.cases_file, image_dir), args.shard)
    ref = models[0].predictor
    for ci, case in enumerate(cases):
        todo = [(m, c) for m in models for c in conds
                if not (out / "pred" / m.name / c / f"{case}.nii.gz").is_file()
                or not (out / "utility" / m.name / f"{case}.json").is_file()]
        if not todo:
            print(f"[{ci + 1}/{len(cases)}] {case}: done, skipped", flush=True)
            continue
        t0 = time.time()
        raw, props, seg_raw = C.read_raw(ref, case, image_dir, label_dir)
        data, seg, props_pp = C.preprocess(ref, raw, props, seg_raw)
        lab, lesions = C.lesion_cnr_preprocessed(data, seg)
        gt_tumor = (seg[0] == 2)
        for m in models:
            ufile = out / "utility" / m.name / f"{case}.json"
            util = json.loads(ufile.read_text()) if ufile.is_file() else {
                "case": case, "lesions": [{k: v for k, v in l.items() if k != "bbox"} for l in lesions],
                "loss": {}}
            for c in conds:
                pfile = out / "pred" / m.name / c / f"{case}.nii.gz"
                if pfile.is_file() and c in util["loss"]:
                    continue
                mode, levels = CONDITIONS[c]
                C.set_intervention(m, mode, levels)
                logits = C.predict_logits(m, data)
                prob = torch.softmax(logits, 0)[2].numpy()
                util["loss"][c] = [soft_dice_loss_in_box(prob, gt_tumor, l["bbox"], args.box_margin)
                                   for l in lesions]
                C.export_seg(m, logits, props_pp, pfile)
            C.set_intervention(m, "none")
            C.write_json(util, ufile)
        print(f"[{ci + 1}/{len(cases)}] {case}: {len(lesions)} lesions, {len(todo)} predictions, "
              f"{time.time() - t0:.0f}s", flush=True)


def cmd_summarize(args):
    sys.path.insert(0, str(C.REPO / "experiments" / "eval"))
    import stats as S

    out = Path(args.out)
    runs = [f"{g}_s{s}" for g, s in C.parse_runs(args.runs)]
    conds = args.conditions or list(CONDITIONS)
    label_dir, image_dir = Path(args.label_dir), Path(args.image_dir)

    # 1) evaluate every (run, condition) prediction folder
    for r in runs:
        for c in conds:
            ev = out / "eval" / r / c
            if not (ev / "summary.json").is_file():
                print(f"evaluate {r} {c}", flush=True)
                C.evaluate(out / "pred" / r / c, ev, label_dir,
                           image_dir if c == "none" else None, args.cases_file, args.workers)

    # 2) sanity: 'none' must reproduce the end-of-training validation
    check = []
    for r in runs:
        g, s = r.split("_s")
        ref = C.model_dir(g, int(s), args.results_root) / "fold_0" / "eval_val" / "summary.json"
        if not ref.is_file() and args.results_root is None:
            ref = C.REPO / "results" / r / "eval_val" / "summary.json"
        if ref.is_file():
            a = json.loads(ref.read_text())["Tumor"]["tumor_dice_mean"]
            b = json.loads((out / "eval" / r / "none" / "summary.json").read_text())["Tumor"]["tumor_dice_mean"]
            check.append({"run": r, "eval_val_tumor_dice": a, "e11_none_tumor_dice": b, "abs_diff": abs(a - b)})
    if check:
        pd.DataFrame(check).to_csv(out / "check_none_vs_eval_val.csv", index=False)
        print(pd.DataFrame(check).to_string(index=False))

    # 3) paired comparisons of each condition against 'none' (seeds averaged per case)
    rng = np.random.default_rng(args.seed)
    ref = S.load_method([str(out / "eval" / r / "none") for r in runs], args.cohort, "0.1")
    rows = []
    for c in conds:
        if c == "none":
            continue
        cur = S.load_method([str(out / "eval" / r / c) for r in runs], args.cohort, "0.1")
        for rec in S.compare(ref, cur, "0.1", args.n_boot, rng):
            rows.append({"condition": c, **rec})
    cmp = pd.DataFrame(rows)
    cmp["p_holm"] = np.nan
    for _, g in cmp.groupby("metric"):
        cmp.loc[g.index, "p_holm"] = S.holm(g.p_wilcoxon.tolist())
    cmp.to_csv(out / "e11_paired_vs_none.csv", index=False)

    # 4) per-lesion utility, averaged over seeds per lesion, median per CNR tertile
    recs = []
    for r in runs:
        for f in sorted((out / "utility" / r).glob("*.json")):
            u = json.loads(f.read_text())
            base = u["loss"].get("none")
            if base is None:
                continue
            for c, losses in u["loss"].items():
                for l, lc, lb in zip(u["lesions"], losses, base):
                    recs.append({"run": r, "case": u["case"], "lesion": l["label"], "voxels": l["voxels"],
                                 "cnr": l["cnr"], "stratum": l["stratum"], "condition": c, "U": lc - lb})
    util = pd.DataFrame(recs)
    util.to_csv(out / "e11_lesion_utility.csv", index=False)
    per_lesion = (util[util.voxels >= args.min_lesion_voxels]
                  .groupby(["condition", "case", "lesion", "stratum"], as_index=False).U.mean())
    med = per_lesion.groupby(["condition", "stratum"]).U.agg(["median", "count"]).unstack("stratum")

    # 5) Table 6
    def cell(c, metric, scale=100.0, fmt="{:+.2f}"):
        r = cmp[(cmp.condition == c) & (cmp.metric == metric)]
        if r.empty:
            return "–"
        r = r.iloc[0]
        return f"{fmt.format(r.delta * scale)} [{fmt.format(r.ci_lo * scale)}, {fmt.format(r.ci_hi * scale)}]"

    lines = ["| Intervention | Δ Tumor DSC (pp) [95% CI] | Δ Lesion recall (pp) [95% CI] | Δ FP/scan [95% CI] "
             "| Median U_j low | mid | high |", "|---|---|---|---|---|---|---|"]
    for c in conds:
        if c == "none":
            lines.append(f"| {LABELS[c]} | 0 | 0 | 0 | – | – | – |")
            continue
        us = []
        for st in ("low", "mid", "high"):
            try:
                us.append(f"{med.loc[c, ('median', st)]:+.4f} (n={int(med.loc[c, ('count', st)])})")
            except KeyError:
                us.append("–")
        lines.append(f"| {LABELS[c]} | {cell(c, 'tumor_dice')} | {cell(c, 'lesion_recall@0.1')} | "
                     f"{cell(c, 'fp_per_scan@0.1', 1.0, '{:+.3f}')} | " + " | ".join(us) + " |")
    note = (f"\nRuns: {', '.join(runs)}; cohort {args.cohort}; per-case values averaged over seeds; "
            f"{args.n_boot} case-level bootstrap resamples. U_j over lesions ≥ {args.min_lesion_voxels} "
            f"voxels (preprocessed space), averaged over seeds per lesion.\n")
    (out / "table6_interventions.md").write_text("\n".join(lines) + "\n" + note)
    print("\n".join(lines) + note)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--runs", nargs="+", default=["A6:0,1,2"])
        p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=None)
        p.add_argument("--out", default=str(C.RESULTS_ROOT / "_analysis" / "E11"))
        p.add_argument("--results-root", default=None)
        p.add_argument("--image-dir", default=str(C.RAW_DATASET / "imagesTr"))
        p.add_argument("--label-dir", default=str(C.RAW_DATASET / "labelsTr"))
        p.add_argument("--cases-file", default=str(C.REPO / "experiments" / "eval" / "val_cases.txt"))
    run = sub.choices["run"]
    run.add_argument("--shard", default=None, help="i/n: this process handles cases i, i+n, ...")
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--box-margin", type=int, default=10)
    summ = sub.choices["summarize"]
    summ.add_argument("--cohort", default="Tumor", choices=("Tumor", "All"))
    summ.add_argument("--n-boot", type=int, default=10_000)
    summ.add_argument("--seed", type=int, default=0)
    summ.add_argument("--min-lesion-voxels", type=int, default=50)
    summ.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
