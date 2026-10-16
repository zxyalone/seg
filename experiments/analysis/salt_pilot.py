#!/usr/bin/env python3
"""
SALT pilot: MSD08 metrics and the registered continue conditions (docs/SALT模型设计.md §4). Metrics come from the
frozen test-protocol code (4-mm post-processing, 26-connectivity, IoU > 0.1; train_side_pilot.cmd_metrics), with
A0_s0 (VoComni nnU-Net, no SALT module) recomputed by the same code as the reference.

  Q1-Q4 (lesion query decoder): benefit = pooled lesion F1 >= +3.0 pp or FP/scan <= -0.5, and per-case tumour
          Dice >= +1.0 pp; guards: all-lesion recall >= -2.0 pp, low-CNR recall >= -3.0 pp.
  F1 (FAF-3D only):        per-case tumour Dice >= +1.0 pp; guards: F1 >= -2.0 pp, all-lesion recall >= -2.0 pp.

  python experiments/analysis/salt_pilot.py metrics --runs A0_s0 Q4_s0 --out results/salt/pilot_msd08_Q4_s0
  python experiments/analysis/salt_pilot.py decide --tables results/salt/pilot_msd08_Q4_s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REF = "A0_s0"
QUERY_GROUPS = ("Q1", "Q2", "Q3", "Q4")
KEYS = ("f1", "fp_scan", "tumor_dice", "recall_all", "recall_low", "recall_small")


def summary(L, C, run):
    l, c = L[L.run == run], C[C.run == run]
    tp, fn, fp = int(l.detected.sum()), int((~l.detected.astype(bool)).sum()), int(c.fp.sum())
    return dict(n_lesions=int(len(l)), f1=100 * 2 * tp / max(2 * tp + fp + fn, 1),
                recall_all=100 * float(l.detected.mean()),
                recall_small=100 * float(l[l.size_stratum == "<10"].detected.mean()),
                recall_low=100 * float(l[l.cnr_stratum == "low"].detected.mean()),
                fp_scan=float(c.fp.mean()), tumor_dice=100 * float(c[c.has_tumor].tumor_dice.mean()))


def check(group, s, ref):
    d = {k: s[k] - ref[k] for k in KEYS}
    if group in QUERY_GROUPS:
        conds = {"benefit (F1 +3 or FP -0.5)": d["f1"] >= 3.0 or d["fp_scan"] <= -0.5,
                 "tumour Dice >= +1": d["tumor_dice"] >= 1.0,
                 "recall_all >= -2": d["recall_all"] >= -2.0, "recall_low >= -3": d["recall_low"] >= -3.0}
    elif group == "F1":
        conds = {"tumour Dice >= +1": d["tumor_dice"] >= 1.0, "F1 >= -2": d["f1"] >= -2.0,
                 "recall_all >= -2": d["recall_all"] >= -2.0}
    else:
        raise ValueError(group)
    conds = {k: bool(v) for k, v in conds.items()}
    return {k: float(v) for k, v in d.items()}, conds, all(conds.values())


def cmd_metrics(a):
    from train_side_pilot import cmd_metrics as metrics
    metrics(a)


def cmd_decide(a):
    import pandas as pd
    tab = Path(a.tables)
    L, C = pd.read_csv(tab / "lesions_pp4mm.csv"), pd.read_csv(tab / "cases_pp4mm.csv")
    ref = summary(L, C, REF)
    res = dict(reference=ref, runs={})
    lines = ["# SALT pilot（MSD08，相对 A0_s0；docs/SALT模型设计.md §4）", "",
             "| run | F1 | FP/scan | 肿瘤 Dice | 全部召回 | 低 CNR 召回 | < 10 mm 召回 | 条件 | 继续 |",
             "|---|---|---|---|---|---|---|---|---|",
             f"| {REF} | {ref['f1']:.1f} | {ref['fp_scan']:.2f} | {ref['tumor_dice']:.1f} | {ref['recall_all']:.1f} | "
             f"{ref['recall_low']:.1f} | {ref['recall_small']:.1f} | 对照 | — |"]
    for run in sorted(set(L.run) - {REF}):
        group = run.rsplit("_s", 1)[0]
        if group not in QUERY_GROUPS + ("F1",):
            continue
        s = summary(L, C, run)
        d, conds, go = check(group, s, ref)
        res["runs"][run] = dict(summary=s, delta=d, conditions=conds, CONTINUE=go)
        cells = " | ".join(f"{s[k]:.{2 if k == 'fp_scan' else 1}f} ({d[k]:+.{2 if k == 'fp_scan' else 1}f})"
                           for k in ("f1", "fp_scan", "tumor_dice", "recall_all", "recall_low", "recall_small"))
        lines.append(f"| {run} | {cells} | {'; '.join(f'{k}: {v}' for k, v in conds.items())} | **{go}** |")
    (tab / "decision.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    (tab / "DECISION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("metrics")
    m.add_argument("--runs", nargs="+", required=True)
    m.add_argument("--results-root", default="/root/autodl-tmp/nnUNet_data/nnUNet_results_v2")
    m.add_argument("--out", required=True)
    m.add_argument("--jobs", type=int, default=4)
    d = sub.add_parser("decide")
    d.add_argument("--tables", required=True)
    a = ap.parse_args()
    {"metrics": cmd_metrics, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
