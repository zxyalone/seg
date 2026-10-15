#!/usr/bin/env python3
"""
训练侧 pilot 的 MSD08 指标与继续条件（docs/训练侧频域计划.md §4，事先登记）。

  metrics  用 test_protocol 的冻结口径（4 mm 后处理、26 连通、IoU > 0.1）为给定 run 生成 MSD08 表格；
           对照 A13b_s0 一并重算，保证同一份代码。
  decide   各 pilot run 相对 A13b_s0 的继续条件：
             B13：< 10 mm 病灶召回（全部 CNR）≥ +4 个百分点；低 CNR 召回 ≥ −2；FP/scan ≤ +0.3
             E13：FP/scan ≤ −0.3；低 CNR 召回 ≥ −2 个百分点；逐病例肿瘤 Dice ≥ −1.0 个百分点

  python experiments/analysis/train_side_pilot.py metrics --runs A13b_s0 E13_s0 --out results/train_side/pilot_msd08
  python experiments/analysis/train_side_pilot.py decide --tables results/train_side/pilot_msd08
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REF = "A13b_s0"
CRITERIA = {
    "B13": dict(small_recall=4.0, low_recall=-2.0, fp_scan=0.3),
    "E13": dict(fp_scan=-0.3, low_recall=-2.0, tumor_dice=-1.0),
}


def summary(L, C, run):
    l, c = L[L.run == run], C[C.run == run]
    return dict(n_lesions=int(len(l)), recall_all=100 * float(l.detected.mean()),
                recall_small=100 * float(l[l.size_stratum == "<10"].detected.mean()),
                recall_low=100 * float(l[l.cnr_stratum == "low"].detected.mean()),
                fp_scan=float(c.fp.mean()), tumor_dice=100 * float(c[c.has_tumor].tumor_dice.mean()))


def check(group, s, ref):
    d = {k: s[k] - ref[k] for k in ("recall_small", "recall_low", "fp_scan", "tumor_dice")}
    cr = CRITERIA[group]
    conds = {}
    if "small_recall" in cr:
        conds["recall_small >= +4"] = d["recall_small"] >= cr["small_recall"]
    if group == "B13":
        conds["fp_scan <= +0.3"] = d["fp_scan"] <= cr["fp_scan"]
    else:
        conds["fp_scan <= -0.3"] = d["fp_scan"] <= cr["fp_scan"]
    conds["recall_low >= -2"] = d["recall_low"] >= cr["low_recall"]
    if "tumor_dice" in cr:
        conds["tumor_dice >= -1"] = d["tumor_dice"] >= cr["tumor_dice"]
    conds = {k: bool(v) for k, v in conds.items()}
    return {k: float(v) for k, v in d.items()}, conds, all(conds.values())


def cmd_metrics(a):
    import test_protocol as TP
    import external_msd08_protocol  # noqa: F401  registers the 'external_msd08' split
    TP.GROUPS, TP.DESCRIPTIVE = (), tuple(a.runs)
    TP.cmd_metrics(argparse.Namespace(split="external_msd08", results_root=a.results_root, out=a.out, jobs=a.jobs))


def cmd_decide(a):
    import pandas as pd
    tab = Path(a.tables)
    L, C = pd.read_csv(tab / "lesions_pp4mm.csv"), pd.read_csv(tab / "cases_pp4mm.csv")
    ref = summary(L, C, REF)
    res, lines = dict(reference=ref, runs={}), ["# 训练侧 pilot（MSD08，相对 A13b_s0；计划 §4）", "",
                                                  "| run | < 10 mm 召回 | 低 CNR 召回 | FP/scan | 肿瘤 Dice | 条件 | 继续 |",
                                                  "|---|---|---|---|---|---|---|",
                                                  f"| {REF} | {ref['recall_small']:.1f} | {ref['recall_low']:.1f} | "
                                                  f"{ref['fp_scan']:.2f} | {ref['tumor_dice']:.1f} | 对照 | — |"]
    for run in sorted(set(L.run) - {REF}):
        group = run.rsplit("_s", 1)[0]
        if group not in CRITERIA:
            continue
        s = summary(L, C, run)
        d, conds, go = check(group, s, ref)
        res["runs"][run] = dict(summary=s, delta=d, conditions=conds, CONTINUE=go)
        lines.append(f"| {run} | {s['recall_small']:.1f} ({d['recall_small']:+.1f}) | {s['recall_low']:.1f} "
                     f"({d['recall_low']:+.1f}) | {s['fp_scan']:.2f} ({d['fp_scan']:+.2f}) | {s['tumor_dice']:.1f} "
                     f"({d['tumor_dice']:+.1f}) | {'; '.join(f'{k}: {v}' for k, v in conds.items())} | **{go}** |")
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
