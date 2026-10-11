#!/usr/bin/env python3
"""验证集失败模式诊断：A0/A1/A6/A7 的提升来自哪里，漏检集中在哪里。

只读 results/<run>/eval_val/{cases,lesions,pred_lesions}.csv（21 例 fold-0 验证集，其中 18 例含肿瘤），
不需要 GPU 和影像。输出的数字写在 docs/方法重设计_可检出性计划.md §1。

  python experiments/analysis/val_failure_modes.py [--results results]

各节：
  1  假阳性连通域按直径分层（每个 run 合计，再按组取均值）
  2  最小连通域后处理的模拟：去掉直径 < t mm 的预测连通域后的 FP/scan、召回、精确率、F1、近似 Dice。
     阈值在同一批验证病例上扫描，结果偏乐观，只用来说明量级；Dice 按连通域体素近似。
  3  18 例含肿瘤病例的体素分解（TP / FP / FN，先对 seed 取平均再求和）
  4  逐病例肿瘤 Dice（seed 平均）及相对 A0 的差值，按 A6 − A0 排序
  5  按大小、CNR 分层的病灶召回与匹配病灶 IoU
  6  A0 三个 seed 都漏掉的病灶
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ("A0", "A1", "A6", "A7")


def load(results: Path, name: str) -> pd.DataFrame:
    frames = []
    for d in sorted(results.glob("A*_s*/eval_val")):
        run = d.parent.name
        if run.split("_")[0] not in GROUPS:
            continue
        frames.append(pd.read_csv(d / name).assign(run=run, group=run.split("_")[0]))
    return pd.concat(frames, ignore_index=True)


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[2] / "results"))
    args = ap.parse_args()
    res = Path(args.results)
    pd.set_option("display.width", 200)
    C, L, P = load(res, "cases.csv"), load(res, "lesions.csv"), load(res, "pred_lesions.csv")
    tumor_cases = set(C.loc[C.has_tumor, "case"])
    n_all, n_tum = C.case.nunique(), len(tumor_cases)
    print(f"runs: {sorted(C.run.unique())}；病例 {n_all}（含肿瘤 {n_tum}）")

    section("1 假阳性连通域按直径分层（每个 run 21 例合计，组内取均值）")
    fp = P[P["fp@0.1"]].copy()
    fp["size"] = pd.cut(fp.diameter_mm, [0, 5, 10, 20, np.inf], labels=["<5", "5-10", "10-20", ">20"], right=False)
    per_run = fp.groupby(["group", "run", "size"], observed=True).size().unstack(fill_value=0)
    per_run["total"] = per_run.sum(1)
    print(per_run)
    print("\n组均值：")
    print(per_run.groupby(level="group").mean().round(1))

    section("2 最小连通域后处理模拟（直径 < t mm 的预测连通域删除；18 例含肿瘤病例；阈值在同批数据上扫描）")
    rows = []
    for run, Pr in P.groupby("run"):
        Cr = C[(C.run == run) & C.has_tumor].set_index("case")
        n_gt = int(L[(L.run == run) & L.case.isin(tumor_cases)].shape[0])
        for t in (0, 3, 4, 5, 6, 8):
            keep, rem = Pr[Pr.diameter_mm >= t], Pr[Pr.diameter_mm < t]
            tp = int((~keep["fp@0.1"] & keep.case.isin(tumor_cases)).sum())
            fpn = int((keep["fp@0.1"] & keep.case.isin(tumor_cases)).sum())
            rfp = rem[rem["fp@0.1"]].groupby("case").voxels.sum().reindex(Cr.index).fillna(0)
            rtp = rem[~rem["fp@0.1"]].groupby("case").voxels.sum().reindex(Cr.index).fillna(0).clip(upper=Cr.tumor_tp)
            tpv, fpv = Cr.tumor_tp - rtp, Cr.tumor_fp - rfp
            fnv = Cr.tumor_fn + rtp
            rec, prec = tp / n_gt, tp / max(tp + fpn, 1)
            rows.append(dict(t_mm=t, group=run.split("_")[0], run=run,
                             dice_approx=100 * (2 * tpv / (2 * tpv + fpv + fnv)).mean(),
                             recall=100 * rec, precision=100 * prec, f1=100 * 2 * rec * prec / max(rec + prec, 1e-9),
                             fp_per_scan=fpn / n_tum))
    D = pd.DataFrame(rows)
    print(D.groupby(["t_mm", "group"])[["dice_approx", "recall", "precision", "f1", "fp_per_scan"]].mean().round(2))

    section("3 体素分解：18 例含肿瘤病例合计（seed 平均，单位千体素）")
    Ct = C[C.has_tumor]
    agg = Ct.groupby(["group", "case"])[["tumor_tp", "tumor_fp", "tumor_fn", "tumor_dice"]].mean().reset_index()
    print((agg.groupby("group")[["tumor_tp", "tumor_fp", "tumor_fn"]].sum() / 1000).round(1))

    section("4 逐病例肿瘤 Dice（seed 平均，%）及相对 A0 的差值")
    w = agg.pivot(index="case", columns="group", values="tumor_dice") * 100
    for g in GROUPS[1:]:
        if g in w:
            w[f"{g}-A0"] = w[g] - w["A0"]
    gt_vox = Ct[Ct.run == Ct.run.iloc[0]].set_index("case")[["tumor_tp", "tumor_fn"]].sum(1).rename("gt_voxels")
    print(w.join(gt_vox).sort_values("A6-A0").round(1))
    print("\n逐 seed 肿瘤 Dice（%）：")
    print((Ct.groupby("run").tumor_dice.mean() * 100).round(2))

    section("5 病灶召回与匹配病灶 IoU（%），按大小、CNR 分层（run 合并，组内平均）")
    a = L[L.run == L.run.iloc[0]]
    print(pd.crosstab(a.size_stratum, a.cnr_stratum, margins=True))
    for col in ("size_stratum", "cnr_stratum"):
        print(f"\n召回 by {col}")
        print((L.groupby(["group", col])["detected@0.1"].mean().unstack() * 100).round(1))
        print(f"匹配病灶 IoU by {col}")
        print((L[L["detected@0.1"]].groupby(["group", col])["iou@0.1"].mean().unstack() * 100).round(1))

    section("6 A0 三个 seed 都漏掉的病灶")
    a0 = L[L.group == "A0"]
    rate = a0.groupby(["case", "lesion"])["detected@0.1"].mean()
    first = a0[a0.run == sorted(a0.run.unique())[0]].set_index(["case", "lesion"])
    miss = first.join(rate.rename("det_rate"))
    miss = miss[miss.det_rate == 0][["voxels", "diameter_mm", "cnr", "cnr_stratum"]]
    print(miss.round(2).to_string())
    print(f"\n共 {len(miss)} 个，其中低 CNR {int((miss.cnr_stratum == 'low').sum())} 个")


if __name__ == "__main__":
    main()
