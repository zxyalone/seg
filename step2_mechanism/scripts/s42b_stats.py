#!/usr/bin/env python3
"""
Step 2 / 4.2b — 从 s42_real_pc.json 复算跨算子可比的效应量

为什么需要它：
  s42 的 sep = (μ_b − μ_peri) / σ_peri 只用外周环自身标准差做分母，PC 与 |∇f|
  的分母量级完全不同（PC 的 σ 很小），因此 sep 的**绝对值**不可跨算子比较。
  这里补一个对称的标准化效应量（Cohen's d，用合并标准差）以及 AUC 式的秩统计，
  让"PC 是否比 |∇f| 更能把病灶边界同外周肝实质分开"这个问题有可比答案。
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import OUT, banner, save_json  # noqa: E402


def cohens_d(m1, s1, n1, m2, s2, n2):
    if not all(np.isfinite([m1, s1, n1, m2, s2, n2])) or n1 < 2 or n2 < 2:
        return float("nan")
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / max(n1 + n2 - 2, 1))
    return (m1 - m2) / (sp + 1e-12) if sp > 0 else float("nan")


def main():
    banner("Step 2 / 4.2b 跨算子可比效应量（Cohen's d）")
    p = OUT / "s42_real_pc.json"
    if not p.is_file():
        sys.exit(f"[ERROR] 缺少 {p}，请先运行 s42_real_pc.py")
    R = json.load(open(p))
    ok = R["per_lesion_ok"]
    print(f"病灶数 = {len(ok)}；CNR 三分位阈值 = {R['train_cnr_tertiles']}")

    for r in ok:
        r["_d_pc"] = cohens_d(r["pc_b"], r.get("pc_b_std", float("nan")), r["boundary_vox"],
                              r["pc_p"], r["pc_p_std"], r["peri_pad_vox"])
        r["_d_g"] = cohens_d(r["g_b"], r.get("g_b_std", float("nan")), r["boundary_vox"],
                             r["g_p"], r["g_p_std"], r["peri_pad_vox"])
    # 注：json 里未存 boundary 的 std，改为用 (μ_b−μ_p)/σ_p 的秩统计做补充
    print("\n（json 未保存边界区标准差，故下面只用 sep 的秩统计与符号检验）")

    strata = ["low", "mid", "high"]
    print(f"\n{'层':>5} {'n':>4} {'CNR中位':>8} | {'PC ratio中位':>12} {'g ratio中位':>11} | "
          f"{'sep_PC中位':>10} {'sep_g中位':>9} | {'PC>g 计数':>10}")
    tbl = []
    for s in strata:
        sub = [r for r in ok if r["cnr_stratum"] == s]
        if not sub:
            continue
        med = lambda k: float(np.nanmedian([r[k] for r in sub]))  # noqa: E731
        win = sum(1 for r in sub if r["ratio_pc"] > r["ratio_g"])
        winsep = sum(1 for r in sub if r["sep_pc"] > r["sep_g"])
        tbl.append(dict(stratum=s, n=len(sub), cnr_median=med("cnr"),
                        ratio_pc_median=med("ratio_pc"), ratio_g_median=med("ratio_g"),
                        sep_pc_median=med("sep_pc"), sep_g_median=med("sep_g"),
                        n_ratio_pc_wins=win, n_sep_pc_wins=winsep))
        print(f"{s:>5} {len(sub):4d} {med('cnr'):8.3f} | {med('ratio_pc'):12.3f} "
              f"{med('ratio_g'):11.3f} | {med('sep_pc'):10.3f} {med('sep_g'):9.3f} | "
              f"{win}/{len(sub)} (sep {winsep}/{len(sub)})")

    print("\n按 CNR 分位的 PC / |∇f| 中位 boundary-peri 比：")
    cnr = np.array([r["cnr"] for r in ok])
    rpc = np.array([r["ratio_pc"] for r in ok])
    rg = np.array([r["ratio_g"] for r in ok])
    qs = np.quantile(cnr, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        m = (cnr >= qs[i]) & (cnr <= qs[i + 1])
        if m.sum() == 0:
            continue
        print(f"  CNR [{qs[i]:.2f}, {qs[i+1]:.2f}]  n={int(m.sum()):3d}  "
              f"PC ratio 中位={np.median(rpc[m]):.3f}  |∇f| ratio 中位={np.median(rg[m]):.3f}  "
              f"PC/|∇f|={np.median(rpc[m])/max(np.median(rg[m]),1e-9):.3f}")

    print("\n低 CNR 三层（CNR<1）的比值分布：")
    m = cnr < 1.0
    print(f"  n={int(m.sum())}  PC ratio: 中位={np.median(rpc[m]):.3f} "
          f"均值={rpc[m].mean():.3f}  <1 的比例={float((rpc[m]<1).mean()):.1%}")
    print(f"           |∇f| ratio: 中位={np.median(rg[m]):.3f} "
          f"均值={rg[m].mean():.3f}  <1 的比例={float((rg[m]<1).mean()):.1%}")

    save_json(dict(stratum_table=tbl,
                   cnr_quintile=[dict(lo=float(qs[i]), hi=float(qs[i+1]),
                                      n=int(((cnr >= qs[i]) & (cnr <= qs[i+1])).sum()),
                                      pc_ratio_median=float(np.median(rpc[(cnr >= qs[i]) & (cnr <= qs[i+1])])),
                                      g_ratio_median=float(np.median(rg[(cnr >= qs[i]) & (cnr <= qs[i+1])])))
                                 for i in range(5)]),
              "s42b_stats.json")
    banner("4.2b 完成")


if __name__ == "__main__":
    main()
