#!/usr/bin/env python3
"""Step 2 / 4.2c — 仅从 out/s42_real_pc.json 重绘 4.2 的图（避免重跑 30 分钟的数据部分）。

用途：s42_real_pc.py 的绘图段曾因 boxplot 参数（缺 patch_artist）异常退出；
本脚本只读已落盘的 JSON 并出图。
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import FIGS, OUT, banner  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COL = {"pc": "#1D9E75", "g": "#D85A30"}
NAME = {"pc": "phase congruency", "g": "|grad f|"}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    banner("Step 2 / 4.2c 重绘 fig42（只读 JSON）")
    R = json.load(open(OUT / "s42_real_pc.json"))
    ok = R["per_lesion_ok"]
    rows = R["stratum_table"]
    prof_bins = R["profile_bins"]
    pc_mean = np.array(R["profile_pc_mean"])
    g_mean = np.array(R["profile_grad_mean"])

    cnrs = np.array([r["cnr"] for r in ok])
    sizes = np.array([r["size_vox"] for r in ok])
    sep_pc = np.array([r["sep_pc"] for r in ok])
    sep_g = np.array([r["sep_g"] for r in ok])
    print(f"n={len(ok)}  cnr 中位={np.median(cnrs):.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2))
    x = np.arange(len(prof_bins))
    ax = axes[0]
    ax.plot(x, pc_mean, "o-", color=COL["pc"], label=NAME["pc"])
    ax2 = ax.twinx()
    ax2.plot(x, g_mean, "s-", color=COL["g"], label=NAME["g"])
    ax2.set_ylabel("|grad f|", color=COL["g"])
    ax.set_ylabel("phase congruency", color=COL["pc"])
    ax.set_xticks(x); ax.set_xticklabels(prof_bins, rotation=60, ha="right", fontsize=8)
    ax.axvline(prof_bins.index("out1-2") - 0.5, color="gray", ls="--", lw=1)
    ax.set_title("Mean response vs distance from lesion surface\n(pooled over 21 val cases)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc="upper right"); ax2.legend(fontsize=8, loc="center right")

    ax = axes[1]
    ax.scatter(cnrs, sep_pc, s=16, color=COL["pc"], alpha=.8, label=NAME["pc"])
    ax.scatter(cnrs, sep_g, s=16, color=COL["g"], alpha=.8, label=NAME["g"])
    ax.set_ylim(-3, 8); ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("lesion CNR"); ax.set_ylabel("sep = (mu_b - mu_peri) / sigma_peri")
    ax.set_title(f"Boundary separability vs CNR\nSpearman PC={spearman(cnrs, sep_pc):+.2f}, "
                 f"|grad|={spearman(cnrs, sep_g):+.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[2]
    order = [r["stratum"] for r in rows]
    rng = np.random.default_rng(0)
    for i, sn in enumerate(order):
        sub = [r for r in ok if r["cnr_stratum"] == sn]

        def fin(v):
            v = float(v)
            return v if np.isfinite(v) and v > 1e-6 else np.nan
        ax.boxplot([[fin(r["ratio_pc"]) for r in sub], [fin(r["ratio_g"]) for r in sub]],
                   positions=[i * 2 + 1, i * 2 + 2], widths=.7,
                   patch_artist=True, boxprops=dict(facecolor="none"),
                   medianprops=dict(color="black"))
        for p, k in zip([i * 2 + 1, i * 2 + 2], ("pc", "g")):
            ax.scatter(np.full(len(sub), p) + rng.uniform(-.15, .15, len(sub)),
                       [max(fin(r[f"ratio_{k}"]), 1e-6) for r in sub], s=9,
                       color=COL[k], alpha=.65)
    ax.set_xticks([i * 2 + 1.5 for i in range(len(order))]); ax.set_xticklabels(order)
    ax.set_yscale("log"); ax.set_ylabel("boundary / peri-liver ratio (log)")
    ax.set_title("Ratio by CNR tertile (left=PC, right=|grad f|)")
    ax.grid(alpha=.25, axis="y")
    ax.legend(handles=[Patch(facecolor=COL["pc"], label=NAME["pc"]),
                       Patch(facecolor=COL["g"], label=NAME["g"])], fontsize=8)

    ax = axes[3]
    ax.scatter(sizes, sep_pc, s=16, color=COL["pc"], alpha=.8, label=NAME["pc"])
    ax.scatter(sizes, sep_g, s=16, color=COL["g"], alpha=.8, label=NAME["g"])
    ax.set_ylim(-3, 8); ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("lesion size (voxels, log)"); ax.set_ylabel("sep")
    ax.set_title(f"Separability vs lesion size\nSpearman PC={spearman(sizes, sep_pc):+.2f}, "
                 f"|grad|={spearman(sizes, sep_g):+.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle("4.2 Real cases (fold-0 validation, 21 cases): PC vs |grad f| at boundary / interior / peri-liver (C1)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fp = FIGS / "fig42_real_case_pc_response.png"
    fig.savefig(fp, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {fp}")
    banner("4.2c 完成")


if __name__ == "__main__":
    main()
