#!/usr/bin/env python3
"""
Step 2 / 4.4b — 核实 pc_max_voxels=2^20 对最浅层 PC 分辨率的影响

代码路径（modules.py PhaseCongruency.forward）：
    factor=1; while _numel(factor) > max_voxels and all(s//(factor*2) >= 8): factor *= 2
    if factor > 1: xf = downsample(xf, /factor); PC 在 work 上计算
    ...
    if factor > 1: pc = upsample(pc, 回 spatial)      # 注意：上采样回原分辨率后返回
因此 route 张量的形状仍是 128³，但其信息量已被限制在 64³。

本脚本用真实 patch 直接比较：
    PC(max_voxels=2^20   —— 模型实际配置)
    PC(max_voxels=2^26   —— 不降采样)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import (  # noqa: E402
    FIGS, banner, lesions_from_seg, load_input_and_seg, save_json, splits,
)
from nnunetv2.training.nnUNetTrainer.phaser.modules import PhaseCongruency  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CAP = 1 << 20
FULL = 1 << 26


def pc_with(x, max_voxels):
    m = PhaseCongruency(dims=3, n_scales=3, max_voxels=max_voxels).to(DEV).eval()
    with torch.no_grad():
        t = torch.from_numpy(x)[None, None].to(DEV).float()
        return m(t)[0, 0].float().cpu().numpy()


def main():
    banner("Step 2 / 4.4b pc_max_voxels=2^20 对最浅层 PC 有效分辨率的影响")
    print(f"cap = {CAP}  (2^20)   full = {FULL}")
    for s in (128, 64, 32):
        print(f"  {s}^3 = {s**3:>9d}   > cap ? {s**3 > CAP}")
    tr, va, te = splits()

    rows = []
    patches = []
    for case in va:
        img, seg = load_input_and_seg(case)
        les = lesions_from_seg(seg, 2000)
        if not les:
            continue
        _, mask, size = les[0]
        c = np.argwhere(mask).mean(0).round().astype(int)
        lo = np.maximum(c - 64, 0)
        hi = np.minimum(lo + 128, np.array(seg.shape))
        lo = np.maximum(hi - 128, 0)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        pi = img[sl]
        if pi.shape != (128,) * 3:
            continue
        patches.append((case, pi.copy()))
        if len(patches) >= 3:
            break

    fig, axes = plt.subplots(len(patches), 4, figsize=(16, 4.1 * len(patches)))
    if len(patches) == 1:
        axes = axes[None]
    for r, (case, pi) in enumerate(patches):
        pc_cap = pc_with(pi, CAP)
        pc_full = pc_with(pi, FULL)
        diff = np.abs(pc_cap - pc_full)
        zc = 64
        rows.append(dict(case=case, mean_abs_diff=float(diff.mean()),
                         max_abs_diff=float(diff.max()),
                         mean_pc_cap=float(pc_cap.mean()),
                         mean_pc_full=float(pc_full.mean()),
                         frac_changed_gt_1e3=float((diff > 1e-3).mean()),
                         # 勘误：旧版用 argsort 排名，PC 有大量并列的 0 值，
                         # 并列被任意拆开，秩相关被系统性低估。改用平均秩。
                         spearman=float(spearmanr(pc_cap.ravel(), pc_full.ravel())[0]),
                         spearman_nonzero=float(spearmanr(
                             pc_cap.ravel()[(pc_cap.ravel() > 0) | (pc_full.ravel() > 0)],
                             pc_full.ravel()[(pc_cap.ravel() > 0) | (pc_full.ravel() > 0)])[0]),
                         frac_zero_cap=float((pc_cap == 0).mean()),
                         frac_zero_full=float((pc_full == 0).mean())))
        print(f"[{case}] mean|ΔPC|={rows[-1]['mean_abs_diff']:.5f} "
              f"max|ΔPC|={rows[-1]['max_abs_diff']:.5f} "
              f"Δ>1e-3 体素占比={rows[-1]['frac_changed_gt_1e3']:.3%} "
              f"PC 均值 cap={rows[-1]['mean_pc_cap']:.4f} full={rows[-1]['mean_pc_full']:.4f} "
              f"秩相关={rows[-1]['spearman']:.4f}")

        for c, (arr, ttl) in enumerate([(pi[zc], f"{case} CT z={zc}"),
                                        (pc_cap[zc], f"PC cap=2^20 (model)\nmean={pc_cap.mean():.4f}"),
                                        (pc_full[zc], f"PC no cap\nmean={pc_full.mean():.4f}"),
                                        (diff[zc], f"|cap - no-cap|\nmax={diff.max():.4f}")]):
            im = axes[r, c].imshow(arr, cmap="gray" if c == 0 else "inferno")
            axes[r, c].set_title(ttl, fontsize=9)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if c > 0:
                plt.colorbar(im, ax=axes[r, c], fraction=.046)
    fig.suptitle("4.4b Effect of pc_max_voxels=2^20 on the shallowest skip (128^3 > cap -> PC computed at 64^3, upsampled back)",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    fp = FIGS / "fig44b_pc_max_voxels_effect.png"
    fig.savefig(fp, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {fp}")

    save_json(dict(cap=CAP, full=FULL,
                   note="PhaseCongruency 在 cap 触发时先降采样到 work 分辨率计算，再把 PC 图插值回原尺寸；"
                        "因此 route 张量形状仍为 128^3，但其信息量被限制在 64^3。",
                   per_case=rows), "s44b_pc_cap.json")
    banner("4.4b 完成")


if __name__ == "__main__":
    main()
