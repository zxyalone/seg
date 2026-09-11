#!/usr/bin/env python3
"""
Step 2 / 4.1b — 机制解释图：PC 为什么在低 CNR 下失去定位能力

s41 的核心反常现象：
  · 无噪声阶梯边缘上，PC 在**整个半空间**都高（中心 0.64 → 边缘 0.996 → 体外 0.93），
    而 |∇f| 是干净的单点脉冲（边缘 4.9，其余 1e-4）。
  · 说明 PC 的定位能力并非来自算子本身，而是来自 Kovesi 自适应噪声阈值 T：
      T ∝ median(|最高频 scale 的响应|) × gain
    噪声升高 → median 被噪声抬高 → T 抬高 → 弱幅度的"内部"被压掉、强幅度的"边缘"被保留。
  · 但当噪声进一步升高，T 相对所有位置都过大 → PC 全域趋于 0/常数，定位能力消失。

本脚本用剖面 + 阈值曲线把这条因果链做成图。
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import FIGS, PC_MAX_VOXELS_FULL, banner, save_json  # noqa: E402
from nnunetv2.training.nnUNetTrainer.phaser.dim_utils import (  # noqa: E402
    spatial_gradient_magnitude,
)
from nnunetv2.training.nnUNetTrainer.phaser.modules import PhaseCongruency  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

N, BASE, SOFT, R = 160, 60.0, 1.0, 60.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pc_mod = PhaseCongruency(dims=3, n_scales=3, max_voxels=PC_MAX_VOXELS_FULL).to(DEV).eval()

z = np.arange(N, dtype=np.float32) - (N - 1) / 2
# z 坐标 -> 数组下标：z=0 落在下标 (N-1)/2 = 79.5，边缘 z=±R 对应下标 R+(N-1)/2
IDX = lambda v: int(round(v + (N - 1) / 2))  # noqa: E731
_profile = (1.0 / (1.0 + np.exp((np.abs(z) - R) / SOFT))).astype(np.float32)
OBJ = np.broadcast_to(_profile[:, None, None], (N, N, N)).copy()


def run(x):
    with torch.no_grad():
        t = torch.from_numpy(x)[None, None].to(DEV).float()
        pc = pc_mod(t)[0, 0].cpu().numpy()
        gr = spatial_gradient_magnitude(t, 3)[0, 0].cpu().numpy()
    return pc, gr


def kovesi_threshold(x):
    """复现 PhaseCongruency 内部阈值 T 与各 scale 幅度，用于解释性曲线。"""
    with torch.no_grad():
        t = torch.from_numpy(x)[None, None].to(DEV).float().float()
        F = torch.fft.rfftn(t, dim=(-3, -2, -1))
        log_rho, unit, dc = pc_mod._freq_grid((N, N, N), t.device)
        rho_s = 0.5 * torch.sigmoid(pc_mod.rho_raw)
        sig_s = 0.05 + 0.9 * torch.sigmoid(pc_mod.sigma_raw)
        A = []
        for s in range(3):
            shp = (-1,) + (1,) * 3
            lnr = log_rho.unsqueeze(0) - torch.log(rho_s[s]).view(*shp)
            den = 2.0 * (torch.log(sig_s[s]) ** 2).view(*shp)
            g = torch.exp(-(lnr ** 2) / den).masked_fill(dc.unsqueeze(0), 0.0)
            K = torch.complex(g, torch.zeros_like(g))
            r = torch.fft.irfftn(F * K.unsqueeze(0), s=(N, N, N), dim=(-3, -2, -1))
            A.append(r[0, 0].abs())
        Astack = torch.stack(A, 0)                      # [3, D,H,W]
        A0 = Astack[0].detach()
        med = A0.flatten().median()
        tau = med / np.sqrt(np.log(4.0))
        gain = (rho_s / rho_s.max()).sum()
        k = float(torch.nn.functional.softplus(pc_mod.noise_k))
        T = float(tau * gain * (np.sqrt(np.pi / 2) + k * np.sqrt((4 - np.pi) / 2)))
        den = float(Astack.sum(0).mean())
        center = Astack[:, N // 2, N // 2, N // 2].tolist()
        edge = Astack[:, IDX(R), N // 2, N // 2].tolist()
        return dict(T=T, den_mean=den, med_A0=float(med), tau=float(tau),
                    gain=float(gain), k=k,
                    amp_center=center, amp_edge=edge,
                    rho=rho_s.tolist(), sigma=sig_s.tolist())


def main():
    banner("Step 2 / 4.1b 机制解释：PC 的定位依赖 Kovesi 自适应噪声阈值")
    out = {}
    zi = N // 2
    prof_z = np.arange(N) - (N - 1) / 2

    # ---- 无噪声 ----
    pc0, gr0 = run((BASE + 20.0 * OBJ).astype(np.float32))
    s0 = kovesi_threshold((BASE + 20.0 * OBJ).astype(np.float32))
    out["noiseless"] = dict(kovesi=s0)
    pc_c = pc0[:, zi, zi]
    gr_c = gr0[:, zi, zi]
    inside = np.abs(prof_z) <= R - 30
    PC_EDGE_NL = float(pc_c[IDX(R)])
    PC_INSIDE_NL = float(pc_c[inside].mean())
    PC_CEIL = PC_EDGE_NL / max(PC_INSIDE_NL, 1e-9)
    print(f"[无噪声 A=20]  PC: 中心={pc0[zi,zi,zi]:.4f} 边缘(z=+R)={pc0[IDX(R),zi,zi]:.4f} "
          f"体外(z=+77)={pc0[IDX(77),zi,zi]:.4f}  内部均值={pc_c[inside].mean():.4f}")
    print(f"               |∇f|: 边缘={gr_c[IDX(R)]:.4f}  内部均值={gr_c[inside].mean():.3e}")
    print(f"               PC 边缘/内部对比度 = {pc_c[IDX(R)]/max(pc_c[inside].mean(),1e-9):.3f}× "
          f"（|∇f| 对比度 = {gr_c[IDX(R)]/max(gr_c[inside].mean(),1e-12):.3e}×）")
    print(f"               T={s0['T']:.4e}  den均值={s0['den_mean']:.4e}  median(A0)={s0['med_A0']:.4e}")
    print(f"               scale 幅度 中心={['%.2e'%v for v in s0['amp_center']]} "
          f"边缘={['%.2e'%v for v in s0['amp_edge']]}")

    # ---- 不同 CNR ----
    CNRS = [0.25, 1.0, 4.0, 16.0]
    SIGMA = 5.0
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6))
    for cnr in CNRS:
        A = cnr * SIGMA
        rng = np.random.default_rng(0)
        x = (BASE + A * OBJ + SIGMA * rng.normal(0, 1, OBJ.shape)).astype(np.float32)
        pc, gr = run(x)
        kk = kovesi_threshold(x)
        pc_c = pc[:, zi, zi]
        gr_c = gr[:, zi, zi]
        b_ = np.abs(prof_z - R) <= 1.5
        _ = None
        i_ = np.abs(prof_z) <= R - 30
        rows.append(dict(cnr=cnr, A=A, sigma=SIGMA,
                         pc_center=float(pc[zi, zi, zi]), pc_edge=float(pc[IDX(R), zi, zi]),
                         pc_outside=float(pc[IDX(77), zi, zi]),
                         pc_interior_mean=float(pc_c[i_].mean()),
                         grad_edge=float(gr_c[IDX(R)]), grad_interior_mean=float(gr_c[i_].mean()),
                         pc_edge_over_interior=float(pc_c[b_].mean() / max(pc_c[i_].mean(), 1e-9)),
                         grad_edge_over_interior=float(gr_c[b_].mean() / max(gr_c[i_].mean(), 1e-12)),
                         T=kk["T"], den_mean=kk["den_mean"], med_A0=kk["med_A0"],
                         amp_center=kk["amp_center"], amp_edge=kk["amp_edge"]))
        print(f"[CNR={cnr:5.2f} A={A:5.2f}] PC 中心={pc[zi,zi,zi]:.4f} 边缘={pc[IDX(R),zi,zi]:.4f} "
              f"内部均值={pc_c[i_].mean():.4f} 边缘/内部={rows[-1]['pc_edge_over_interior']:.3f}  "
              f"T={kk['T']:.3e} den={kk['den_mean']:.3e} T/den={kk['T']/max(kk['den_mean'],1e-12):.4f}  "
              f"|∇f| 边缘/内部={rows[-1]['grad_edge_over_interior']:.3f}")
    out["with_noise"] = rows

    # ---- 图 ----
    ax = axes[0, 0]
    ax.plot(prof_z, pc0[:, zi, zi], color="#1D9E75", lw=1.8, label="PC (noiseless)")
    ax.axvline(R, color="gray", ls=":", lw=1); ax.axvline(-R, color="gray", ls=":", lw=1)
    ax.set_xlabel("z (voxels, lesion centre = 0)"); ax.set_ylabel("phase congruency", color="#1D9E75")
    ax.set_ylim(0, 1.05)
    a2 = ax.twinx()
    a2.plot(prof_z, gr0[:, zi, zi], color="#D85A30", lw=1.8, label="|grad f| (noiseless)")
    a2.set_ylabel("|grad f|", color="#D85A30"); a2.set_yscale("symlog", linthresh=1e-4)
    ax.set_title("A. Noiseless slab: PC is high across the WHOLE half-space;\n|grad f| is a clean edge impulse")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right"); ax.grid(alpha=.25)

    ax = axes[0, 1]
    for cnr in CNRS:
        A = cnr * SIGMA
        rng = np.random.default_rng(0)
        x = (BASE + A * OBJ + SIGMA * rng.normal(0, 1, OBJ.shape)).astype(np.float32)
        pc, _ = run(x)
        ax.plot(prof_z, pc[:, zi, zi], lw=1.7, label=f"CNR={cnr} (A={A:.1f} HU)")
    ax.axvline(R, color="gray", ls=":", lw=1); ax.axvline(-R, color="gray", ls=":", lw=1)
    ax.set_xlabel("z (voxels)"); ax.set_ylabel("phase congruency"); ax.set_ylim(0, 1.05)
    ax.set_title("B. PC profile vs CNR (sigma=5 HU)\nLocalisation only emerges at high CNR")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[0, 2]
    cnrs = [r["cnr"] for r in rows]
    ax.plot(cnrs, [r["den_mean"] for r in rows], "o-", color="#534AB7",
            label="den = mean(sum_s |A_s|)")
    ax.plot(cnrs, [r["T"] for r in rows], "s-", color="#A32D2D",
            label="T (Kovesi threshold)")
    ax.plot(cnrs, [r["med_A0"] for r in rows], "^-", color="#BA7517",
            label="median |A_0| (noise estimate)")
    ax.plot(cnrs, [np.mean(r["amp_edge"]) for r in rows], "v--", color="#1D9E75",
            label="mean |response| at edge")
    ax.plot(cnrs, [np.mean(r["amp_center"]) for r in rows], "v:", color="#D85A30",
            label="mean |response| at centre")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("CNR (log)"); ax.set_ylabel("magnitude (same units)")
    for r in rows:
        ax.annotate(f"T/den={r['T']/max(r['den_mean'],1e-12):.2f}",
                    (r["cnr"], r["T"]), textcoords="offset points", xytext=(2, 6),
                    fontsize=7, color="#A32D2D")
    ax.set_title("C. T exceeds den at low CNR -> PC collapses to 0\n"
                 "(annotations show T/den)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.25, which="both")

    ax = axes[1, 0]
    ax.plot(cnrs, [r["pc_edge_over_interior"] for r in rows], "o-",
            color="#1D9E75", label="PC: edge / interior")
    ax.plot(cnrs, [r["grad_edge_over_interior"] for r in rows], "s-",
            color="#D85A30", label="|grad f|: edge / interior")
    ax.axhline(PC_CEIL, color="#1D9E75", ls=":", lw=1.2,
               label=f"PC noiseless ceiling ({PC_CEIL:.2f}x)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("CNR (log)"); ax.set_ylabel("edge / interior ratio (log)")
    ax.set_title("D. PC ratio only exceeds |grad f| once CNR is high\n(the direction opposite to C1)")
    ax.legend(fontsize=8); ax.grid(alpha=.25, which="both")

    ax = axes[1, 1]
    ax.plot(cnrs, [r["pc_center"] for r in rows], "o-", color="#1D9E75", label="PC at lesion centre")
    ax.plot(cnrs, [r["pc_edge"] for r in rows], "s-", color="#0F6E56", label="PC at lesion edge")
    ax.plot(cnrs, [r["pc_outside"] for r in rows], "^--", color="#BA7517", label="PC outside lesion")
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("CNR (log)"); ax.set_ylabel("phase congruency")
    ax.set_title("E. At low CNR PC collapses uniformly (abstains);\n"
                 "it never becomes a localiser")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[1, 2]
    for s, marker in zip(range(3), ("o", "s", "^")):
        ax.plot(cnrs, [r["amp_edge"][s] for r in rows], marker=marker, ls="-",
                label=f"edge, scale {s+1} (lambda={3.0*2.1**s:.1f})")
        ax.plot(cnrs, [r["amp_center"][s] for r in rows], marker=marker, ls=":",
                alpha=.65, label=f"centre, scale {s+1}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("CNR (log)"); ax.set_ylabel("|log-Gabor response|")
    ax.set_title("F. Per-scale amplitudes: the centre signal decays\n"
                 "with CNR faster than the edge signal")
    ax.legend(fontsize=7, ncol=1); ax.grid(alpha=.25, which="both")

    fig.suptitle("4.1b Mechanism: PC's edge localisation comes from the adaptive Kovesi threshold, not from the operator",
                 fontsize=13, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .975))
    fp = FIGS / "fig41b_pc_threshold_mechanism.png"
    fig.savefig(fp, dpi=140)
    plt.close(fig)
    print(f"\n[fig] {fp}")
    save_json(out, "s41b_mechanism.json")
    banner("4.1b 完成")


if __name__ == "__main__":
    main()
