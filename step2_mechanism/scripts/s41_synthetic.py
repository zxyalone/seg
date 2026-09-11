#!/usr/bin/env python3
"""
Step 2 / 4.1 — 合成实验（检验 C1）

命题 C1（机制）：相位一致性（PC）在低对比度下比梯度幅值更稳。
证伪条件：低 CNR 区间 PC 的"边界/内部响应比"不高于 |∇f|。

v2 相对 v1 的关键修正
--------------------
v1 的"内部"参考区距边界仅 4.5 体素，而最大 log-Gabor 波长为 13.2 体素，
滤波器空间支撑覆盖了参考区，导致 PC_interior 被真实边缘信号污染（无噪声时
全volume均值 PC 高达 0.43），结论不可用。

v2 强制 **参考区距边界 ≥ 30 体素 ≈ 2.27 × λ_max**，并把判据从单一"比值"
扩充为"比值 + d′ + AUC"三个不依赖绝对标度的指标（PC∈[0,1] 而 |∇f| 无界，
比值是唯一量纲无关的比较量，但易被分母趋零放大）。

设计
----
* 体素间距 1.0（各向同性），体积 160³。
* 几何甲「厚板病灶」：|z| ≤ 60；边界 |z| ∈ [58.5, 61.5]；内部 |z| ≤ 30。
* 几何乙「球病灶」  ：半径 60；边界 |r − 60| ≤ 1.5；内部 r ≤ 30。
* 成像模型 x = B + A·obj + σ·noise，B=60（类肝实质本底，滤波前被 DC 抑制）。
* S1 固定 σ=5 HU，A = CNR·σ，CNR ∈ [0.25 … 16]   —— 主扫描
* S2 固定 A=15 HU，σ ∈ {1,2,5,10,20,40}          —— 分离"噪声主导"
* S3 无噪声，A ∈ {5,10,20,40}                     —— 直接检验对比度不变性
* 对照算子：仓库 PhaseCongruency(n_scales=3)；|∇f| 原始值；|∇f| 逐样本 min-max
  （PCSF grad 路径实际使用）；|∇f| 用全局噪声底归一化（对梯度最有利的乐观对照）。
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import (  # noqa: E402
    FIGS, PC_MAX_VOXELS_FULL, banner, minmax01, save_json,
)
from nnunetv2.training.nnUNetTrainer.phaser.dim_utils import (  # noqa: E402
    spatial_gradient_magnitude,
)
from nnunetv2.training.nnUNetTrainer.phaser.modules import PhaseCongruency  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

N = 160
BASE = 60.0
SOFT = 1.0
R = 60.0          # 病灶半径 / 半厚
MARGIN = 30.0     # 参考区距边界的最小距离（≈2.27 λ_max）
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# 几何
# ----------------------------------------------------------------------
def _axis():
    return np.arange(N, dtype=np.float32) - (N - 1) / 2.0


def geometry(kind: str):
    """返回 (obj, boundary_mask, interior_mask)。

    kind: 'slab' | 'ball' | 'slab_soft'
    slab_soft 用 3 体素过渡带模拟真实病灶的模糊边缘（真实 LiTS 病灶边缘约 2–5 体素）。
    """
    soft = 3.0 if kind == "slab_soft" else SOFT
    base = kind.replace("_soft", "")
    z = _axis()
    zz = np.broadcast_to(z[:, None, None], (N, N, N))
    if base == "slab":
        d_abs = np.abs(zz)
    else:
        zz2, yy2, xx2 = np.meshgrid(z, z, z, indexing="ij")
        d_abs = np.sqrt(zz2 ** 2 + yy2 ** 2 + xx2 ** 2)
    obj = (1.0 / (1.0 + np.exp((d_abs - R) / soft))).astype(np.float32)
    b = np.abs(d_abs - R) <= 1.5 * soft
    i = d_abs <= (R - MARGIN)
    return obj, b, i


# ----------------------------------------------------------------------
# 算子
# ----------------------------------------------------------------------
_pc = PhaseCongruency(dims=3, n_scales=3, max_voxels=PC_MAX_VOXELS_FULL).to(DEV).eval()


def compute_pc(x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(x)[None, None].to(DEV).float()
        out = _pc(t)
    return out[0, 0].float().cpu().numpy().astype(np.float64)


def compute_grad(x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(x)[None, None].to(DEV).float()
        g = spatial_gradient_magnitude(t, 3)
    return g[0, 0].float().cpu().numpy().astype(np.float64)


def auc_fast(pos: np.ndarray, neg: np.ndarray, rng, n_max=40000) -> float:
    """Mann-Whitney AUC，正类 > 负类。对样本做下采样以控制成本。"""
    p = pos if pos.size <= n_max else rng.choice(pos, n_max, replace=False)
    n = neg if neg.size <= n_max else rng.choice(neg, n_max, replace=False)
    if p.size == 0 or n.size == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = allv.argsort()
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # 处理并列
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sum_by_val = np.zeros(cnt.size)
    np.add.at(sum_by_val, inv, ranks)
    ranks = (sum_by_val / cnt)[inv]
    r_pos = ranks[: p.size].sum()
    return float((r_pos - p.size * (p.size + 1) / 2.0) / (p.size * n.size))


def metrics(resp: np.ndarray, b, i, rng) -> dict:
    rb, ri = resp[b], resp[i]
    mb, mi = float(rb.mean()), float(ri.mean())
    si = float(ri.std())
    return dict(
        mean_b=mb, mean_i=mi, std_i=si,
        ratio=mb / max(mi, 1e-6),
        dprime=(mb - mi) / (si + 1e-9),
        auc=auc_fast(rb, ri, rng),
    )


# ----------------------------------------------------------------------
# 单次试验
# ----------------------------------------------------------------------
_CACHE = {}


def one_trial(kind: str, A: float, sigma: float, seed: int):
    if kind not in _CACHE:
        _CACHE[kind] = geometry(kind)
    obj, b, i = _CACHE[kind]
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (N, N, N)).astype(np.float32)
    x = (BASE + A * obj + sigma * noise).astype(np.float32)

    pc = compute_pc(x)
    gr = compute_grad(x)
    # 对梯度最有利的乐观对照：用已知噪声底 σ 归一化
    gr_sn = gr / (sigma * 1.4142 + 1e-8) if sigma > 0 else gr
    gr_n = minmax01(gr)

    urng = np.random.default_rng(12345 + seed)
    res = {}
    for name, mp in (("pc", pc), ("grad", gr), ("grad_snr", gr_sn), ("grad_norm", gr_n)):
        for k, v in metrics(mp, b, i, urng).items():
            res[f"{name}_{k}"] = v
    res["A"] = A
    res["sigma"] = sigma
    res["cnr_nominal"] = A / sigma if sigma > 0 else float("inf")
    res["interior_voxels"] = int(i.sum())
    res["boundary_voxels"] = int(b.sum())
    return res


def aggregate(records: list[dict]) -> dict:
    keys = [k for k in records[0] if not k.endswith("_voxels") and k not in ("A", "sigma")]
    out = {"interior_voxels": int(records[0]["interior_voxels"]),
           "boundary_voxels": int(records[0]["boundary_voxels"]),
           "A": float(records[0]["A"]), "sigma": float(records[0]["sigma"])}
    for k in keys:
        v = np.array([r[k] for r in records], dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k + "_mean"] = float(v.mean()) if v.size else float("nan")
        out[k + "_std"] = float(v.std(ddof=0)) if v.size else float("nan")
    return out


def main():
    banner("Step 2 / 4.1 合成实验 v2 — PC vs |∇f| 随 CNR 的变化（检验 C1）")
    print(f"device={DEV}  N={N}  margin={MARGIN} voxels  参考区可见性已校正")

    seeds = [0, 1, 2, 3, 4]
    result = {"config": dict(N=N, margin=MARGIN, base=BASE, soft=SOFT, R=R,
                             device=DEV, pc_max_voxels=PC_MAX_VOXELS_FULL,
                             lambda_max=3.0 * 2.1 ** 2),
              "geometries": {}}

    SIGMA_FIX = 5.0
    CNRS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

    for kind in ("slab", "ball", "slab_soft"):
        obj, b, i = _CACHE.setdefault(kind, geometry(kind))
        print(f"\n[{kind}] boundary={int(b.sum())} vox  interior={int(i.sum())} vox")

        rows = []
        for cnr in CNRS:
            A = cnr * SIGMA_FIX
            recs = [one_trial(kind, A, SIGMA_FIX, s) for s in seeds]
            agg = aggregate(recs)
            agg["cnr"] = cnr
            rows.append(agg)
        result["geometries"][kind] = {"S1_fixed_sigma": rows}

        print(f"  S1 固定 σ={SIGMA_FIX} HU")
        print(f"  {'CNR':>5} {'Δ(HU)':>6} | {'PC ratio':>9} {'grad ratio':>10} "
              f"{'PC/grad':>8} | {'PC AUC':>7} {'grad AUC':>8} | {'PC d′':>7} {'grad d′':>8}")
        for r in rows:
            pr, gr = r["pc_ratio_mean"], r["grad_ratio_mean"]
            print(f"  {r['cnr']:5.2f} {r['A']:6.2f} | {pr:9.3f} {gr:10.3f} "
                  f"{pr/max(gr,1e-9):8.3f} | {r['pc_auc_mean']:7.3f} {r['grad_auc_mean']:8.3f} "
                  f"| {r['pc_dprime_mean']:7.3f} {r['grad_dprime_mean']:8.3f}")

        rows2 = []
        for sg in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
            recs = [one_trial(kind, 15.0, sg, s) for s in seeds]
            agg = aggregate(recs)
            agg["cnr"] = 15.0 / sg
            rows2.append(agg)
        result["geometries"][kind]["S2_fixed_amp"] = rows2
        print(f"  S2 固定 A=15 HU")
        print(f"  {'σ':>5} {'CNR':>6} | {'PC ratio':>9} {'grad ratio':>10} {'PC/grad':>8} "
              f"| {'PC AUC':>7} {'grad AUC':>8}")
        for r in rows2:
            pr, gr = r["pc_ratio_mean"], r["grad_ratio_mean"]
            print(f"  {r['sigma']:5.1f} {r['cnr']:6.2f} | {pr:9.3f} {gr:10.3f} "
                  f"{pr/max(gr,1e-9):8.3f} | {r['pc_auc_mean']:7.3f} {r['grad_auc_mean']:8.3f}")

    # ---------------- S3 无噪声对比度不变性 ----------------
    print("\nS3 无噪声（σ=0）：PC 应逐体素恒定，|∇f| 应随 A 线性缩放")
    inv = {}
    for kind in ("slab", "ball", "slab_soft"):
        obj, b, i = _CACHE[kind]
        ref_pc = ref_gr = None
        row = []
        for A in (5.0, 10.0, 20.0, 40.0):
            x = (BASE + A * obj).astype(np.float32)
            pc = compute_pc(x)
            gr = compute_grad(x)
            if ref_pc is None:
                ref_pc, ref_gr = pc, gr
            row.append(dict(
                A=A,
                pc_max_abs_diff=float(np.abs(pc - ref_pc).max()),
                pc_boundary_mean=float(pc[b].mean()),
                pc_interior_mean=float(pc[i].mean()),
                grad_boundary_mean=float(gr[b].mean()),
                grad_boundary_over_A=float(gr[b].mean()) / A,
                grad_interior_mean=float(gr[i].mean()),
            ))
        inv[kind] = row
        print(f"[{kind}]")
        for r in row:
            print(f"  A={r['A']:5.1f}  max|ΔPC|={r['pc_max_abs_diff']:.2e}  "
                  f"PC_b={r['pc_boundary_mean']:.4f} PC_i={r['pc_interior_mean']:.4f}  "
                  f"|∇f|_b={r['grad_boundary_mean']:8.3f}  |∇f|_b/A={r['grad_boundary_over_A']:.5f}  "
                  f"|∇f|_i={r['grad_interior_mean']:.3e}")
    result["S3_contrast_invariance"] = inv

    # ---------------- 判定 ----------------
    print("\n" + "-" * 74)
    print("判定（C1）：PC 相对 |∇f| 的优势是否随 CNR 下降而扩大？")
    verdict = {}
    for kind in ("slab", "ball", "slab_soft"):
        rows = result["geometries"][kind]["S1_fixed_sigma"]
        adv = [(r["cnr"], r["pc_ratio_mean"] / max(r["grad_ratio_mean"], 1e-9),
                r["pc_auc_mean"], r["grad_auc_mean"], r["pc_dprime_mean"], r["grad_dprime_mean"])
               for r in rows]
        lo, hi = adv[0], adv[-1]
        auc_win = sum(1 for a in adv if a[2] > a[3])
        verdict[kind] = dict(
            cnr_low=lo[0], ratio_adv_low=lo[1], ratio_adv_high=hi[1],
            ratio_advantage_grows_as_cnr_drops=bool(lo[1] > hi[1]),
            auc_pc_low=lo[2], auc_grad_low=lo[3],
            auc_pc_high=hi[2], auc_grad_high=hi[3],
            n_cnr_pc_auc_wins=auc_win, n_cnr_total=len(adv),
            dprime_advantage_low=lo[4] - lo[5], dprime_advantage_high=hi[4] - hi[5],
        )
        v = verdict[kind]
        print(f"  [{kind}] ratio 优势 CNR={lo[0]} → {lo[1]:.2f}×,  CNR={hi[0]} → {hi[1]:.2f}× "
              f"→ {'随 CNR 下降而扩大 (支持 C1)' if v['ratio_advantage_grows_as_cnr_drops'] else '随 CNR 上升而扩大 (不支持 C1)'}")
        print(f"           AUC: 低 CNR PC={lo[2]:.3f} vs grad={lo[3]:.3f} ; "
              f"高 CNR PC={hi[2]:.3f} vs grad={hi[3]:.3f} ; PC 在 {auc_win}/{len(adv)} 个 CNR 上 AUC 更高")
        print(f"           d′ 差值: 低 CNR {v['dprime_advantage_low']:+.3f} ; "
              f"高 CNR {v['dprime_advantage_high']:+.3f}")
    result["C1_verdict"] = verdict

    # ---------------- 作图（英文标签，避免缺 CJK 字体） ----------------
    C = {"pc": "#1D9E75", "grad": "#D85A30", "grad_norm": "#7F77DD", "grad_snr": "#BA7517"}
    L = {"pc": "Phase congruency (PFCR / PCSF-pc)",
         "grad": "|grad f| raw",
         "grad_norm": "|grad f| min-max normalized (PCSF-grad, actual)",
         "grad_snr": "|grad f| / sqrt(2) sigma (oracle noise floor)"}

    KINDS_PLOT = ("slab", "ball", "slab_soft")
    fig, axes = plt.subplots(3, 4, figsize=(21, 14.2))
    for r, kind in enumerate(KINDS_PLOT):
        rows = result["geometries"][kind]["S1_fixed_sigma"]
        cnr = np.array([x["cnr"] for x in rows])

        ax = axes[r, 0]
        for k in ("pc", "grad", "grad_norm", "grad_snr"):
            ax.errorbar(cnr, [x[f"{k}_ratio_mean"] for x in rows],
                        yerr=[x[f"{k}_ratio_std"] for x in rows],
                        marker="o", ms=4, lw=1.6, capsize=2, color=C[k], label=L[k])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("CNR = delta / sigma (log)")
        ax.set_ylabel("boundary / interior response ratio")
        ax.set_title(f"{kind} | S1 sigma=5 HU | ratio vs CNR")
        ax.grid(alpha=.25, which="both")
        if r == 0:
            ax.legend(fontsize=8, loc="upper left")

        ax = axes[r, 1]
        for k in ("pc", "grad", "grad_snr"):
            ax.errorbar(cnr, [x[f"{k}_auc_mean"] for x in rows],
                        yerr=[x[f"{k}_auc_std"] for x in rows],
                        marker="s", ms=4, lw=1.6, capsize=2, color=C[k], label=L[k])
        ax.axhline(0.5, color="gray", lw=0.9, ls="--")
        ax.set_xscale("log"); ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("CNR = delta / sigma (log)")
        ax.set_ylabel("AUC (boundary vs interior)")
        ax.set_title(f"{kind} | S1 | AUC vs CNR")
        ax.grid(alpha=.25)

        ax = axes[r, 2]
        for k in ("pc", "grad", "grad_snr"):
            ax.errorbar(cnr, [x[f"{k}_dprime_mean"] for x in rows],
                        yerr=[x[f"{k}_dprime_std"] for x in rows],
                        marker="^", ms=4, lw=1.6, capsize=2, color=C[k], label=L[k])
        ax.axhline(0, color="gray", lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("CNR = delta / sigma (log)")
        ax.set_ylabel("d-prime = (mu_b - mu_i) / sigma_i")
        ax.set_title(f"{kind} | S1 | d-prime vs CNR")
        ax.grid(alpha=.25)

        rows2 = result["geometries"][kind]["S2_fixed_amp"]
        sg = np.array([x["sigma"] for x in rows2])
        ax = axes[r, 3]
        for k in ("pc", "grad", "grad_norm"):
            ax.errorbar(sg, [x[f"{k}_ratio_mean"] for x in rows2],
                        yerr=[x[f"{k}_ratio_std"] for x in rows2],
                        marker="v", ms=4, lw=1.6, capsize=2, color=C[k], label=L[k])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("noise sigma (HU, log)")
        ax.set_ylabel("boundary / interior response ratio")
        ax.set_title(f"{kind} | S2 A=15 HU | ratio vs sigma")
        ax.grid(alpha=.25, which="both")

    fig.suptitle("4.1 Synthetic: phase congruency vs gradient magnitude under contrast/noise (C1)",
                 fontsize=14, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    fp = FIGS / "fig41_synthetic_pc_vs_grad.png"
    fig.savefig(fp, dpi=140)
    plt.close(fig)
    print(f"\n[fig] {fp}")

    save_json(result, "s41_synthetic.json")
    banner("4.1 完成")


if __name__ == "__main__":
    main()
