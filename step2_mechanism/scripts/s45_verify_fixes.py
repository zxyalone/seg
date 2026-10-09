#!/usr/bin/env python3
"""
Step 2 / 4.5 — P0 修复的逐项隔离验证（v2）

每个修改单独开关，避免混淆：

  V0  原始行为        kappa=0.0, cap=2^20, spacing=(1,1,1)
  V1  仅修 I2        kappa=0.0, cap=2^24, spacing=(1,1,1)
  V2  修 I2 + I6     kappa=0.0, cap=2^24, plan spacing   ← 新默认
  V3  再加 I1        kappa=0.5, cap=2^24, plan spacing   ← 已知负结果，保留取证

判定口径：boundary vs interior 的 AUC、boundary/peri-liver 比值、以及
"PC > 0 的体素占比"（稀疏性 = 该算子唯一的定位机制）。

另含 C 部分：prior 动态范围（I3）与 route 饱和诊断。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import (  # noqa: E402
    DATASET_JSON, FIGS, PLANS_JSON, banner, lesions_from_seg,
    load_raw_resampled, save_json, splits,
)
from nnunetv2.training.nnUNetTrainer.phaser.modules import PhaseCongruency  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, BASE, SOFT, R, MARGIN = 160, 60.0, 1.0, 60.0, 30.0
PLAN_SP = (1.0, 0.7791720032691956, 0.7791720032691956)

VARIANTS = {
    "V0 原始 (kappa=0, cap=2^20, sp=1)":      dict(kappa=0.0, max_voxels=1 << 20, spacing=(1.0, 1.0, 1.0)),
    "V1 仅修 I2 (cap=2^24)":                  dict(kappa=0.0, max_voxels=1 << 24, spacing=(1.0, 1.0, 1.0)),
    "V2 修 I2+I6 (新默认)":                    dict(kappa=0.0, max_voxels=1 << 24, spacing=PLAN_SP),
    "V3 +I1 (kappa=0.5, 负结果)":              dict(kappa=0.5, max_voxels=1 << 24, spacing=PLAN_SP),
}


def slab():
    z = np.arange(N, dtype=np.float32) - (N - 1) / 2.0
    zz = np.broadcast_to(z[:, None, None], (N, N, N))
    d = np.abs(zz)
    obj = (1.0 / (1.0 + np.exp((d - R) / SOFT))).astype(np.float32)
    return obj, (np.abs(d - R) <= 1.5), (d <= R - MARGIN)


def auc_fast(pos, neg, rng, n_max=40000):
    p = pos if pos.size <= n_max else rng.choice(pos, n_max, replace=False)
    n = neg if neg.size <= n_max else rng.choice(neg, n_max, replace=False)
    if p.size == 0 or n.size == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = allv.argsort()
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    sbv = np.zeros(cnt.size)
    np.add.at(sbv, inv, ranks)
    ranks = (sbv / cnt)[inv]
    return float((ranks[: p.size].sum() - p.size * (p.size + 1) / 2.0)
                 / (p.size * n.size))


def pc_of(mod, x):
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(x))[None, None].to(DEV).float()
        return mod(t)[0, 0].float().cpu().numpy().astype(np.float64)


def main():
    banner("Step 2 / 4.5 P0 修复的逐项隔离验证")
    import inspect
    _sig = inspect.signature(PhaseCongruency.__init__).parameters
    print("模块出厂默认值：max_voxels=%s  spacing=%s  noise_rel_kappa=%s" % (
        _sig['max_voxels'].default, _sig['spacing'].default, _sig['noise_rel_kappa'].default))
    print("  -> 应与原始行为一致：2^20 = %d, spacing None, kappa 0.0" % (1 << 20))
    mods = {k: PhaseCongruency(dims=3, n_scales=3, max_voxels=v["max_voxels"],
                               spacing=v["spacing"],
                               noise_rel_kappa=v["kappa"]).to(DEV).eval()
            for k, v in VARIANTS.items()}
    print("变体：")
    for k, v in VARIANTS.items():
        print(f"  {k:<38} kappa={v['kappa']}  cap=2^{int(np.log2(v['max_voxels']))}  spacing={v['spacing']}")
    result = {"variants": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in VARIANTS.items()}}

    # ---------------- A. 合成厚板 ----------------
    obj, bmask, imask = slab()
    SIGMA, CNRS = 5.0, [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    print("\n" + "=" * 100)
    print("A. 合成厚板（σ=5 HU）")
    print("=" * 100)
    hdr = f"{'CNR':>5} |" + "".join(f"{k.split()[0]:>20}" for k in VARIANTS)
    print(hdr)
    print(f"{'':>5} |" + "".join(f"{'AUC':>9}{'ratio':>9}" for _ in VARIANTS))
    rows = []
    urng = np.random.default_rng(0)
    for cnr in CNRS:
        A = cnr * SIGMA
        rng = np.random.default_rng(0)
        x = (BASE + A * obj + SIGMA * rng.normal(0, 1, obj.shape)).astype(np.float32)
        rec = {"cnr": cnr, "A": A, "variants": {}}
        line = f"{cnr:5.2f} |"
        for k, mod in mods.items():
            pc = pc_of(mod, x)
            d = mod._last_diag
            e, i = float(pc[bmask].mean()), float(pc[imask].mean())
            rec["variants"][k] = dict(
                T_over_den=d["T_over_den_mean"], T_kovesi_over_den=d["T_kovesi_over_den_mean"],
                pc_edge=e, pc_interior=i, ratio=e / max(i, 1e-9),
                auc=auc_fast(pc[bmask], pc[imask], urng),
                pc_frac_nonzero=d["pc_frac_nonzero"], num_over_den=d["num_over_den_mean"])
            line += f"{rec['variants'][k]['auc']:9.3f}{rec['variants'][k]['ratio']:9.1f}"
        rows.append(rec)
        print(line)
    result["A_synthetic"] = rows

    # 边缘/内部的 num/den 重叠度（解释为什么比值本身不可分）
    A = 4 * SIGMA
    rng = np.random.default_rng(0)
    x = (BASE + A * obj + SIGMA * rng.normal(0, 1, obj.shape)).astype(np.float32)
    with torch.no_grad():
        t = torch.from_numpy(x)[None, None].to(DEV).float()
        pc = None
    print("\n  num/den 的重叠（CNR=4，V0 配置）：")
    for k in list(VARIANTS)[:1] + list(VARIANTS)[-1:]:
        pc_raw = pc_of(mods[k], x)
        nd = mods[k]._last_diag["num_over_den_mean"]
        print(f"    {k:<38} 全图 num/den 均值={nd:.4f}  PC 均值={mods[k]._last_diag['pc_mean']:.4f}")
    # 直接测 edge/interior 的 num/den（用 kappa=0 时的 pc + kappa 反推不精确，故用大 kappa 探测）
    probe = PhaseCongruency(dims=3, n_scales=3, max_voxels=1 << 24,
                            spacing=(1.0, 1.0, 1.0), noise_rel_kappa=0.999).to(DEV).eval()
    pc_probe = pc_of(probe, x)
    # pc = num/den - kappa  (当 kappa*den > T_kovesi 时)
    nd_edge = float(pc_probe[bmask].mean()) + 0.999
    nd_int = float(pc_probe[imask].mean()) + 0.999
    print(f"    边缘 num/den ≈ {nd_edge:.4f}   内部 num/den ≈ {nd_int:.4f}   "
          f"重叠 => 比值不可分")
    result["A_num_over_den"] = dict(edge=nd_edge, interior=nd_int)

    # 对比度不变性（每个变体都必须保持）
    print("\n  A2. 对比度不变性（无噪声，max|ΔPC| 相对 A=5 HU）")
    inv = {}
    for k, mod in mods.items():
        ref = pc_of(mod, (BASE + 5.0 * obj).astype(np.float32))
        dd = [float(np.abs(pc_of(mod, (BASE + a * obj).astype(np.float32)) - ref).max())
              for a in (10.0, 20.0, 40.0)]
        inv[k] = dd
        print(f"    {k:<38} {['%.2e' % v for v in dd]}  -> {'OK' if max(dd) < 1e-2 else '破坏!'}")
    result["A2_contrast_invariance"] = inv

    # ---------------- B. 真实病例 ----------------
    tr, va, te = splits()
    case, patch = None, None
    for c in va:
        img, seg = load_raw_resampled(c)
        les = lesions_from_seg(seg, 2000)
        if not les:
            continue
        _, mask, size = les[0]
        ctr = np.argwhere(mask).mean(0).round().astype(int)
        lo = np.maximum(ctr - 64, 0)
        hi = np.minimum(lo + 128, np.array(seg.shape))
        lo = np.maximum(hi - 128, 0)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        pi, ps = img[sl], seg[sl]
        if pi.shape == (128,) * 3 and (ps == 2).sum() > 500 and (ps == 0).sum() > 500:
            case, patch = c, (pi.copy(), ps.copy())
            break
    print("\n" + "=" * 100)
    print(f"B. 真实病例 patch（{case}，128³，HU）")
    print("=" * 100)
    if case is None:
        print("  [skip]")
    else:
        from scipy import ndimage as ndi
        pi, ps = patch
        tum = ps == 2
        d_in = ndi.distance_transform_edt(tum)
        d_out = ndi.distance_transform_edt(~tum)
        bnd = tum & (d_in <= 1.0)
        peri = (~tum) & (ps == 1) & (d_out >= 2) & (d_out <= 6)
        print(f"  tumour={int(tum.sum())}  boundary={int(bnd.sum())}  peri-liver={int(peri.sum())}")
        print(f"\n  {'变体':<38} {'T/den':>7} {'PC均值':>8} {'PC>0':>8} "
              f"{'PC边界':>8} {'PC外周':>8} {'边界/外周':>10}")
        brows = {}
        for k, mod in mods.items():
            pc = pc_of(mod, pi)
            d = mod._last_diag
            be, pe = float(pc[bnd].mean()), float(pc[peri].mean())
            brows[k] = dict(T_over_den=d["T_over_den_mean"],
                            T_kovesi_over_den=d["T_kovesi_over_den_mean"],
                            pc_mean=d["pc_mean"], pc_frac_nonzero=d["pc_frac_nonzero"],
                            pc_boundary=be, pc_peri=pe, ratio=be / max(pe, 1e-9))
            print(f"  {k:<38} {d['T_over_den_mean']:7.3f} {d['pc_mean']:8.4f} "
                  f"{d['pc_frac_nonzero']:8.1%} {be:8.4f} {pe:8.4f} {be/max(pe,1e-9):10.3f}")
        result["B_real_patch"] = dict(case=case, variants=brows)

    # ---------------- C. 模型级 ----------------
    print("\n" + "=" * 100)
    print("C. 模型级检查（I3 prior 语义 + route 饱和诊断）")
    print("=" * 100)
    os.environ.update({
        "PHASER_PFCR": "1", "PHASER_PFCR_USE_PRIOR": "1", "PHASER_SMALL_HEAD": "1",
        "PHASER_PC_SCALES": "3", "PHASER_PCSF": "0", "PHASER_LWFD": "0",
        "PHASER_ACG": "0", "PHASER_T_RAF": "none", "PHASER_LA_ACG": "0",
        "PHASER_FUSION": "none", "VOCOMNI_NNUNET_CKPT": "",
    })
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_phaser import (  # noqa: E402
        nnUNetTrainer_phaser, resolve_pc_spacing,
    )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402

    pm = PlansManager(json.load(open(PLANS_JSON)))
    cm = pm.get_configuration("3d_fullres")
    model = nnUNetTrainer_phaser.build_network_architecture(
        pm, json.load(open(DATASET_JSON)), cm, 1, False).to(DEV).eval()

    # 必须加载 Copy2 权重，否则 router[-1] 仍是零初始化、route 恰好是 (1/3,1/3,1/3)，
    # 与 4.4 的实测不可比。
    from common import CKPTS  # noqa: E402
    ck = torch.load(CKPTS["copy2_ep48"], map_location="cpu")
    sd = ck.get("network_weights", ck)
    cur = model.state_dict()
    new_sd = {k: v for k, v in sd.items()
              if k in cur and cur[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"  已加载 Copy2 权重: {len(new_sd)} keys, missing={len(missing)}, "
          f"unexpected={len(unexpected)}, epoch={ck.get('current_epoch')}")
    print(f"  PC spacing 解析 = {resolve_pc_spacing(cm, 3)}   plan spacing = {cm.spacing}")
    print(f"  pfcr[0].pc: max_voxels={model.pfcr_modules[0].pc.max_voxels} "
          f"({model.pfcr_modules[0].pc.max_voxels == 1 << 24})  "
          f"kappa={model.pfcr_modules[0].pc.noise_rel_kappa}  "
          f"spacing={model.pfcr_modules[0].pc.spacing}")

    if case is not None:
        pi, ps = patch
        t = torch.from_numpy(pi)[None, None].to(DEV).float()
        with torch.no_grad():
            skips = list(model.encoder(t))
            logits = model.small_head(skips)
            old_prior = torch.sigmoid(torch.sigmoid(logits[:, 1:2]))
            new_prior = torch.softmax(logits, dim=1)[:, 1:2]
            model(t)
        tm = torch.from_numpy((ps == 2).astype(np.float32))[None, None].to(DEV) > .5
        bm = torch.from_numpy((ps == 0).astype(np.float32))[None, None].to(DEV) > .5
        print("\n  I3 prior 动态范围（同一 logits，Copy2 权重；head 未训练，故判别力仍为负）")
        for nm, pr in (("旧 sigmoid(sigmoid(z))", old_prior), ("新 softmax(z)[1]", new_prior)):
            print(f"    {nm:<24} range=[{float(pr.min()):.4f}, {float(pr.max()):.4f}]  "
                  f"std={float(pr.std()):.4f}  肿瘤均值={float(pr[tm].mean()):.4f}  "
                  f"背景均值={float(pr[bm].mean()):.4f}  差={float(pr[tm].mean()-pr[bm].mean()):+.4f}")
        d0 = model.pfcr_modules[0].pc._last_diag
        print(f"\n  pfcr[0] PC 诊断: T/den={d0['T_over_den_mean']:.4f} "
              f"PC 均值={d0['pc_mean']:.4f} 非零={d0['pc_frac_nonzero']:.2%} "
              f"num/den={d0['num_over_den_mean']:.4f}")
        rt = model._last_pfcr_routes[0][0]
        tm2, bm2 = tm[0, 0], bm[0, 0]
        print("  route[0] 各通道 [low, mid, high]:")
        for c in range(3):
            ch = rt[c]
            print(f"    ch{c}: mean={float(ch.mean()):.6f} std={float(ch.std()):.3e} "
                  f"min={float(ch.min()):.6f} max={float(ch.max()):.6f} "
                  f"肿瘤内={float(ch[tm2].mean()):.6f} 背景内={float(ch[bm2].mean()):.6f} "
                  f"选择差={float(ch[tm2].mean()-ch[bm2].mean()):+.2e}")
        result["C_model"] = dict(
            resolved_spacing=str(resolve_pc_spacing(cm, 3)),
            pc_max_voxels=int(model.pfcr_modules[0].pc.max_voxels),
            noise_rel_kappa=float(model.pfcr_modules[0].pc.noise_rel_kappa),
            pc_spacing=list(model.pfcr_modules[0].pc.spacing),
            prior_old=dict(rng=[float(old_prior.min()), float(old_prior.max())],
                           std=float(old_prior.std()),
                           gap=float(old_prior[tm].mean() - old_prior[bm].mean())),
            prior_new=dict(rng=[float(new_prior.min()), float(new_prior.max())],
                           std=float(new_prior.std()),
                           gap=float(new_prior[tm].mean() - new_prior[bm].mean())),
            diag_scale0={k: v for k, v in d0.items() if not isinstance(v, list)},
            route0_mean=[float(rt[c].mean()) for c in range(3)],
            route0_std=[float(rt[c].std()) for c in range(3)],
            route0_tumour=[float(rt[c][tm2].mean()) for c in range(3)],
            route0_bg=[float(rt[c][bm2].mean()) for c in range(3)],
        )

    # ---------------- 作图 ----------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.9))
    cnrs = [r["cnr"] for r in rows]
    COLS = {"V0 原始 (kappa=0, cap=2^20, sp=1)": "#888780",
            "V1 仅修 I2 (cap=2^24)": "#7F77DD",
            "V2 修 I2+I6 (新默认)": "#1D9E75",
            "V3 +I1 (kappa=0.5, 负结果)": "#D85A30"}
    ASCII = {"V0 原始 (kappa=0, cap=2^20, sp=1)": "V0 baseline (kappa=0, cap=2^20, d=1)",
             "V1 仅修 I2 (cap=2^24)": "V1 +remove cap (cap=2^24)",
             "V2 修 I2+I6 (新默认)": "V2 +spacing (I2+I6)",
             "V3 +I1 (kappa=0.5, 负结果)": "V3 +kappa=0.5 (rejected)"}
    for k in VARIANTS:
        axes[0].plot(cnrs, [r["variants"][k]["auc"] for r in rows], "o-", color=COLS[k], label=ASCII[k])
        axes[1].plot(cnrs, [min(r["variants"][k]["ratio"], 1e5) for r in rows], "s-", color=COLS[k], label=ASCII[k])
        axes[2].plot(cnrs, [min(r["variants"][k]["T_over_den"], 5) for r in rows], "^-", color=COLS[k], label=ASCII[k])
        axes[3].plot(cnrs, [r["variants"][k]["pc_frac_nonzero"] for r in rows], "v-", color=COLS[k], label=ASCII[k])
    axes[0].axhline(0.5, color="gray", ls="--", lw=1)
    axes[0].set_ylim(0.2, 1.02); axes[0].set_ylabel("AUC (boundary vs interior)")
    axes[0].set_title("A1. Localisation")
    axes[1].set_yscale("log"); axes[1].set_ylabel("boundary / interior ratio")
    axes[1].set_title("A2. Ratio (sparsity-driven)")
    axes[2].axhline(1.0, color="red", ls="--", lw=1)
    axes[2].set_ylabel("T / den"); axes[2].set_title("A3. Threshold ratio")
    axes[3].set_ylabel("fraction of voxels with PC > 0")
    axes[3].set_title("A4. Sparsity")
    for a in axes:
        a.set_xscale("log"); a.set_xlabel("CNR (log)"); a.grid(alpha=.25)
        a.legend(fontsize=7)
    fig.suptitle("4.5 Fix verification, isolated: I2 (cap) and I6 (spacing) help; I1 (kappa) hurts (synthetic slab, sigma=5 HU)",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    fp = FIGS / "fig45_fix_verification.png"
    fig.savefig(fp, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {fp}")
    save_json(result, "s45_fix_verification.json")
    banner("4.5 完成")


if __name__ == "__main__":
    main()
