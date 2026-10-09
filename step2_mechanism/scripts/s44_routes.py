#!/usr/bin/env python3
"""
Step 2 / 4.4 — 路由图分析（检验 C2 / budget）

命题 C2（结构）：相位证据用于 Haar 低/中/高频路由，优于只用它做普通 skip gate。
证伪：PFCR 与 PCSF(pc) 无显著差异。
命题（预算）：路由是否按 (病灶 0.20/0.40/0.40，背景 0.70/0.22/0.08) 分配频带。

本节的**不训练**可做部分：
  1. 真实病例前向，取出每层 PFCR 的 route 张量，统计
        · 在肿瘤 / 肝实质 / 背景内的 low-mid-high 均值（是否按语义分配）
        · route 的空间标准差与极差（是否退化为空间常数）
        · route 的语义选择性 d_c = μ(route_c | 肿瘤) − μ(route_c | 背景)
  2. 与 phase_frequency_budget_loss 的目标预算对比。
  3. 典型病例可视化（CT + GT + 三路 route 热图）。
  4. 跨 checkpoint 对比（含 Phase B 已开启 budget 监督的 run）。

注意：测试集 27 例保持锁定，patch 取自 fold 0 验证集病例。

勘误（2026-10-09）：旧版逐尺度打印的"语义选择性"先对 low/mid/high 三个频带的
**有符号**差值求平均再取绝对值。route 逐体素和为 1，三带差值之和恒为 0，
因此旧值（10⁻⁹~10⁻⁸）只是浮点残差，不是选择性。正确定义是逐频带
|μ(route_b|肿瘤) − μ(route_b|背景)| 的均值（以及三带 L1 和）。
summary 里的 mean_abs_selectivity 一直是逐频带取绝对值，不受影响。
self_test() 在合成 route 上验证该统计量。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import (  # noqa: E402
    CKPTS, DATASET_JSON, FIGS, NNUNET, PLANS_JSON, banner, lesions_from_seg,
    load_input_and_seg, save_json, splits,
)

# 必须在 build_network_architecture 之前设置（函数从环境变量读开关）
os.environ.update({
    "PHASER_PFCR": "1", "PHASER_PFCR_USE_PRIOR": "1", "PHASER_SMALL_HEAD": "1",
    "PHASER_PC_SCALES": "3", "PHASER_PC_MAX_VOXELS": str(1 << 20),
    "PHASER_PCSF": "0", "PHASER_LWFD": "0", "PHASER_ACG": "0",
    "PHASER_T_RAF": "none", "PHASER_LA_ACG": "0", "PHASER_FUSION": "none",
    "VOCOMNI_NNUNET_CKPT": "",
})
import matplotlib  # noqa: E402
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH = 128
SEED = 0
BG_BUDGET = np.array([0.70, 0.22, 0.08])
LESION_BUDGET = np.array([0.20, 0.40, 0.40])
BAND = ("low", "mid", "high")


def build_model():
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_phaser import nnUNetTrainer_phaser
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = json.load(open(PLANS_JSON))
    djs = json.load(open(DATASET_JSON))
    pm = PlansManager(plans)
    cm = pm.get_configuration("3d_fullres")
    m = nnUNetTrainer_phaser.build_network_architecture(
        pm, djs, cm, num_input_channels=1, enable_deep_supervision=False)
    return m.to(DEV).eval()


def load_weights(model, path):
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("network_weights", ck)
    cur = model.state_dict()
    ok, skip = 0, []
    new = {}
    for k, v in sd.items():
        kk = k[7:] if k.startswith("module.") else k
        if kk in cur and cur[kk].shape == v.shape:
            new[kk] = v
            ok += 1
        else:
            skip.append(kk)
    missing, unexpected = model.load_state_dict(new, strict=False)
    return dict(loaded=ok, skipped=len(skip), missing=len(missing),
                epoch=ck.get("current_epoch"), best_ema=ck.get("_best_ema"))


def make_patches(cases, n_patch=3, min_tumor=800):
    """每个病例取一个含肿瘤的 128³ patch，返回 (case, patch_img, patch_seg)。"""
    out = []
    for case in cases:
        img, seg = load_input_and_seg(case)
        les = [l for l in lesions_from_seg(seg, 200)]
        if not les:
            continue
        # 优先取中等大小病灶，保证 patch 内同时有肿瘤/肝/背景
        _, mask, size = sorted(les, key=lambda t: abs(t[2] - 4000))[0]
        if size < min_tumor:
            continue
        c = np.argwhere(mask).mean(0).round().astype(int)
        lo = np.maximum(c - PATCH // 2, 0)
        hi = np.minimum(lo + PATCH, np.array(seg.shape))
        lo = np.maximum(hi - PATCH, 0)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        pi, ps = img[sl], seg[sl]
        if pi.shape != (PATCH,) * 3:
            continue
        if (ps == 2).sum() < 200 or (ps == 0).sum() < 500:
            continue
        out.append((case, pi.copy(), ps.copy()))
        if len(out) >= n_patch:
            break
    return out


@torch.no_grad()
def forward_routes(model, img):
    t = torch.from_numpy(img)[None, None].to(DEV).float()
    model(t)
    routes = [r.detach().float().cpu().numpy()[0] for r in model._last_pfcr_routes]
    return routes


def analyse_patch(routes, seg, patch_img):
    """返回 per-scale 的 route 统计。"""
    tumor = torch.from_numpy((seg == 2).astype(np.float32))[None, None]
    liver = torch.from_numpy((seg == 1).astype(np.float32))[None, None]
    bg = torch.from_numpy((seg == 0).astype(np.float32))[None, None]
    recs = []
    for s, r in enumerate(routes):
        D, H, W = r.shape[1:]
        size = (D, H, W)

        def sel(m):
            return F.interpolate(m, size=size, mode="nearest")[0, 0] > 0.5

        mt, ml, mb = sel(tumor), sel(liver), sel(bg)
        rec = dict(scale=s, route_shape=list(size),
                   n_tumor=int(mt.sum()), n_liver=int(ml.sum()), n_bg=int(mb.sum()),
                   bands={})
        for c, b in enumerate(BAND):
            ch = r[c]
            e = dict(
                global_mean=float(ch.mean()),
                spatial_std=float(ch.std()),
                spatial_range=float(ch.max() - ch.min()),
                tumor_mean=float(ch[mt].mean()) if mt.sum() else float("nan"),
                liver_mean=float(ch[ml].mean()) if ml.sum() else float("nan"),
                bg_mean=float(ch[mb].mean()) if mb.sum() else float("nan"),
            )
            e["selectivity"] = e["tumor_mean"] - e["bg_mean"] if mt.sum() and mb.sum() else float("nan")
            rec["bands"][b] = e
        # 选择性：逐频带取绝对值（三带有符号差值之和恒为 0，不能先平均）
        sels = [rec["bands"][b]["selectivity"] for b in BAND]
        rec["selectivity_mean_abs"] = float(np.mean(np.abs(sels)))
        rec["selectivity_l1"] = float(np.sum(np.abs(sels)))
        # 预算偏差
        rec["route_mean_global"] = [rec["bands"][b]["global_mean"] for b in BAND]
        rec["route_mean_tumor"] = [rec["bands"][b]["tumor_mean"] for b in BAND]
        rec["route_mean_bg"] = [rec["bands"][b]["bg_mean"] for b in BAND]
        rec["dev_vs_bg_budget"] = float(np.abs(np.array(rec["route_mean_bg"]) - BG_BUDGET).sum())
        rec["dev_vs_lesion_budget"] = float(np.abs(np.array(rec["route_mean_tumor"]) - LESION_BUDGET).sum())
        # 空间变化幅度（相对其均值）
        rec["rel_spatial_std"] = float(np.mean([
            rec["bands"][b]["spatial_std"] / max(rec["bands"][b]["global_mean"], 1e-9)
            for b in BAND]))
        recs.append(rec)
    return recs


def self_test():
    """合成 route：肿瘤内 (0.2,0.4,0.4)、背景 (0.7,0.22,0.08)，选择性已知。"""
    seg = np.zeros((16, 16, 16), np.int16)
    seg[4:12, 4:12, 4:12] = 2
    r = np.empty((3,) + seg.shape, np.float32)
    for c, (t, b) in enumerate(zip(LESION_BUDGET, BG_BUDGET)):
        r[c] = np.where(seg == 2, t, b)
    rec = analyse_patch([r], seg, None)[0]
    per_band = [abs(rec["bands"][b]["selectivity"]) for b in BAND]
    want = np.abs(LESION_BUDGET - BG_BUDGET)
    assert np.allclose(per_band, want, atol=1e-6), (per_band, want)
    assert abs(rec["selectivity_mean_abs"] - want.mean()) < 1e-6
    assert abs(rec["selectivity_l1"] - want.sum()) < 1e-6
    # 旧（错误）统计量：有符号均值恒为 0
    assert abs(np.mean([rec["bands"][b]["selectivity"] for b in BAND])) < 1e-6
    print(f"[self-test] route 选择性统计正确：逐带 |Δ| = {np.round(per_band, 3).tolist()}")


def main():
    self_test()
    banner("Step 2 / 4.4 路由图分析 — PFCR 路由是否按语义分配频带（C2 / budget）")
    tr, va, te = splits()
    print(f"队列：fold 0 验证集（测试集 {len(te)} 例锁定）")
    print(f"预算目标：背景 {BG_BUDGET.tolist()}  病灶 {LESION_BUDGET.tolist()}")
    print(f"device={DEV}  patch={PATCH}³  route 分辨率随 encoder skip 递减\n")

    patches = make_patches(va)
    print("选取的 patch：")
    for case, pi, ps in patches:
        print(f"  {case}  tumor={int((ps==2).sum())}  liver={int((ps==1).sum())}  bg={int((ps==0).sum())}")

    model = build_model()
    n_param = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量 {n_param/1e6:.2f} M；PFCR 模块数 {len(model.pfcr_modules)}")

    results = {}
    order = ["copy2_ep48", "retrain_ema_best", "repro0905_ema_best",
             "scratch_ema_best", "copy1"]
    for tag in order:
        path = CKPTS.get(tag)
        if path is None or not path.is_file():
            print(f"[skip] {tag}")
            continue
        info = load_weights(model, path)
        print(f"\n{'='*74}\n{tag}  epoch={info['epoch']}  best_ema={info['best_ema']}  "
              f"loaded={info['loaded']} skipped={info['skipped']} missing={info['missing']}\n{'='*74}")
        per_case = []
        for case, pi, ps in patches:
            routes = forward_routes(model, pi)
            recs = analyse_patch(routes, ps, pi)
            per_case.append(dict(case=case, scales=recs))
        results[tag] = dict(meta=info, per_case=per_case)

        # 打印：每个 scale 的 route 均值（肿瘤 / 背景）与空间 std
        print(f"{'scale':>5} {'shape':>12} | " +
              " ".join(f"{b+'_T':>8}" for b in BAND) + " | " +
              " ".join(f"{b+'_BG':>8}" for b in BAND) + " | " +
              " ".join(f"{b+'_std':>10}" for b in BAND))
        nsc = len(per_case[0]["scales"])
        for s in range(nsc):
            shp = per_case[0]["scales"][s]["route_shape"]

            def avg_scale(key, scale=s):
                return float(np.nanmean([c["scales"][scale]["bands"][b][key]
                                         for c in per_case for b in BAND]))

            def band_avg(key, b, scale=s):
                return float(np.nanmean([c["scales"][scale]["bands"][b][key]
                                         for c in per_case]))

            print(f"{s:>5} {str(shp):>12} | " +
                  " ".join(f"{band_avg('tumor_mean', b):8.4f}" for b in BAND) + " | " +
                  " ".join(f"{band_avg('bg_mean', b):8.4f}" for b in BAND) + " | " +
                  " ".join(f"{band_avg('spatial_std', b):10.2e}" for b in BAND))
        # 汇总指标
        for s in range(nsc):
            gstd = np.mean([c["scales"][s]["bands"][b]["spatial_std"] for c in per_case for b in BAND])
            gmean = np.mean([c["scales"][s]["bands"][b]["global_mean"] for c in per_case for b in BAND])
            sel = np.nanmean([c["scales"][s]["selectivity_mean_abs"] for c in per_case])
            sel_l1 = np.nanmean([c["scales"][s]["selectivity_l1"] for c in per_case])
            print(f"      scale {s}: 空间std/均值={gstd/max(gmean,1e-9):.3e}  "
                  f"语义选择性 逐带|肿瘤-背景|均值={sel:.4e}  L1={sel_l1:.4e}")
        dbg = np.mean([c["scales"][s]["dev_vs_bg_budget"] for c in per_case for s in range(nsc)])
        dls = np.mean([c["scales"][s]["dev_vs_lesion_budget"] for c in per_case for s in range(nsc)])
        print(f"  平均 |route − 背景预算|_1 = {dbg:.4f}   |route_tumor − 病灶预算|_1 = {dls:.4f}")
        results[tag]["summary"] = dict(
            mean_dev_vs_bg_budget=float(dbg), mean_dev_vs_lesion_budget=float(dls),
            mean_rel_spatial_std=float(np.mean([c["scales"][s]["rel_spatial_std"]
                                                for c in per_case for s in range(nsc)])),
            mean_abs_selectivity=float(np.nanmean([
                abs(c["scales"][s]["bands"][b]["selectivity"])
                for c in per_case for s in range(nsc) for b in BAND])))

    # ---------------- 可视化（Copy2） ----------------
    ref = "copy2_ep48"
    if ref in results:
        load_weights(model, CKPTS[ref])
        nrow, ncol = len(patches), 4
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow))
        if nrow == 1:
            axes = axes[None]
        for r, (case, pi, ps) in enumerate(patches):
            routes = forward_routes(model, pi)
            r0 = routes[0]
            zc = int(np.argmax((ps == 2).sum(axis=(1, 2))))
            ct = pi[zc]
            gt = ps[zc]
            axes[r, 0].imshow(ct, cmap="gray", vmin=np.percentile(ct, 1), vmax=np.percentile(ct, 99))
            axes[r, 0].imshow(np.ma.masked_where((gt != 1) & (gt != 2), gt),
                              cmap=ListedColormap(["#4C9EE8", "#E24B4A"]), alpha=.45,
                              vmin=1, vmax=2)
            axes[r, 0].set_title(f"{case}  CT + GT (liver blue / tumor red)")
            for c, b in enumerate(BAND):
                m = r0[c][zc]
                im = axes[r, 1 + c].imshow(m, cmap="inferno")
                axes[r, 1 + c].set_title(f"route {b} (scale0, z={zc})\nmin={m.min():.4f} "
                                         f"max={m.max():.4f} mean={m.mean():.4f}")
                plt.colorbar(im, ax=axes[r, 1 + c], fraction=.046)
            for a in axes[r]:
                a.set_xticks([]); a.set_yticks([])
        fig.suptitle("4.4 PFCR route maps (Copy2): low/mid/high band weight at the finest injected scale",
                     fontsize=13, y=.998)
        fig.tight_layout()
        fp = FIGS / "fig44_route_maps.png"
        fig.savefig(fp, dpi=135, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[fig] {fp}")

        # 汇总对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
        tags = [t for t in order if t in results]
        x = np.arange(len(tags))
        ax = axes[0]
        for c, b in enumerate(BAND):
            v = [np.nanmean([cc["scales"][s]["bands"][b]["bg_mean"]
                             for cc in results[t]["per_case"]
                             for s in range(len(cc["scales"]))]) for t in tags]
            ax.bar(x + (c - 1) * .27, v, .27, label=f"{b}")
            ax.axhline(BG_BUDGET[c], color=f"C{c}", ls="--", lw=.9)
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("mean route in BACKGROUND region")
        ax.set_title("Background route vs budget (dashed)")
        ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

        ax = axes[1]
        for c, b in enumerate(BAND):
            v = [np.nanmean([cc["scales"][s]["bands"][b]["tumor_mean"]
                             for cc in results[t]["per_case"]
                             for s in range(len(cc["scales"]))]) for t in tags]
            ax.bar(x + (c - 1) * .27, v, .27, label=f"{b}")
            ax.axhline(LESION_BUDGET[c], color=f"C{c}", ls="--", lw=.9)
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("mean route in TUMOR region")
        ax.set_title("Lesion route vs budget (dashed)")
        ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

        ax = axes[2]
        rel = [results[t]["summary"]["mean_rel_spatial_std"] for t in tags]
        sel = [results[t]["summary"]["mean_abs_selectivity"] for t in tags]
        ax.bar(x - .19, rel, .38, label="rel. spatial std of route", color="#534AB7")
        ax.bar(x + .19, sel, .38, label="|mean difference tumor - bg|", color="#1D9E75")
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=25, ha="right", fontsize=8)
        ax.set_title("Route spatial variation and semantic selectivity")
        ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

        fig.suptitle("4.4 Route statistics across checkpoints (fold-0 validation patches)",
                     fontsize=13, y=1.0)
        fig.tight_layout()
        fp = FIGS / "fig44_route_stats.png"
        fig.savefig(fp, dpi=135, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {fp}")

    results["protocol"] = dict(cohort="fold0 validation", test_set_locked=True,
                              patch=PATCH, budget_bg=BG_BUDGET.tolist(),
                              budget_lesion=LESION_BUDGET.tolist(),
                              pc_max_voxels=1 << 20, bands=list(BAND),
                              patches=[p[0] for p in patches])
    save_json(results, "s44_routes.json")
    banner("4.4 完成")


if __name__ == "__main__":
    main()
