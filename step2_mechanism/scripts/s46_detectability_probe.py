#!/usr/bin/env python3
"""
Step 3 / 探针 P — 病灶"可检出性"线索是否存在（方法重设计的进入条件）

背景
----
验证集分析显示，A0/A1/A6/A7 的主要漏检是低 CNR 病灶（低 CNR 层召回 64%，中/高 83–87%），
包括 LiTS_039 中直径 87 mm、CNR 0.16 的病灶；任何变体都没有改变这一层。
重设计的候选模块用 Rose 型可检出性统计量代替相位一致性：在中心窗口与周围环带之间做
按噪声归一化、随窗口体积增长的两样本对比，再加一个纹理（高通残差方差比）项。
本探针在**训练集病灶**上检验这些线索本身是否携带信息，再决定是否把它做成网络模块。
计划与事先登记的判据见 docs/方法重设计_可检出性计划.md。

特征（全部在肝脏掩膜内做归一化卷积；HU 量纲；重采样到计划间距）
-------------------------------------------------------------
对中心高斯 σ_c ∈ {1, 2, 4, 8, 16} mm、环带 σ_s = 3σ_c：
  μ_c, σ²_c  中心窗口内的加权均值、方差
  μ_a, σ²_a  环带核 A = (G_s − k·G_c)/(1 − k)，k = Π(σ_c/σ_s)，使 A(0) = 0 且 A ≥ 0
  N_c, N_a   核的有效样本数 (Σw)²/Σw²（白噪声下的独立体素数），按窗口落在肝内的比例折减
  t_r = (μ_c − μ_a) / sqrt(σ²_c/N_c + σ²_a/N_a)   Welch 型两样本统计量（可检出性 d′）
  v_r = ½·log(P_c / P_a)                          纹理：高通残差 h = x − G_1mm*x 的局部功率比
每个尺度的量都用该病例"远离病灶的肝实质"做经验零分布标准化（中位数 / 1.4826·MAD），
以吸收 CT 噪声的空间相关性和肝实质本身的不均匀性，然后：
  I    细尺度强度：σ = 1 体素平滑后的 HU，标准化后取 |·|（基线）
  Ims  多尺度强度：各尺度 μ_c 标准化后取 max|·|（只做面积积分，没有局部噪声归一化）
  D    可检出性：各尺度 t_r 标准化后取 max|·|
  T    纹理：各尺度 v_r 标准化后取 max|·|
  DT   max(D, T)

每个病灶的指标
--------------
  AUC_bg     主指标（检出）：病灶体素 vs 同病例远处肝实质（label==1，距任何病灶 > 15 mm）
  hit        病灶内特征中位数 > 远处肝实质的 99% 分位数
  AUC_local  次指标（定位）：病灶体素 vs 病灶外 3–15 mm 的肝实质环。大尺度特征在边界附近会扩散，
             这一指标天然偏向细尺度特征，只作参考
CNR 定义与 s42 完全一致（病灶外 2–6 体素的肝实质环），可直接套用冻结的三分位阈值 1.2785 / 1.9208。

用法
----
  python s46_detectability_probe.py --selftest      # 合成数据自检（本地可跑，不需要数据）
  python s46_detectability_probe.py --cohort train   # 主分析：fold 0 训练集 83 例（服务器）
  python s46_detectability_probe.py --cases LiTS_039 LiTS_094 --montage
        # 目视检查用的切片图；这两例属于验证集，结果不进入任何进入/停止判据；测试集病例拒绝运行

只读：不写 nnUNet_data 下任何文件；产出写入 step2_mechanism/{out,figs}。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

SCALES_MM = (1.0, 2.0, 4.0, 8.0, 16.0)
SURROUND_RATIO = 3.0
HIGHPASS_MM = 1.0
MIN_LESION = 50                       # 与 s42 相同（体素）
CNR_RING = (2, 6)                     # 与 s42 相同（体素）
CNR_TERTILES = (1.2785, 1.9208)       # 在 518 个训练病灶上冻结（s42）
LOCAL_RING_MM = (3.0, 15.0)
OTHER_LESION_GAP_MM = 3.0
BACKGROUND_GAP_MM = 15.0
MIN_LIVER_FRACTION = 0.3              # 窗口内肝脏占比低于此值视为无效
MAX_SAMPLES = 20000
FEATURES = ("I", "Ims", "D", "T", "DT")


# ----------------------------------------------------------------------------
# 算子
# ----------------------------------------------------------------------------
def gauss_neff(sigma_vox):
    """连续各向异性高斯核的有效样本数 (Σw)²/Σw² = Π 2√π σ_i（体素）。"""
    return float(np.prod([2.0 * np.sqrt(np.pi) * s for s in sigma_vox]))


def annulus_neff(sc, ss):
    """环带核 A = (G_s − k G_c)/(1 − k) 的有效样本数（连续近似，解析计算）。"""
    k = float(np.prod([a / b for a, b in zip(sc, ss)]))

    def cross(a, b):  # Σ G_a G_b
        return float(np.prod([1.0 / np.sqrt(2.0 * np.pi * (x * x + y * y)) for x, y in zip(a, b)]))

    s2 = (cross(ss, ss) - 2 * k * cross(ss, sc) + k * k * cross(sc, sc)) / (1 - k) ** 2
    return 1.0 / s2


def block_mean(a, f):
    if f == 1:
        return a
    a = np.pad(a, [(0, (-n) % f) for n in a.shape], mode="constant")
    s = a.shape
    return a.reshape(s[0] // f, f, s[1] // f, f, s[2] // f, f).mean(axis=(1, 3, 5))


def upsample(a, f, shape):
    """把块平均网格上的图插值回原网格（像素边缘对齐），再裁到原尺寸。"""
    up = a if f == 1 else ndi.zoom(a, f, order=1, mode="nearest", grid_mode=True)
    pad = [(0, max(0, n - m)) for n, m in zip(shape, up.shape)]
    if any(p[1] for p in pad):
        up = np.pad(up, pad, mode="edge")
    return up[: shape[0], : shape[1], : shape[2]]


def masked_smooth(x, m, sigma_vox):
    return ndi.gaussian_filter(x * m, sigma_vox) / np.maximum(ndi.gaussian_filter(m, sigma_vox), 1e-6)


def scale_maps(img, liver, hp2, spacing, sigma_mm):
    """返回某一尺度下的 (μ_c, t, v, 有效掩膜)，均在原网格上。"""
    sc_n = [sigma_mm / sp for sp in spacing]
    ss_n = [SURROUND_RATIO * s for s in sc_n]
    f = int(2 ** max(0, int(np.floor(np.log2(min(sc_n))))))
    M = liver.astype(np.float32)
    stacks = [block_mean(a, f) for a in (M, img * M, img * img * M, hp2 * M)]
    sc = [s / f for s in sc_n]
    ss = [s / f for s in ss_n]
    k = float(np.prod([a / b for a, b in zip(sc_n, ss_n)]))
    gc = [ndi.gaussian_filter(a, sc, mode="constant", truncate=3.0) for a in stacks]
    gs = [ndi.gaussian_filter(a, ss, mode="constant", truncate=3.0) for a in stacks]
    ga = [(s - k * c) / (1 - k) for s, c in zip(gs, gc)]
    wc, wa = np.maximum(gc[0], 1e-6), np.maximum(ga[0], 1e-6)
    mu_c, mu_a = gc[1] / wc, ga[1] / wa
    var_c = np.maximum(gc[2] / wc - mu_c ** 2, 1e-3)
    var_a = np.maximum(ga[2] / wa - mu_a ** 2, 1e-3)
    n_c = gauss_neff(sc_n) * np.clip(gc[0], 0, 1)
    n_a = annulus_neff(sc_n, ss_n) * np.clip(ga[0], 0, 1)
    t = (mu_c - mu_a) / np.sqrt(var_c / np.maximum(n_c, 1) + var_a / np.maximum(n_a, 1))
    v = 0.5 * np.log(np.maximum(gc[3] / wc, 1e-6) / np.maximum(ga[3] / wa, 1e-6))
    out = [upsample(a.astype(np.float32), f, img.shape) for a in (mu_c, t, v)]
    # 有效区域在原网格上判定（先插值肝内占比再取阈值），避免粗尺度的块状边界
    frac_c, frac_a = (upsample(a.astype(np.float32), f, img.shape) for a in (gc[0], ga[0]))
    out.append((frac_c >= MIN_LIVER_FRACTION) & (frac_a >= MIN_LIVER_FRACTION))
    return out


def robust_z(a, bg):
    vals = a[bg & np.isfinite(a)]
    if vals.size < 100:
        return np.full_like(a, np.nan)
    med = np.median(vals)
    rsd = 1.4826 * np.median(np.abs(vals - med)) + 1e-6
    return (a - med) / rsd


def feature_maps(img, liver, background, spacing):
    """返回 {I, Ims, D, T, DT} 原网格特征图（无效位置为 NaN）。"""
    M = liver.astype(np.float32)
    fine = masked_smooth(img, M, 1.0)
    hp = img - masked_smooth(img, M, [HIGHPASS_MM / sp for sp in spacing])
    feats = {"I": np.abs(robust_z(fine, background)).astype(np.float32)}
    acc = {k: np.zeros(img.shape, np.float32) for k in ("Ims", "D", "T")}
    any_valid = np.zeros(img.shape, bool)
    for s in SCALES_MM:
        mu, t, v, valid = scale_maps(img, liver, hp * hp, spacing, s)
        for src, key in ((mu, "Ims"), (t, "D"), (v, "T")):
            z = np.abs(robust_z(np.where(valid, src, np.nan), background & valid))
            np.fmax(acc[key], np.nan_to_num(z, nan=0.0), out=acc[key])
        any_valid |= valid
    for key, a in acc.items():
        a[~any_valid] = np.nan
        feats[key] = a
    feats["DT"] = np.fmax(feats["D"], feats["T"])
    for a in feats.values():
        a[~liver] = np.nan
    return feats


def auc(pos, neg, rng):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size < 5 or neg.size < 5:
        return float("nan")
    if pos.size > MAX_SAMPLES:
        pos = rng.choice(pos, MAX_SAMPLES, replace=False)
    if neg.size > MAX_SAMPLES:
        neg = rng.choice(neg, MAX_SAMPLES, replace=False)
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size)
    ranks[order] = np.arange(1, allv.size + 1)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]          # 并列取平均秩
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


# ----------------------------------------------------------------------------
# 单病例
# ----------------------------------------------------------------------------
def lesion_cnr(img, liver_only, lesion, others):
    """与 s42 完全相同的 CNR（体素单位的 2–6 环）。"""
    idx = np.argwhere(lesion)
    lo = np.maximum(idx.min(0) - 10, 0)
    hi = np.minimum(idx.max(0) + 11, np.array(lesion.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    m, liv, oth = lesion[sl], liver_only[sl], others[sl]
    d_out = ndi.distance_transform_edt(~m)
    peri = (~m) & liv & (d_out >= CNR_RING[0]) & (d_out <= CNR_RING[1]) & ~oth
    if peri.sum() < 20:
        peri = (~m) & liv & (d_out >= 1) & (d_out <= CNR_RING[1]) & ~oth
    if peri.sum() < 10:
        return None
    im = img[sl]
    return abs(float(im[m].mean()) - float(im[peri].mean())) / (float(im[peri].std()) + 1e-8)


def process_volume(case, img, seg, spacing, rng):
    zyx = np.argwhere(seg > 0)
    lo = np.maximum(zyx.min(0) - 8, 0)
    hi = np.minimum(zyx.max(0) + 9, np.array(seg.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    img, seg = img[sl].astype(np.float32), seg[sl]
    liver, liver_only, tumor = seg > 0, seg == 1, seg == 2
    lab, n = ndi.label(tumor)
    dist_any = ndi.distance_transform_edt(~tumor, sampling=spacing)
    background = liver_only & (dist_any > BACKGROUND_GAP_MM)
    if background.sum() < 1000:
        background = liver_only & (dist_any > LOCAL_RING_MM[1] / 2)
    feats = feature_maps(img, liver, background, spacing)
    bg_vals = {k: v[background] for k, v in feats.items()}
    thr = {k: float(np.nanquantile(v, 0.99)) for k, v in bg_vals.items()}
    voxel_mm3 = float(np.prod(spacing))
    recs = []
    for i in range(1, n + 1):
        les = lab == i
        size = int(les.sum())
        if size < MIN_LESION:
            continue
        others = tumor & ~les
        cnr = lesion_cnr(img, liver_only, les, others)
        idx = np.argwhere(les)
        blo = np.maximum(idx.min(0) - 25, 0)
        bhi = np.minimum(idx.max(0) + 26, np.array(les.shape))
        bsl = tuple(slice(int(a), int(b)) for a, b in zip(blo, bhi))
        d_this = ndi.distance_transform_edt(~les[bsl], sampling=spacing)
        d_oth = (ndi.distance_transform_edt(~others[bsl], sampling=spacing) if others[bsl].any()
                 else np.full(d_this.shape, 1e9))
        ring = (liver_only[bsl] & (d_this >= LOCAL_RING_MM[0]) & (d_this <= LOCAL_RING_MM[1])
                & (d_oth > OTHER_LESION_GAP_MM))
        cz, cy, cx = (idx.mean(0) + lo).round(1)
        d = float((6 * size * voxel_mm3 / np.pi) ** (1 / 3))
        rec = dict(case=case, voxels=size, diameter_mm=d, cz=float(cz), cy=float(cy), cx=float(cx),
                   cnr=cnr, ring_voxels=int(ring.sum()), background_voxels=int(background.sum()))
        rec["cnr_stratum"] = (None if cnr is None else
                              "low" if cnr < CNR_TERTILES[0] else "mid" if cnr <= CNR_TERTILES[1] else "high")
        rec["size_stratum"] = "<10" if d < 10 else ("10-20" if d <= 20 else ">20")
        for k, fmap in feats.items():
            fl = fmap[bsl][les[bsl]]
            rec[f"auc_bg_{k}"] = auc(fl, bg_vals[k], rng)
            rec[f"auc_local_{k}"] = auc(fl, fmap[bsl][ring], rng)
            med = float(np.nanmedian(fl)) if np.isfinite(fl).any() else float("nan")
            rec[f"med_{k}"] = med
            rec[f"hit_{k}"] = bool(np.isfinite(med) and med > thr[k])
        recs.append(rec)
    return recs, feats, (img, seg)


# ----------------------------------------------------------------------------
# 汇总与判据（判据见 docs/方法重设计_可检出性计划.md §4，事先登记）
# ----------------------------------------------------------------------------
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok]))
    rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize(recs, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = dict(n_lesions=len(recs), n_cases=len({r["case"] for r in recs}), strata={})
    for key in ("cnr_stratum", "size_stratum"):
        for s in sorted({r[key] for r in recs if r[key] is not None}):
            sub = [r for r in recs if r[key] == s]
            row = dict(n=len(sub))
            for f in FEATURES:
                row[f"auc_bg_{f}"] = float(np.nanmedian([r[f"auc_bg_{f}"] for r in sub]))
                row[f"auc_local_{f}"] = float(np.nanmedian([r[f"auc_local_{f}"] for r in sub]))
                row[f"hit_{f}"] = float(np.mean([r[f"hit_{f}"] for r in sub]))
            out["strata"][f"{key}={s}"] = row
    low = [r for r in recs if r["cnr_stratum"] == "low"]
    cases = sorted({r["case"] for r in low})
    by_case = {c: [r for r in low if r["case"] == c] for c in cases}

    def gain(rs):
        return [r["auc_bg_D"] - r["auc_bg_I"] for r in rs]

    def p2_stat(rs):
        return spearman([r["diameter_mm"] for r in rs], gain(rs))

    boots_p2, boots_gain = [], []
    for _ in range(n_boot):
        pick = rng.choice(cases, len(cases), replace=True)
        rs = [r for c in pick for r in by_case[c]]
        boots_p2.append(p2_stat(rs))
        boots_gain.append(np.nanmedian(gain(rs)))
    bp2 = np.array([b for b in boots_p2 if np.isfinite(b)])
    bg = np.array([b for b in boots_gain if np.isfinite(b)])
    lowrow = out["strata"].get("cnr_stratum=low", {})
    gain_ci = [float(np.quantile(bg, 0.025)), float(np.quantile(bg, 0.975))] if bg.size else [float("nan")] * 2
    p1 = bool(lowrow and lowrow["auc_bg_D"] >= lowrow["auc_bg_I"] + 0.05 and gain_ci[0] > 0
              and lowrow["hit_D"] > lowrow["hit_I"])
    p2_ci = [float(np.quantile(bp2, 0.025)), float(np.quantile(bp2, 0.975))] if bp2.size else [float("nan")] * 2
    p2 = bool(np.isfinite(p2_ci[0]) and p2_ci[0] > 0)
    p3 = bool(lowrow and lowrow["auc_bg_T"] >= 0.60)
    p4 = bool(lowrow and lowrow["auc_bg_D"] >= lowrow["auc_bg_Ims"] + 0.02)
    out["predictions"] = dict(
        P1_low_cnr_D_beats_I=p1, P1_median_gain=float(np.nanmedian(gain(low))) if low else float("nan"),
        P1_gain_ci95=gain_ci,
        P2_gain_grows_with_size=p2, P2_spearman=p2_stat(low), P2_ci95=p2_ci,
        P3_texture_signal=p3,
        P4_local_normalisation_adds=p4,
        decision=("GO" if p1 else ("GO-texture-only" if p3 else "NO-GO")),
        rule="GO if P1; GO-texture-only if not P1 but P3; else NO-GO. P2 and P4 shape the narrative only.")
    return out


def print_summary(s):
    print(f"\n病灶 {s['n_lesions']} 个，来自 {s['n_cases']} 例")
    for metric in ("auc_bg", "hit", "auc_local"):
        print(f"\n[{metric}]（{'中位数' if metric != 'hit' else '比例'}）")
        print(f"{'stratum':>22} {'n':>4} | " + " ".join(f"{f:>7}" for f in FEATURES))
        for k, r in s["strata"].items():
            print(f"{k:>22} {r['n']:4d} | " + " ".join(f"{r[f'{metric}_{f}']:7.3f}" for f in FEATURES))
    p = s["predictions"]
    print(f"\nP1 低 CNR 层 D 胜过 I：{p['P1_low_cnr_D_beats_I']}（AUC_bg 增益中位数 {p['P1_median_gain']:+.3f}，"
          f"95% CI {p['P1_gain_ci95'][0]:+.3f}…{p['P1_gain_ci95'][1]:+.3f}）")
    print(f"P2 增益随病灶增大：{p['P2_gain_grows_with_size']}（Spearman {p['P2_spearman']:+.3f}，"
          f"95% CI {p['P2_ci95'][0]:+.3f}…{p['P2_ci95'][1]:+.3f}）")
    print(f"P3 纹理信号：{p['P3_texture_signal']}")
    print(f"P4 局部噪声归一化有额外贡献（D vs Ims）：{p['P4_local_normalisation_adds']}")
    print(f"判定：{p['decision']}")


def write_csv(recs, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"[write] {path}")


def plot_summary(recs, path):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    strata = ["low", "mid", "high"]
    cols = {"I": "#999999", "Ims": "#6A8EAE", "D": "#C8463D", "T": "#5FB3A9", "DT": "#6A4C9C"}
    w = 0.16
    for j, f in enumerate(FEATURES):
        x = np.arange(3) + (j - 2) * w
        ax[0].bar(x, [np.nanmedian([r[f"auc_bg_{f}"] for r in recs if r["cnr_stratum"] == s]) for s in strata],
                  w, color=cols[f], label=f)
        ax[1].bar(x, [np.mean([r[f"hit_{f}"] for r in recs if r["cnr_stratum"] == s]) for s in strata],
                  w, color=cols[f], label=f)
    for a, t in zip(ax[:2], ("median AUC: lesion vs remote liver", "hit rate: lesion median > remote-liver q99")):
        a.set_xticks(range(3)); a.set_xticklabels([f"CNR {s}" for s in strata]); a.set_title(t)
        a.legend(fontsize=8); a.grid(alpha=.25, axis="y")
    ax[0].axhline(0.5, color="k", lw=0.6, ls="--")
    low = [r for r in recs if r["cnr_stratum"] == "low"]
    xs = [r["diameter_mm"] for r in low]
    ys = [r["auc_bg_D"] - r["auc_bg_I"] for r in low]
    ax[2].scatter(xs, ys, s=14, color="#C8463D")
    ax[2].axhline(0, color="k", lw=0.6, ls="--"); ax[2].set_xscale("log")
    ax[2].set_xlabel("lesion diameter (mm)"); ax[2].set_ylabel("AUC_bg(D) − AUC_bg(I)")
    ax[2].set_title(f"low-CNR lesions: gain vs size (Spearman {spearman(xs, ys):+.2f})"); ax[2].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"[fig] {path}")


def montage(case, img, seg, feats, path):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    lab, n = ndi.label(seg == 2)
    if n == 0:
        print(f"{case}: 无病灶")
        return
    big = lab == (int(np.argmax(ndi.sum(seg == 2, lab, range(1, n + 1)))) + 1)
    z = int(np.round(np.argwhere(big)[:, 0].mean()))
    panels = [("CT (W400/L50)", img[z], dict(cmap="gray", vmin=-150, vmax=250))]
    for k in FEATURES:
        panels.append((f"{k} (|z|)", feats[k][z], dict(cmap="magma", vmin=0, vmax=np.nanquantile(feats[k][z], 0.995))))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for a, (t, im, kw) in zip(ax, panels):
        a.imshow(im, **kw)
        a.contour(seg[z] == 2, [0.5], colors="cyan", linewidths=0.6)
        a.contour(seg[z] > 0, [0.5], colors="lime", linewidths=0.4)
        a.set_title(t, fontsize=9); a.axis("off")
    fig.suptitle(f"{case} z={z} (largest lesion; cyan = tumor GT, green = liver GT)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[fig] {path}")


# ----------------------------------------------------------------------------
# 合成自检（检验实现，不检验假设）
# ----------------------------------------------------------------------------
SPECS = [  # (半径 mm, ΔHU, 噪声倍数, 中心偏移 mm)
    (4, -2.5, 1, (-25, -40, -40)), (10, -2.5, 1, (-25, 30, -35)), (22, -2.5, 1, (15, 0, -15)),
    (10, -20, 1, (25, -40, 40)), (10, 0, 2.0, (25, 40, 30))]


def synth(rng, hetero):
    spacing = (1.0, 0.78, 0.78)
    shape = (120, 220, 220)
    zz, yy, xx = np.meshgrid(*[np.arange(n) * sp for n, sp in zip(shape, spacing)], indexing="ij")
    c = [n * sp / 2 for n, sp in zip(shape, spacing)]
    liver = ((zz - c[0]) / 55) ** 2 + ((yy - c[1]) / 80) ** 2 + ((xx - c[2]) / 80) ** 2 < 1
    white = 12.0
    img = 100 + white * rng.standard_normal(shape)
    if hetero:
        field = ndi.gaussian_filter(rng.standard_normal(shape), 6)
        img += hetero * field / field.std()
    img = np.where(liver, img, 30 * rng.standard_normal(shape)).astype(np.float32)
    seg = liver.astype(np.int16)
    centers = []
    for r, dmu, nf, off in SPECS:
        m = ((zz - c[0] - off[0]) ** 2 + (yy - c[1] - off[1]) ** 2 + (xx - c[2] - off[2]) ** 2) < r * r
        img[m] += dmu + np.sqrt(max(nf * nf - 1, 0)) * white * rng.standard_normal(int(m.sum()))
        seg[m] = 2
        centers.append(np.argwhere(m).mean(0))
    return img, seg, spacing, centers


def run_synth(rng, hetero):
    img, seg, spacing, centers = synth(rng, hetero)
    t0 = time.time()
    recs, feats, crop = process_volume("synthetic", img, seg, spacing, rng)
    got = {}
    for r in recs:
        got[int(np.argmin([np.linalg.norm(np.array([r["cz"], r["cy"], r["cx"]]) - ce) for ce in centers]))] = r
    print(f"\n合成体（低频不均匀性 {hetero} HU）：{time.time() - t0:.1f}s，病灶 {len(recs)}/{len(SPECS)}")
    print(f"{'r(mm)':>5} {'ΔHU':>5} {'噪声×':>5} {'CNR':>6} | " + " ".join(f"{'bg_'+f:>7}" for f in FEATURES)
          + " | " + " ".join(f"{'loc_'+f:>7}" for f in FEATURES))
    for i, s in enumerate(SPECS):
        r = got[i]
        print(f"{s[0]:5.0f} {s[1]:5.1f} {s[2]:5.1f} {r['cnr'] or float('nan'):6.2f} | "
              + " ".join(f"{r[f'auc_bg_{f}']:7.3f}" for f in FEATURES) + " | "
              + " ".join(f"{r[f'auc_local_{f}']:7.3f}" for f in FEATURES))
    return got, feats, crop, spacing


def selftest():
    """E1 白噪声下面积积分：ΔHU = −2.5（CNR ≈ 0.2）的病灶，AUC_bg(D) 随半径上升，22 mm 时 > AUC_bg(I) + 0.1。
    E2 纹理：均值不变、噪声加倍的病灶，AUC_bg(T) > 0.75 且高于 AUC_bg(D)。
    E3 仿射不变：图像变为 2x + 50 后，D 与 T 基本不变。
    另跑一个带 3 HU 低频不均匀性的场景，只打印、不判定，用来说明真实数据里要回答的问题。"""
    rng = np.random.default_rng(0)
    got, feats, (im_c, sg_c), spacing = run_synth(rng, hetero=0.0)
    e1 = (got[2]["auc_bg_D"] > got[2]["auc_bg_I"] + 0.1
          and got[2]["auc_bg_D"] > got[1]["auc_bg_D"] > got[0]["auc_bg_D"])
    e2 = got[4]["auc_bg_T"] > 0.75 and got[4]["auc_bg_T"] > got[4]["auc_bg_D"]
    liver_c = sg_c > 0
    bg_c = (sg_c == 1) & (ndi.distance_transform_edt(sg_c != 2, sampling=spacing) > BACKGROUND_GAP_MM)
    f1 = feature_maps(im_c, liver_c, bg_c, spacing)
    f2 = feature_maps(2 * im_c + 50, liver_c, bg_c, spacing)
    v = np.isfinite(f1["D"]) & np.isfinite(f2["D"])
    dd = float(np.max(np.abs(f2["D"][v] - f1["D"][v])))
    dt = float(np.max(np.abs(f2["T"][v] - f1["T"][v])))
    e3 = dd < 0.05 and dt < 0.05
    print(("\n✓" if e1 else "\n✗") + " E1 面积积分：淡病灶的 AUC_bg(D) 随半径上升，22 mm 时明显高于 I")
    print(("✓" if e2 else "✗") + " E2 纹理：等均值、噪声加倍的病灶由 T 检出")
    print(("✓" if e3 else "✗") + f" E3 仿射不变：max|ΔD| = {dd:.2e}，max|ΔT| = {dt:.2e}")
    run_synth(np.random.default_rng(1), hetero=3.0)
    ok = e1 and e2 and e3
    print("\n自检通过" if ok else "\n自检失败")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cohort", choices=["train"], default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--max-cases", type=int, default=None)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import FIGS, OUT, banner, load_raw_resampled, preprocessed_spacing, save_json, splits  # noqa: E402

    sp = preprocessed_spacing()                      # (z, y, x) mm
    tr, va, te = splits()
    rng = np.random.default_rng(0)
    if args.cases:
        locked = set(te) & set(args.cases)
        if locked:
            raise SystemExit(f"测试集病例保持锁定，拒绝运行：{sorted(locked)}")
        banner(f"s46 目视检查：{args.cases}（不进入判据）")
        for c in args.cases:
            img, seg = load_raw_resampled(c)
            recs, feats, (im, sg) = process_volume(c, img, seg, sp, rng)
            for r in recs:
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
            if args.montage:
                montage(c, im, sg, feats, FIGS / f"fig46_montage_{c}.png")
        return

    if args.cohort != "train":
        ap.error("需要 --selftest、--cohort train 或 --cases")
    banner("s46 可检出性探针 — fold 0 训练集（验证集与测试集不参与）")
    cases = tr[: args.max_cases] if args.max_cases else tr
    recs = []
    t0 = time.time()
    for i, c in enumerate(cases):
        img, seg = load_raw_resampled(c)
        if not (seg == 2).any():
            continue
        r, _, _ = process_volume(c, img, seg, sp, rng)
        recs += r
        print(f"  {i + 1}/{len(cases)} {c}: {len(r)} 病灶  累计 {len(recs)}  {time.time() - t0:.0f}s", flush=True)
    s = summarize(recs)
    print_summary(s)
    s["protocol"] = dict(cohort="fold0 train", n_cases=len(cases), scales_mm=SCALES_MM,
                         surround_ratio=SURROUND_RATIO, highpass_mm=HIGHPASS_MM, cnr_tertiles=CNR_TERTILES,
                         local_ring_mm=LOCAL_RING_MM, background_gap_mm=BACKGROUND_GAP_MM,
                         min_lesion_voxels=MIN_LESION, spacing_zyx_mm=sp)
    save_json(s, "s46_detectability_probe.json")
    write_csv(recs, OUT / "s46_lesions.csv")
    plot_summary(recs, FIGS / "fig46_detectability.png")
    banner("s46 完成")


if __name__ == "__main__":
    main()
