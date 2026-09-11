#!/usr/bin/env python3
"""
Step 2 / 4.2 — 真实病例 PC 响应（检验 C1）

命题 C1：相位一致性在低对比度下比梯度幅值更稳。
证伪：低 CNR 区间 PC 的边界/内部（或边界/外周肝实质）响应比不高于 |∇f|。

协议
----
* 队列：**fold 0 验证集 21 例**；测试集 27 例保持锁定（validation_plan §2.1）。
* 输入：**原始 CT 重采样到计划间距 1.0/0.779/0.779 mm**，保持 HU 量纲。
  原因：preprocessed 下的 data_cf 已重采样，但 gt_segmentations 未重采样，二者网格不一致；
  nnU-Net 的 CTNormalization 是全局仿射变换，故重采样后的 HU 上 CNR = ΔHU/σ_HU 可直接解释。
* GT：原始 labelsTr 同步最近邻重采样（同网格）。
* 病灶：label==2 连通域，≥50 体素。
* 区域：GT 边界壳层（内部距离 ≤1 体素）；病灶内部；外周肝实质环（体外 2–6 体素且 label==1）。
* 裁剪策略：CNR 用小裁剪；PC 用 bbox+边距的核心裁剪 → **reflect 补边 32 体素**
  再计算，最后取中心区域。滤波器空间支撑 ≈ 26 体素，故补边 32 体素可避免裁剪面
  阶跃进入测量区。核心边长上限 120 体素，超出则标记 truncated 并排除出 PC 统计。
* CNR 分层阈值 **从训练集 83 例病灶冻结**（三分位），再套用到验证集。

实现要点：所有形态学操作改用欧氏距离变换（edt）一次性完成，
取代迭代膨胀（184³ × 6 次迭代 × 每病灶，原文实现过慢）。
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import (  # noqa: E402
    FIGS, PC_MAX_VOXELS_FULL, banner, lesions_from_seg, load_raw_resampled,
    preprocessed_spacing, save_json, splits,
)
from nnunetv2.training.nnUNetTrainer.phaser.dim_utils import (  # noqa: E402
    spatial_gradient_magnitude,
)
from nnunetv2.training.nnUNetTrainer.phaser.modules import PhaseCongruency  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_LESION = 50
PERI_LO, PERI_HI = 2, 6
CORE_MAX = 120
PAD = 32
CNR_MARGIN = 10

_pc = PhaseCongruency(dims=3, n_scales=3, max_voxels=PC_MAX_VOXELS_FULL).to(DEV).eval()


def pc_grad(vol: np.ndarray):
    with torch.no_grad():
        t = torch.from_numpy(vol)[None, None].to(DEV).float()
        pc = _pc(t)[0, 0].float().cpu().numpy()
        g = spatial_gradient_magnitude(t, 3)[0, 0].float().cpu().numpy()
    return pc.astype(np.float64), g.astype(np.float64)


def bbox_slices(mask, margin, shape):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, np.array(shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), lo, hi


def mean_of(a, sel):
    return float(a[sel].mean()) if sel.sum() > 0 else float("nan")


def std_of(a, sel):
    return float(a[sel].std()) if sel.sum() > 1 else float("nan")


def process_case(case, with_pc, tag, progress=""):
    """返回该病例的病灶记录列表。"""
    img, seg = load_raw_resampled(case)
    liver = seg == 1
    tumor_all = seg == 2
    recs = []
    if not with_pc:
        # 训练集只需要 CNR：用整体 bbox + 距离变换
        pass
    for _, mask, size in lesions_from_seg(seg, MIN_LESION):
        others = tumor_all & ~mask
        sl, lo, hi = bbox_slices(mask, CNR_MARGIN, mask.shape)
        m = mask[sl]
        liv = liver[sl]
        oth = others[sl]
        d_out = ndi.distance_transform_edt(~m)
        d_in = ndi.distance_transform_edt(m)
        peri = (~m) & liv & (d_out >= PERI_LO) & (d_out <= PERI_HI) & ~oth
        if peri.sum() < 20:
            peri = (~m) & liv & (d_out >= 1) & (d_out <= PERI_HI) & ~oth
        rec = dict(case=case, size_vox=int(size), cnr=None,
                   peri_vox=int(peri.sum()))
        if peri.sum() >= 10:
            im = img[sl]
            mi, mo = float(im[m].mean()), float(im[peri].mean())
            so = float(im[peri].std())
            rec["cnr"] = abs(mi - mo) / (so + 1e-8)
            rec["hu_in"] = mi
            rec["hu_peri"] = mo
            rec["hu_peri_std"] = so
        if not with_pc or rec["cnr"] is None:
            recs.append(rec)
            continue

        # ---------- 核心裁剪 + reflect 补边 ----------
        idx = np.argwhere(mask)
        center = idx.mean(0).round().astype(int)
        c_lo = np.maximum(center - CORE_MAX // 2, 0)
        c_hi = np.minimum(c_lo + CORE_MAX, np.array(mask.shape))
        c_lo = np.maximum(c_hi - CORE_MAX, 0)
        csl = tuple(slice(int(a), int(b)) for a, b in zip(c_lo, c_hi))
        fits = bool((idx.min(0) >= np.array([s.start for s in csl])).all()
                    and (idx.max(0) <= np.array([s.stop - 1 for s in csl])).all())
        if not fits:
            rec["truncated"] = True
            recs.append(rec)
            continue

        padded = np.pad(img[csl], PAD, mode="reflect")
        m_b = np.zeros(padded.shape, bool)
        m_l = np.zeros(padded.shape, bool)
        core_shape = img[csl].shape
        off = tuple(slice(PAD, PAD + core_shape[i]) for i in range(3))
        m_b[off] = mask[csl]
        m_l[off] = liver[csl]

        pc, gr = pc_grad(padded)

        d_o = ndi.distance_transform_edt(~m_b)
        d_i = ndi.distance_transform_edt(m_b)
        bnd = m_b & (d_i <= 1.0)
        inner = m_b & (d_i > 1.0)
        peri_p = (~m_b) & m_l & (d_o >= PERI_LO) & (d_o <= PERI_HI)
        if peri_p.sum() < 20:
            peri_p = (~m_b) & m_l & (d_o >= 1) & (d_o <= PERI_HI)

        rec.update(
            boundary_vox=int(bnd.sum()), interior_vox=int(inner.sum()),
            peri_pad_vox=int(peri_p.sum()),
            pc_b=mean_of(pc, bnd), pc_i=mean_of(pc, inner), pc_p=mean_of(pc, peri_p),
            g_b=mean_of(gr, bnd), g_i=mean_of(gr, inner), g_p=mean_of(gr, peri_p),
            pc_p_std=std_of(pc, peri_p), g_p_std=std_of(gr, peri_p),
        )
        rec["ratio_pc"] = rec["pc_b"] / max(rec["pc_p"], 1e-6)
        rec["ratio_g"] = rec["g_b"] / max(rec["g_p"], 1e-6)
        rec["sep_pc"] = (rec["pc_b"] - rec["pc_p"]) / (rec["pc_p_std"] + 1e-8)
        rec["sep_g"] = (rec["g_b"] - rec["g_p"]) / (rec["g_p_std"] + 1e-8)
        rec["ratio_i_pc"] = rec["pc_b"] / max(rec["pc_i"], 1e-6)
        rec["ratio_i_g"] = rec["g_b"] / max(rec["g_i"], 1e-6)

        # ---------- 距离剖面 ----------
        signed = np.where(m_b, -d_i, d_o)
        prof = []
        BINS = [("in0-1", (-2.0, 0.0), "in"), ("in1-2", (-3.0, -2.0), "in"),
                ("in2-4", (-5.0, -3.0), "in"), ("in4-8", (-9.0, -5.0), "in"),
                ("in8-16", (-17.0, -9.0), "in"), ("in16+", (-1e9, -17.0), "in"),
                ("out1-2", (0.0, 2.0), "out"), ("out2-4", (2.0, 4.0), "out"),
                ("out4-8", (4.0, 8.0), "out"), ("out8-16", (8.0, 16.0), "out"),
                ("out16+", (16.0, 1e9), "out")]
        for name, (a, b), side in BINS:
            sel = (signed >= a) & (signed < b)
            sel = sel & m_b if side == "in" else sel & (~m_b) & m_l
            prof.append(dict(bin=name, n=int(sel.sum()),
                             pc=mean_of(pc, sel), grad=mean_of(gr, sel)))
        rec["profile"] = prof
        recs.append(rec)
    del img, seg, liver, tumor_all
    return recs


def main():
    banner("Step 2 / 4.2 真实病例 PC 响应 — 边界 vs 外周肝实质（检验 C1）")
    tr, va, te = splits()
    sp = preprocessed_spacing()
    print(f"体素间距 (z,y,x) = {sp} mm")
    print(f"训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)}（测试集锁定，不参与）")
    print(f"device={DEV}  核心上限={CORE_MAX}  补边={PAD}  最小病灶={MIN_LESION} 体素\n")

    print("=== 训练集：CNR 分布（仅用于冻结分层阈值）===")
    tr_recs = []
    for i, c in enumerate(tr):
        tr_recs += process_case(c, with_pc=False, tag="train")
        if (i + 1) % 20 == 0:
            print(f"  train {i+1}/{len(tr)}")
    tr_cnr = np.array([r["cnr"] for r in tr_recs if r.get("cnr") is not None])
    q1, q2 = np.quantile(tr_cnr, [1 / 3, 2 / 3])
    print(f"\n训练集有效病灶 {len(tr_cnr)}（总 {len(tr_recs)}）")
    print(f"CNR 三分位阈值 = {q1:.4f} / {q2:.4f}")
    print(f"训练集 CNR: min={tr_cnr.min():.3f}  q1={np.quantile(tr_cnr,.25):.3f} "
          f"med={np.median(tr_cnr):.3f}  q3={np.quantile(tr_cnr,.75):.3f}  max={tr_cnr.max():.3f}")

    print("\n=== 验证集：PC 与 |∇f| 区域响应 ===")
    va_recs = []
    for i, c in enumerate(va):
        r = process_case(c, with_pc=True, tag="val")
        va_recs += r
        print(f"  val {i+1}/{len(va)} {c}: {len(r)} 病灶")

    ok = [r for r in va_recs if "pc_b" in r]
    trunc = [r for r in va_recs if r.get("truncated")]
    print(f"\n验证集病灶总数={len(va_recs)}  参与 PC/|∇f| 分析={len(ok)}  "
          f"裁剪截断跳过={len(trunc)}  无有效外周环={len(va_recs)-len(ok)-len(trunc)}")

    def stratum(c):
        return "low" if c < q1 else ("mid" if c < q2 else "high")

    for r in ok:
        r["cnr_stratum"] = stratum(r["cnr"])

    rows = []
    print(f"\n{'CNR层':>6} {'n':>4} | {'CNR中位':>8} {'PC_b':>7} {'PC_peri':>8} {'PC_ratio':>9} "
          f"{'g_b':>8} {'g_peri':>8} {'g_ratio':>9} | {'sep_PC':>8} {'sep_g':>8} {'sep比':>7}")
    for sname in ("low", "mid", "high"):
        sub = [r for r in ok if r["cnr_stratum"] == sname]
        if not sub:
            continue
        f = lambda k: float(np.nanmean([r[k] for r in sub]))       # noqa: E731
        fmd = lambda k: float(np.nanmedian([r[k] for r in sub]))   # noqa: E731
        row = dict(stratum=sname, n=len(sub), cnr_median=fmd("cnr"),
                   pc_b=f("pc_b"), pc_p=f("pc_p"), pc_i=f("pc_i"),
                   g_b=f("g_b"), g_p=f("g_p"), g_i=f("g_i"),
                   ratio_pc=fmd("ratio_pc"), ratio_g=fmd("ratio_g"),
                   ratio_i_pc=fmd("ratio_i_pc"), ratio_i_g=fmd("ratio_i_g"),
                   sep_pc=f("sep_pc"), sep_g=f("sep_g"),
                   sep_ratio=fmd("sep_pc") / max(fmd("sep_g"), 1e-9))
        rows.append(row)
        print(f"{sname:>6} {row['n']:4d} | {row['cnr_median']:8.3f} {row['pc_b']:7.4f} "
              f"{row['pc_p']:8.4f} {row['ratio_pc']:9.3f} {row['g_b']:8.3f} {row['g_p']:8.3f} "
              f"{row['ratio_g']:9.3f} | {row['sep_pc']:8.3f} {row['sep_g']:8.3f} {row['sep_ratio']:7.2f}")

    allsep_pc = np.array([r["sep_pc"] for r in ok])
    allsep_g = np.array([r["sep_g"] for r in ok])
    ratio_pc = np.array([r["ratio_pc"] for r in ok])
    ratio_g = np.array([r["ratio_g"] for r in ok])
    cnrs = np.array([r["cnr"] for r in ok])
    sizes = np.array([r["size_vox"] for r in ok])

    print(f"\n全部 {len(ok)} 个病灶：")
    print(f"  sep 中位      PC={np.median(allsep_pc):.3f}  |∇f|={np.median(allsep_g):.3f}  "
          f"→ PC 更优 {int((allsep_pc>allsep_g).sum())}/{len(ok)}")
    print(f"  boundary/peri ratio 中位  PC={np.median(ratio_pc):.3f}  |∇f|={np.median(ratio_g):.3f}")

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    print(f"  Spearman(CNR, sep_PC)={spearman(cnrs,allsep_pc):+.3f}  "
          f"Spearman(CNR, sep_g)={spearman(cnrs,allsep_g):+.3f}")
    print(f"  Spearman(CNR, ratio_PC)={spearman(cnrs,ratio_pc):+.3f}  "
          f"Spearman(CNR, ratio_g)={spearman(cnrs,ratio_g):+.3f}")
    print(f"  Spearman(size, sep_PC)={spearman(sizes,allsep_pc):+.3f}  "
          f"Spearman(size, sep_g)={spearman(sizes,allsep_g):+.3f}")

    prof_bins = [p["bin"] for p in ok[0]["profile"]]
    prof_pc = np.array([[p["pc"] for p in r["profile"]] for r in ok], dtype=np.float64)
    prof_g = np.array([[p["grad"] for p in r["profile"]] for r in ok], dtype=np.float64)
    pc_mean, g_mean = np.nanmean(prof_pc, 0), np.nanmean(prof_g, 0)
    print("\n距离剖面（负=病灶内；out* 仅统计肝实质）")
    print(f"{'bin':>8} {'n':>8} {'PC':>9} {'|∇f|':>11}")
    for i, b in enumerate(prof_bins):
        print(f"{b:>8} {np.mean([r['profile'][i]['n'] for r in ok]):8.0f} "
              f"{pc_mean[i]:9.4f} {g_mean[i]:11.4f}")

    print("\n" + "-" * 74)
    low = next((r for r in rows if r["stratum"] == "low"), None)
    high = next((r for r in rows if r["stratum"] == "high"), None)
    verdict = {}
    if low and high:
        verdict = dict(
            low_cnr_median=low["cnr_median"], high_cnr_median=high["cnr_median"],
            low_sep_ratio=low["sep_ratio"], high_sep_ratio=high["sep_ratio"],
            advantage_grows_as_cnr_drops=bool(low["sep_ratio"] > high["sep_ratio"]),
            low_ratio_pc=low["ratio_pc"], low_ratio_g=low["ratio_g"],
            low_ratio_advantage=bool(low["ratio_pc"] > low["ratio_g"]),
        )
        print(f"低 CNR 层（中位 {low['cnr_median']:.3f}）：PC/|∇f| 中位分离度比 = {low['sep_ratio']:.2f}×")
        print(f"高 CNR 层（中位 {high['cnr_median']:.3f}）：PC/|∇f| 中位分离度比 = {high['sep_ratio']:.2f}×")
        print(f"→ {'低 CNR 优势更大（支持 C1）' if verdict['advantage_grows_as_cnr_drops'] else '低 CNR 优势并未更大（不支持 C1）'}")
        print(f"→ 低 CNR 层 ratio：PC={low['ratio_pc']:.3f} vs |∇f|={low['ratio_g']:.3f}")

    # ---------------- 先落盘，避免绘图异常丢失结果 ----------------
    res_pre = dict(
        protocol=dict(cohort="fold0 validation (21 cases)", test_set_locked=True,
                      min_lesion_voxels=MIN_LESION, peri_ring_voxels=[PERI_LO, PERI_HI],
                      crop_core_max=CORE_MAX, reflect_pad=PAD,
                      spacing_zyx_mm=[sp[2], sp[1], sp[0]], pc_max_voxels=PC_MAX_VOXELS_FULL,
                      input="raw CT resampled to plan spacing (HU preserved; CTNormalization is global affine)",
                      cnr_definition="|mean(lesion)-mean(peri_liver_ring)| / std(peri_liver_ring)"),
        train_cnr_tertiles=[float(q1), float(q2)],
        train_cnr_stats=dict(n=int(tr_cnr.size), min=float(tr_cnr.min()),
                             median=float(np.median(tr_cnr)), max=float(tr_cnr.max())),
        val_lesions_total=len(va_recs), val_lesions_analysed=len(ok),
        val_lesions_truncated=len(trunc),
        stratum_table=rows,
        pooled=dict(n=len(ok),
                    median_sep_pc=float(np.median(allsep_pc)),
                    median_sep_g=float(np.median(allsep_g)),
                    median_ratio_pc=float(np.median(ratio_pc)),
                    median_ratio_g=float(np.median(ratio_g)),
                    n_pc_better_sep=int((allsep_pc > allsep_g).sum()),
                    spearman_cnr_sep_pc=spearman(cnrs, allsep_pc),
                    spearman_cnr_sep_g=spearman(cnrs, allsep_g),
                    spearman_cnr_ratio_pc=spearman(cnrs, ratio_pc),
                    spearman_cnr_ratio_g=spearman(cnrs, ratio_g),
                    spearman_size_sep_pc=spearman(sizes, allsep_pc),
                    spearman_size_sep_g=spearman(sizes, allsep_g)),
        profile_bins=prof_bins, profile_pc_mean=pc_mean.tolist(),
        profile_grad_mean=g_mean.tolist(),
        C1_verdict=verdict, per_lesion_ok=ok, per_lesion_train=tr_recs,
    )
    save_json(res_pre, "s42_real_pc.json")

    # ---------------- 作图 ----------------
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2))
    COL = {"pc": "#1D9E75", "g": "#D85A30"}
    NAME = {"pc": "phase congruency", "g": "|grad f|"}
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

    ax = axes[1]
    ax.scatter(cnrs, allsep_pc, s=16, color=COL["pc"], alpha=.8, label=NAME["pc"])
    ax.scatter(cnrs, allsep_g, s=16, color=COL["g"], alpha=.8, label=NAME["g"])
    ax.set_ylim(-3, 8)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("lesion CNR"); ax.set_ylabel("sep = (mu_b - mu_peri) / sigma_peri")
    ax.set_title(f"Boundary separability vs CNR\nSpearman PC={spearman(cnrs,allsep_pc):+.2f}, "
                 f"|grad|={spearman(cnrs,allsep_g):+.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[2]
    order = [r["stratum"] for r in rows]
    for i, sn in enumerate(order):
        sub = [r for r in ok if r["cnr_stratum"] == sn]
        def _fin(v):
            v = float(v)
            return v if np.isfinite(v) and v > 1e-6 else np.nan
        ax.boxplot([[_fin(r["ratio_pc"]) for r in sub], [_fin(r["ratio_g"]) for r in sub]],
                   positions=[i * 2 + 1, i * 2 + 2], widths=.7,
                   boxprops=dict(facecolor="none"), medianprops=dict(color="black"))
        rng = np.random.default_rng(0)
        for p, k in zip([i * 2 + 1, i * 2 + 2], ("pc", "g")):
            ax.scatter(np.full(len(sub), p) + rng.uniform(-.15, .15, len(sub)),
                       [max(_fin(r[f"ratio_{k}"]), 1e-6) for r in sub], s=9,
                       color=COL[k], alpha=.65)
    ax.set_xticks([i * 2 + 1.5 for i in range(len(order))]); ax.set_xticklabels(order)
    ax.set_yscale("log"); ax.set_ylabel("boundary / peri-liver ratio (log)")
    ax.set_title("Ratio by CNR tertile (left=PC, right=|grad f|)")
    ax.grid(alpha=.25, axis="y")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COL["pc"], label=NAME["pc"]),
                       Patch(facecolor=COL["g"], label=NAME["g"])], fontsize=8)

    ax = axes[3]
    ax.scatter(sizes, allsep_pc, s=16, color=COL["pc"], alpha=.8, label=NAME["pc"])
    ax.scatter(sizes, allsep_g, s=16, color=COL["g"], alpha=.8, label=NAME["g"])
    ax.set_ylim(-3, 8)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("lesion size (voxels, log)"); ax.set_ylabel("sep")
    ax.set_title(f"Separability vs lesion size\nSpearman PC={spearman(sizes,allsep_pc):+.2f}, "
                 f"|grad|={spearman(sizes,allsep_g):+.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle("4.2 Real cases (fold-0 validation, 21 cases): PC vs |grad f| at boundary / interior / peri-liver (C1)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fp = FIGS / "fig42_real_case_pc_response.png"
    fig.savefig(fp, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {fp}")

    res = dict(
        protocol=dict(cohort="fold0 validation (21 cases)", test_set_locked=True,
                      min_lesion_voxels=MIN_LESION, peri_ring_voxels=[PERI_LO, PERI_HI],
                      crop_core_max=CORE_MAX, reflect_pad=PAD,
                      spacing_zyx_mm=[sp[2], sp[1], sp[0]], pc_max_voxels=PC_MAX_VOXELS_FULL,
                      input="raw CT resampled to plan spacing (HU preserved; CTNormalization is global affine)",
                      cnr_definition="|mean(lesion)-mean(peri_liver_ring)| / std(peri_liver_ring)"),
        train_cnr_tertiles=[float(q1), float(q2)],
        train_cnr_stats=dict(n=int(tr_cnr.size), min=float(tr_cnr.min()),
                             median=float(np.median(tr_cnr)), max=float(tr_cnr.max())),
        val_lesions_total=len(va_recs), val_lesions_analysed=len(ok),
        val_lesions_truncated=len(trunc),
        stratum_table=rows,
        pooled=dict(n=len(ok),
                    median_sep_pc=float(np.median(allsep_pc)),
                    median_sep_g=float(np.median(allsep_g)),
                    median_ratio_pc=float(np.median(ratio_pc)),
                    median_ratio_g=float(np.median(ratio_g)),
                    n_pc_better_sep=int((allsep_pc > allsep_g).sum()),
                    spearman_cnr_sep_pc=spearman(cnrs, allsep_pc),
                    spearman_cnr_sep_g=spearman(cnrs, allsep_g),
                    spearman_cnr_ratio_pc=spearman(cnrs, ratio_pc),
                    spearman_cnr_ratio_g=spearman(cnrs, ratio_g),
                    spearman_size_sep_pc=spearman(sizes, allsep_pc),
                    spearman_size_sep_g=spearman(sizes, allsep_g)),
        profile_bins=prof_bins, profile_pc_mean=pc_mean.tolist(),
        profile_grad_mean=g_mean.tolist(),
        C1_verdict=verdict, per_lesion_ok=ok, per_lesion_train=tr_recs,
    )
    save_json(res_pre, "s42_real_pc.json")
    banner("4.2 完成")


if __name__ == "__main__":
    main()
