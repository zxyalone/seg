#!/usr/bin/env python3
"""
F0 特征探针：决定是否实现 WDS（小波细节保留下采样）与 SGF（频域全局滤波）。
计划与事先登记的判据：docs/频域模块计划.md §3（看到结果后不得修改）。

做法
----
每例按 nnU-Net 预处理到计划间距；病灶 = 标签 2 的 6 连通域（≥ 50 体素），CNR 与分层沿用
common.lesion_cnr_preprocessed（冻结三分位 1.2785 / 1.9208），大小分层用 evaluate.size_stratum。
远处肝实质 = 标签 1 且距任何肿瘤 > 15 mm（与 s46 相同）。

以每个病灶为中心取 128³ 块，过网络（AMP，与训练末验证相同），用 forward hook 取编码器 6 个 stage
的输出 e_0 … e_5。每个 stage 上把病灶掩膜与远处肝实质掩膜平均池化成占比图：
  正样本  病灶占比 ≥ 0.5 的体素；没有时取占比最大（> 0）的一个体素；最多 32 个
  负样本  同一块内远处肝实质占比 ≥ 0.9（stage 4、5 为 ≥ 0.5）的体素；最多 64 个；< 5 个记为无效
summarize 对每个 run × stage 训练 L2 逻辑回归（按训练折标准化，正负等权，λ = 1e-2），按病例
5 折（种子 0）；在留出折上，每个病灶的 AUC_s = 该病灶正样本得分 vs 同一块负样本得分。

图像侧对照（SGF 判据）：s46 的 Ims（多尺度肝内平滑强度，本例远处肝实质稳健标准化后取 max|·|），
用该病灶体素 vs 同一块内远处肝实质体素算 AUC_Ims（全分辨率）。预处理后的 CT 是 HU 的分段仿射
（裁剪到前景 0.5–99.5% 分位后 z-score），稳健标准化使其与 s46 在 HU 上的结果一致（裁剪区外除外）。

判据（计划 §3.3）
  F-W  drop = max(AUC_0, AUC_1) − AUC_3；W = median_<10mm(drop) − median_>20mm(drop)
       GO：W ≥ 0.10，病例 bootstrap 95% CI 下限 > 0，且 <10 mm 病灶 max(AUC_0, AUC_1) 中位数 ≥ 0.80
  F-S  低 CNR 且 >20 mm；gap = AUC_Ims − max(AUC_3, AUC_4)；S = median(gap)
       GO：S ≥ 0.05 且 CI 下限 > 0；AUC_deep 与 AUC_Ims 都有效的病灶 < 60% 时不判定（按 NO-GO）
  判据只用 A13b（3 个 seed 的病灶 AUC 先取平均），主队列 MSD08；其他队列只描述。

用法（服务器）
  python experiments/analysis/freq_feature_probe.py --selftest
  # 抽取（可分片；每片一个进程，可共用一张 GPU）
  python experiments/analysis/freq_feature_probe.py extract --cohort msd08 --shard 0/3 &
  ...
  python experiments/analysis/freq_feature_probe.py summarize --cohort msd08
  # 描述性队列：LiTS fold-0 验证集
  python experiments/analysis/freq_feature_probe.py extract --cohort lits_val
  python experiments/analysis/freq_feature_probe.py summarize --cohort lits_val

只读：不写 nnUNet_data 下任何文件；产出写入 results/freq_probe/<cohort>/。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "eval"))
# 追加在末尾：step2_mechanism/scripts 也有一个 common.py，不能遮住 experiments/analysis/common.py
sys.path.append(str(REPO / "step2_mechanism" / "scripts"))

PLAN = REPO / "docs" / "频域模块计划.md"
MIN_VOXELS = 50
BACKGROUND_GAP_MM = 15.0
POS_CAP, NEG_CAP, MIN_NEG = 32, 64, 5
POS_FRAC = 0.5
NEG_FRAC = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.5, 5: 0.5}
N_STAGES = 6
N_FOLDS = 5
LAMBDA = 1e-2
# 判据（计划 §3.3，事先登记）
W_SHALLOW, W_DEEP = (0, 1), 3
W_MIN_EFFECT = 0.10
W_MIN_SHALLOW_AUC = 0.80
# v2（计划 §3.5，v1 失效后登记）
OUT_DIR = "freq_probe_v2"
NEG_PATCHES = 8              # 每例负样本块数：远处肝实质中心
NEG_CENTER_SEP_MM = 30.0     # 负样本块中心两两最小间距
CTRL_POS_CAP = 32            # 位置对照：块中心体素最多取几个
W_MIN_ABS_DROP = 0.05        # 小病灶自身的 drop 中位数下限
W_MIN_STRATUM_N = 3          # 分层 W：每个 CNR 层小、大病灶各至少 3 个才计入
S_DEEP = (3, 4)
S_MIN_EFFECT = 0.05
S_MIN_VALID = 0.60
DECISION_GROUP = "A13b"
DECISION_COHORT = "msd08"
N_BOOT, BOOT_SEED = 2000, 0

COHORTS = {
    "msd08": dict(image_dir="/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08/imagesTs",
                  label_dir="/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08/labelsTs",
                  cases_file="/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08/cases.txt"),
    "lits_val": dict(image_dir="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/imagesTr",
                     label_dir="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/labelsTr",
                     cases_file=str(REPO / "experiments" / "eval" / "val_cases.txt")),
}


# ---------------------------------------------------------------------------
# 纯函数（单元测试覆盖）
# ---------------------------------------------------------------------------

def auc(pos, neg) -> float:
    """Mann–Whitney AUC，并列取平均秩；正样本 ≥ 1、负样本 ≥ MIN_NEG，否则 NaN。"""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size < 1 or neg.size < MIN_NEG:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size)
    ranks[order] = np.arange(1, allv.size + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def block_fraction(mask: np.ndarray, shape) -> np.ndarray:
    """掩膜（块网格）平均池化到 stage 网格；各维须整除。"""
    f = [a // b for a, b in zip(mask.shape, shape)]
    if any(a != b * k for a, b, k in zip(mask.shape, shape, f)):
        raise ValueError(f"{mask.shape} is not a multiple of {tuple(shape)}")
    m = mask.astype(np.float32)
    return m.reshape(shape[0], f[0], shape[1], f[1], shape[2], f[2]).mean(axis=(1, 3, 5))


def stage_samples(feat: np.ndarray, les_frac: np.ndarray, bg_frac: np.ndarray, stage: int, rng):
    """feat [C, *sp]；返回 (pos [n, C], neg [m, C]) float16，或 None（无效）。"""
    lf, bf = les_frac.ravel(), bg_frac.ravel()
    pos = np.flatnonzero(lf >= POS_FRAC)
    if pos.size == 0:
        j = int(np.argmax(lf))
        if lf[j] <= 0:
            return None
        pos = np.array([j])
    neg = np.flatnonzero((bf >= NEG_FRAC[stage]) & (lf == 0))
    if neg.size < MIN_NEG:
        return None
    if pos.size > POS_CAP:
        pos = np.sort(rng.choice(pos, POS_CAP, replace=False))
    if neg.size > NEG_CAP:
        neg = np.sort(rng.choice(neg, NEG_CAP, replace=False))
    flat = feat.reshape(feat.shape[0], -1)
    return flat[:, pos].T.astype(np.float16), flat[:, neg].T.astype(np.float16)


def pos_indices(les_frac: np.ndarray, rng):
    """正样本体素（stage 网格的扁平下标）：占比 ≥ 0.5；没有时取占比最大（> 0）的一个；最多 POS_CAP 个。"""
    lf = les_frac.ravel()
    pos = np.flatnonzero(lf >= POS_FRAC)
    if pos.size == 0:
        j = int(np.argmax(lf))
        if lf[j] <= 0:
            return None
        pos = np.array([j])
    if pos.size > POS_CAP:
        pos = np.sort(rng.choice(pos, POS_CAP, replace=False))
    return pos


def stage_samples_v2(feat: np.ndarray, les_frac: np.ndarray, neg_feats: list, neg_ok: list, rng):
    """v2 位置匹配采样。正样本同 v1；负样本取各负样本块（远处肝实质为中心）中与正样本**相同扁平下标**的体素，
    只保留 neg_ok 为真者（远处肝实质占比达标、无肿瘤），最多 NEG_CAP 个，< MIN_NEG 记为无效。
    正负样本在块内的位置分布因此相同，探针不能靠块内位置区分。"""
    pos = pos_indices(les_frac, rng)
    if pos is None:
        return None
    ks, ids = [], []
    for k, ok in enumerate(neg_ok):
        sel = pos[ok[pos]]
        ks.append(np.full(sel.size, k))
        ids.append(sel)
    ks, ids = np.concatenate(ks) if ks else np.zeros(0, int), np.concatenate(ids) if ids else np.zeros(0, int)
    if ids.size < MIN_NEG:
        return None
    if ids.size > NEG_CAP:
        pick = np.sort(rng.choice(ids.size, NEG_CAP, replace=False))
        ks, ids = ks[pick], ids[pick]
    c = feat.shape[0]
    neg = np.stack([neg_feats[k].reshape(c, -1)[:, i] for k, i in zip(ks, ids)])
    return feat.reshape(c, -1)[:, pos].T.astype(np.float16), neg.astype(np.float16)


def control_samples(feat: np.ndarray, ok: np.ndarray, rng):
    """位置对照：同一个远处肝实质块内，块中心（各维 |g − c| ≤ max(0.5, n/16)）对比四周
    （某一维 |g − c| ≥ n/4）的体素，只取 ok 者。两组都是远处肝实质，探针若能区分，靠的只是块内位置。"""
    sp = feat.shape[1:]
    g = np.indices(sp).reshape(len(sp), -1).astype(float)
    c = (np.array(sp, float)[:, None] - 1) / 2
    n = np.array(sp, float)[:, None]
    d = np.abs(g - c)
    center = (d <= np.maximum(0.5, n / 16)).all(0) & ok
    edge = (d >= n / 4).any(0) & ok
    pos, neg = np.flatnonzero(center), np.flatnonzero(edge)
    if pos.size < 1 or neg.size < MIN_NEG:
        return None
    if pos.size > CTRL_POS_CAP:
        pos = np.sort(rng.choice(pos, CTRL_POS_CAP, replace=False))
    if neg.size > NEG_CAP:
        neg = np.sort(rng.choice(neg, NEG_CAP, replace=False))
    flat = feat.reshape(feat.shape[0], -1)
    return flat[:, pos].T.astype(np.float16), flat[:, neg].T.astype(np.float16)


def pick_neg_centers(far: np.ndarray, spacing, k: int = NEG_PATCHES, min_sep_mm: float = NEG_CENTER_SEP_MM):
    """远处肝实质内离其边界最远的点，贪心取 k 个，两两相距 ≥ min_sep_mm（v2 负样本块的中心）。"""
    if not far.any():
        return []
    idx = np.argwhere(far)
    lo = np.maximum(idx.min(0) - 1, 0)
    hi = np.minimum(idx.max(0) + 2, np.array(far.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    d = ndi.distance_transform_edt(far[sl], sampling=spacing)
    r = [int(np.ceil(min_sep_mm / sp)) for sp in spacing]
    centers = []
    for _ in range(k):
        j = int(np.argmax(d))
        if d.flat[j] <= 0:
            break
        c = np.unravel_index(j, d.shape)
        centers.append(tuple(int(v + o) for v, o in zip(c, lo)))
        box = tuple(slice(max(ci - ri, 0), ci + ri + 1) for ci, ri in zip(c, r))
        grids = np.ogrid[tuple(slice(b.start, min(b.stop, n)) for b, n in zip(box, d.shape))]
        dist2 = sum(((gi - ci) * sp) ** 2 for gi, ci, sp in zip(grids, c, spacing))
        d[box][dist2 < min_sep_mm ** 2] = 0
    return centers


def centered_patch(arr: np.ndarray, center, size, fill):
    """arr [..., Z, Y, X] → 以 center 为中心、大小 size 的块，越界处填 fill（同 e13_mechanism）。"""
    sp = arr.shape[-3:]
    out = np.full(arr.shape[:-3] + tuple(size), fill, dtype=arr.dtype)
    src, dst = [], []
    for c, n, s in zip(center, sp, size):
        lo = int(c) - s // 2
        a, b = max(lo, 0), min(lo + s, n)
        src.append(slice(a, b))
        dst.append(slice(a - lo, b - lo))
    out[(..., *dst)] = arr[(..., *src)]
    return out


def far_liver(seg: np.ndarray, spacing) -> np.ndarray:
    """标签 1 且距任何肿瘤 > 15 mm；在肝+肿瘤包围盒（外扩 24 体素）内计算。"""
    fg = seg > 0
    out = np.zeros(seg.shape, bool)
    if not fg.any():
        return out
    idx = np.argwhere(fg)
    lo = np.maximum(idx.min(0) - 24, 0)
    hi = np.minimum(idx.max(0) + 25, np.array(seg.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    s = seg[sl]
    tumor = s == 2
    if tumor.any():
        d = ndi.distance_transform_edt(~tumor, sampling=spacing)
        out[sl] = (s == 1) & (d > BACKGROUND_GAP_MM)
    else:
        out[sl] = s == 1
    return out


def ims_map(img: np.ndarray, liver: np.ndarray, background: np.ndarray, spacing) -> np.ndarray:
    """s46 的 Ims（只算这一项，结果与 s46.feature_maps(...)['Ims'] 相同，自检核对）。"""
    import s46_detectability_probe as S
    M = liver.astype(np.float32)
    img = img.astype(np.float32)
    acc = np.zeros(img.shape, np.float32)
    any_valid = np.zeros(img.shape, bool)
    for s in S.SCALES_MM:
        sc_n = [s / sp for sp in spacing]
        ss_n = [S.SURROUND_RATIO * v for v in sc_n]
        f = int(2 ** max(0, int(np.floor(np.log2(min(sc_n))))))
        Mb, IMb = S.block_mean(M, f), S.block_mean(img * M, f)
        sc, ss = [v / f for v in sc_n], [v / f for v in ss_n]
        k = float(np.prod([a / b for a, b in zip(sc_n, ss_n)]))
        gc0 = S.gaussian_filter(Mb, sc, mode="constant", truncate=3.0)
        gc1 = S.gaussian_filter(IMb, sc, mode="constant", truncate=3.0)
        gs0 = S.gaussian_filter(Mb, ss, mode="constant", truncate=3.0)
        ga0 = (gs0 - k * gc0) / (1 - k)
        mu = S.upsample((gc1 / np.maximum(gc0, 1e-6)).astype(np.float32), f, img.shape)
        frac_c = S.upsample(gc0.astype(np.float32), f, img.shape)
        frac_a = S.upsample(ga0.astype(np.float32), f, img.shape)
        valid = (frac_c >= S.MIN_LIVER_FRACTION) & (frac_a >= S.MIN_LIVER_FRACTION)
        z = np.abs(S.robust_z(np.where(valid, mu, np.nan), background & valid))
        np.fmax(acc, np.nan_to_num(z, nan=0.0), out=acc)
        any_valid |= valid
    acc[~any_valid] = np.nan
    acc[~liver] = np.nan
    return acc


def ims_for_case(img: np.ndarray, seg: np.ndarray, spacing, far: np.ndarray) -> np.ndarray:
    """与 s46.process_volume 相同的裁剪（肝+肿瘤包围盒外扩 8 体素）后算 Ims，再放回整图（盒外 NaN）。"""
    idx = np.argwhere(seg > 0)
    lo = np.maximum(idx.min(0) - 8, 0)
    hi = np.minimum(idx.max(0) + 9, np.array(seg.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    s = seg[sl]
    bg = far[sl]
    if bg.sum() < 1000:  # s46 的退路：放宽到 7.5 mm
        tumor = s == 2
        d = ndi.distance_transform_edt(~tumor, sampling=spacing) if tumor.any() else np.full(s.shape, 1e9)
        bg = (s == 1) & (d > BACKGROUND_GAP_MM / 2)
    out = np.full(seg.shape, np.nan, np.float32)
    out[sl] = ims_map(img[sl], s > 0, bg, spacing)
    return out


# ---------------------------------------------------------------------------
# 探针训练与判据（纯函数）
# ---------------------------------------------------------------------------

def case_folds(cases, n_folds=N_FOLDS, seed=0) -> dict:
    u = np.array(sorted(set(cases)))
    perm = np.random.default_rng(seed).permutation(len(u))
    return {c: i for i, part in enumerate(np.array_split(u[perm], n_folds)) for c in part}


def fit_logreg(X: np.ndarray, y: np.ndarray, lam: float = LAMBDA, iters: int = 200):
    """L2 逻辑回归（正负类等权，偏置不正则），LBFGS。返回 (w, b)。"""
    import torch
    Xt = torch.from_numpy(np.asarray(X, np.float32))
    yt = torch.from_numpy(np.asarray(y, np.float32))
    n_pos, n_neg = float(yt.sum()), float((1 - yt).sum())
    sw = torch.where(yt > 0, 0.5 / max(n_pos, 1.0), 0.5 / max(n_neg, 1.0))
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")).sum() \
            + 0.5 * lam * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def probe_stage(records: list[dict], folds: dict) -> list[dict]:
    """records: [{case, lesion, pos [n, C], neg [m, C]}]（同一 run、同一 stage）。
    按病例分折训练、留出折打分；返回每个病灶的 AUC。"""
    out = []
    for k in sorted(set(folds.values())):
        train = [r for r in records if folds[r["case"]] != k]
        test = [r for r in records if folds[r["case"]] == k]
        if not train or not test:
            continue
        X = np.concatenate([np.concatenate([r["pos"], r["neg"]]) for r in train]).astype(np.float32)
        y = np.concatenate([np.r_[np.ones(len(r["pos"])), np.zeros(len(r["neg"]))] for r in train])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        w, b = fit_logreg((X - mu) / sd, y)
        for r in test:
            sp = ((r["pos"].astype(np.float32) - mu) / sd) @ w + b
            sn = ((r["neg"].astype(np.float32) - mu) / sd) @ w + b
            out.append(dict(case=r["case"], lesion=r["lesion"], auc=auc(sp, sn)))
    return out


def _median(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def boot_ci(df, stat, n_boot=N_BOOT, seed=BOOT_SEED):
    """按病例重采样（病例内病灶整体带入）；返回 (点估计, 下限, 上限)。"""
    point = stat(df)
    df = df.reset_index(drop=True)
    rows = {c: idx.to_numpy() for c, idx in df.groupby("case").groups.items()}
    cases = np.array(sorted(rows))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cases, len(cases), replace=True)
        vals.append(stat(df.iloc[np.concatenate([rows[c] for c in pick])]))
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size < n_boot * 0.9:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def w_stratified(x) -> float:
    """v2：在每个 CNR 层内比较小、大病灶的 drop 中位数，按该层小病灶数加权平均（消除两组 CNR 构成不同的影响）。"""
    num = den = 0.0
    for c in ("low", "mid", "high"):
        xs = x[(x.cnr_stratum == c) & (x.size_stratum == "<10")]
        xl = x[(x.cnr_stratum == c) & (x.size_stratum == ">20")]
        if len(xs) >= W_MIN_STRATUM_N and len(xl) >= W_MIN_STRATUM_N:
            num += len(xs) * (_median(xs["drop"]) - _median(xl["drop"]))
            den += len(xs)
    return num / den if den else float("nan")


def decide_w(les) -> dict:
    """les：每个病灶一行（判据组 seed 平均后的 auc_0 … auc_5、size_stratum、cnr_stratum）。
    v2 判据（计划 §3.5）：分层 W ≥ 0.10 且 CI 下限 > 0；小病灶浅层 AUC 中位数 ≥ 0.80；
    小病灶自身 drop 中位数 ≥ 0.05（下采样后确实丢了可分性）。"""
    d = les.copy()
    d["shallow"] = d[[f"auc_{s}" for s in W_SHALLOW]].max(axis=1, skipna=False)
    d["drop"] = d["shallow"] - d[f"auc_{W_DEEP}"]
    d = d[np.isfinite(d["drop"])]
    small, large = d[d.size_stratum == "<10"], d[d.size_stratum == ">20"]
    w, lo, hi = boot_ci(d[d.size_stratum.isin(["<10", ">20"])], w_stratified)
    shallow_small, drop_small = _median(small["shallow"]), _median(small["drop"])
    w_unstrat = drop_small - _median(large["drop"])
    go = bool(np.isfinite(lo) and w >= W_MIN_EFFECT and lo > 0 and shallow_small >= W_MIN_SHALLOW_AUC
              and drop_small >= W_MIN_ABS_DROP)
    per = {c: dict(n_small=int(((d.cnr_stratum == c) & (d.size_stratum == "<10")).sum()),
                   n_large=int(((d.cnr_stratum == c) & (d.size_stratum == ">20")).sum()),
                   drop_small=_median(d.loc[(d.cnr_stratum == c) & (d.size_stratum == "<10"), "drop"]),
                   drop_large=_median(d.loc[(d.cnr_stratum == c) & (d.size_stratum == ">20"), "drop"]))
           for c in ("low", "mid", "high")}
    return dict(W=w, ci=[lo, hi], W_unstratified_v1=w_unstrat, n_small=int(len(small)), n_large=int(len(large)),
                median_drop_small=drop_small, median_drop_large=_median(large["drop"]),
                median_shallow_auc_small=shallow_small, per_cnr=per, GO=go)


def decide_s(les) -> dict:
    d = les[(les.cnr_stratum == "low") & (les.size_stratum == ">20")].copy()
    n = len(d)
    d["deep"] = d[[f"auc_{s}" for s in S_DEEP]].max(axis=1, skipna=True)
    d["gap"] = d["ims_auc"] - d["deep"]
    ok = np.isfinite(d["gap"])
    frac = float(ok.mean()) if n else 0.0
    d = d[ok]
    if n == 0 or frac < S_MIN_VALID:
        return dict(S=float("nan"), ci=[float("nan")] * 2, n=n, valid_fraction=frac,
                    inconclusive=True, GO=False)
    s, lo, hi = boot_ci(d, lambda x: _median(x["gap"]))
    go = bool(np.isfinite(lo) and s >= S_MIN_EFFECT and lo > 0)
    return dict(S=s, ci=[lo, hi], n=n, valid_fraction=frac, inconclusive=False,
                median_ims_auc=_median(d["ims_auc"]), median_deep_auc=_median(d["deep"]), GO=go)


# ---------------------------------------------------------------------------
# 抽取（服务器，需要模型）
# ---------------------------------------------------------------------------

def encoder_features(net, x: np.ndarray, device):
    """一次前向：编码器各 stage 输出（列表，[C, *sp] float32）和块内 argmax。"""
    import torch
    cap = {}
    h = net.encoder.register_forward_hook(lambda m, i, o: cap.__setitem__("enc", o))
    try:
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            out = net(torch.from_numpy(x)[None].to(device).float())
    finally:
        h.remove()
    if isinstance(out, (list, tuple)):
        out = out[0]
    enc = [f[0].cpu().numpy() for f in cap["enc"]]   # AMP 下为 fp16，采样时统一存 fp16
    return enc, out[0].float().argmax(0).cpu().numpy()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_extract(a):
    import torch
    import common as C
    import evaluate as E

    out = Path(a.out or REPO / "results" / OUT_DIR / a.cohort)
    cfg = COHORTS[a.cohort]
    image_dir, label_dir = Path(a.image_dir or cfg["image_dir"]), Path(a.label_dir or cfg["label_dir"])
    device = torch.device(a.device)
    models = [C.load_model(g, s, device, a.results_root) for g, s in C.parse_runs(a.runs)]
    ref = models[0].predictor
    spacing = tuple(float(v) for v in ref.configuration_manager.spacing)
    patch = tuple(int(v) for v in ref.configuration_manager.patch_size)
    vox_mm3 = float(np.prod(spacing))
    cases = C.shard(C.read_cases(a.cases_file or cfg["cases_file"], image_dir), a.shard)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    if a.shard in (None, "0/1") or a.shard.startswith("0/"):
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout
        C.write_json(dict(cohort=a.cohort, runs=[m.name for m in models], git_head=head.strip(),
                          git_dirty=bool(dirty.strip()), plan_sha256=sha256(PLAN), script_sha256=sha256(__file__),
                          spacing=spacing, patch=patch, image_dir=str(image_dir), label_dir=str(label_dir),
                          probe_version=2, neg_patches=NEG_PATCHES, neg_center_sep_mm=NEG_CENTER_SEP_MM,
                          started=time.strftime("%F %T")), out / "run_manifest.json")
    for ci, case in enumerate(cases):
        if (out / "cases" / f"{case}.json").is_file():
            print(f"[{ci + 1}/{len(cases)}] {case}: done, skipped", flush=True)
            continue
        t0 = time.time()
        raw, props, seg_raw = C.read_raw(ref, case, image_dir, label_dir)
        data, seg, _ = C.preprocess(ref, raw, props, seg_raw)
        s0 = seg[0].astype(np.int8)
        lab, lesions = C.lesion_cnr_preprocessed(data, seg, min_voxels=MIN_VOXELS)
        far = far_liver(s0, spacing)
        for les in lesions:
            les["diameter_mm"] = float((6 * les["voxels"] * vox_mm3 / np.pi) ** (1 / 3))
            les["size_stratum"] = E.size_stratum(les["diameter_mm"])
        for les in lesions:
            les["need_ims"] = a.ims == "all" or (les["stratum"] == "low" and les["size_stratum"] == ">20")
        ims = ims_for_case(data[0], s0, spacing, far) if any(l["need_ims"] for l in lesions) else None
        rng = np.random.default_rng(int(hashlib.sha256(case.encode()).hexdigest()[:8], 16))
        arrays, meta = {}, []
        fill = float(data.min())
        # v2 负样本块：以远处肝实质最深处为中心（与病灶块同大小），每个模型各前向一次
        neg_centers = pick_neg_centers(far, spacing)
        tum = s0 == 2
        neg_x = [centered_patch(data, c, patch, fill) for c in neg_centers]
        neg_far = [centered_patch(far, c, patch, False) for c in neg_centers]
        neg_tum = [centered_patch(tum, c, patch, False) for c in neg_centers]
        neg_ok = {}   # stage -> [每个负样本块的扁平可用掩膜]
        recs = []
        for les in lesions:
            center = [int((lo + hi) // 2) for lo, hi in les["bbox"]]
            rec = dict(case=case, lesion=int(les["label"]), voxels=int(les["voxels"]), cnr=les["cnr"],
                       cnr_stratum=les["stratum"], diameter_mm=les["diameter_mm"],
                       size_stratum=les["size_stratum"], ims_auc=float("nan"), coverage={}, stages={})
            if ims is not None and les["need_ims"] and neg_centers:
                lp = centered_patch(lab, center, patch, 0) == les["label"]
                pv = centered_patch(ims, center, patch, np.nan)[lp]
                nv = np.concatenate([centered_patch(ims, c, patch, np.nan)[lp & nf]
                                     for c, nf in zip(neg_centers, neg_far)])
                pv, nv = pv[np.isfinite(pv)], nv[np.isfinite(nv)]
                if pv.size > 50000:
                    pv = rng.choice(pv, 50000, replace=False)
                if nv.size > 50000:
                    nv = rng.choice(nv, 50000, replace=False)
                rec["ims_auc"] = auc(pv, nv)
            recs.append((les, center, rec))
        control = {}
        for m in models:
            neg_enc = [encoder_features(m.network, x, device)[0] for x in neg_x]
            if neg_enc and not neg_ok:
                for s, f in enumerate(neg_enc[0][:N_STAGES]):
                    shp = f.shape[1:]
                    neg_ok[s] = [((block_fraction(nf, shp) >= NEG_FRAC[s]) & (block_fraction(nt, shp) == 0)).ravel()
                                 for nf, nt in zip(neg_far, neg_tum)]
            control[m.name] = []
            if neg_enc:
                for s, f in enumerate(neg_enc[0][:N_STAGES]):
                    smp = control_samples(f, neg_ok[s][0], rng)
                    if smp is not None:
                        arrays[f"{m.name}/ctrl/{s}/pos"], arrays[f"{m.name}/ctrl/{s}/neg"] = smp
                        control[m.name].append(s)
            for les, center, rec in recs:
                x = centered_patch(data, center, patch, fill)
                lp = centered_patch(lab, center, patch, 0) == les["label"]
                enc, pred = encoder_features(m.network, x, device)
                rec["coverage"][m.name] = float((pred[lp] == 2).mean())
                valid = []
                if neg_enc:
                    for s, f in enumerate(enc[:N_STAGES]):
                        smp = stage_samples_v2(f, block_fraction(lp, f.shape[1:]), [e[s] for e in neg_enc],
                                               neg_ok[s], rng)
                        if smp is None:
                            continue
                        key = f"{m.name}/{les['label']}/{s}"
                        arrays[key + "/pos"], arrays[key + "/neg"] = smp
                        valid.append(s)
                rec["stages"][m.name] = valid
            del neg_enc
        meta = [rec for _, _, rec in recs]
        np.savez_compressed(out / "cases" / f"{case}.npz", **arrays)
        C.write_json(dict(case=case, lesions=meta, neg_centers=[list(c) for c in neg_centers], control=control),
                     out / "cases" / f"{case}.json")
        print(f"[{ci + 1}/{len(cases)}] {case}: {len(lesions)} lesions, {len(neg_centers)} negative patches, "
              f"ims={'yes' if ims is not None else 'no'}, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def load(out: Path):
    import pandas as pd
    metas, feats, ctrl = [], {}, {}
    for js in sorted((out / "cases").glob("*.json")):
        meta = json.loads(js.read_text())
        npz = np.load(js.with_suffix(".npz"))
        for run, stages in meta.get("control", {}).items():
            for s in stages:
                ctrl.setdefault((run, s), []).append(dict(case=meta["case"], lesion=-1,
                                                          pos=npz[f"{run}/ctrl/{s}/pos"], neg=npz[f"{run}/ctrl/{s}/neg"]))
        for rec in meta["lesions"]:
            base = {k: rec[k] for k in ("case", "lesion", "voxels", "cnr", "cnr_stratum", "diameter_mm",
                                        "size_stratum", "ims_auc")}
            for run, cov in rec["coverage"].items():
                metas.append(dict(base, run=run, coverage=cov))
                for s in rec["stages"][run]:
                    key = f"{run}/{rec['lesion']}/{s}"
                    feats.setdefault((run, s), []).append(dict(case=rec["case"], lesion=rec["lesion"],
                                                               pos=npz[key + "/pos"], neg=npz[key + "/neg"]))
    return pd.DataFrame(metas), feats, ctrl


def lesion_table(meta, feats):
    """每个 run × 病灶一行，列 auc_0 … auc_5。"""
    import pandas as pd
    folds = case_folds(meta.case.unique())
    rows = []
    for (run, s), recs in sorted(feats.items()):
        for r in probe_stage(recs, folds):
            rows.append(dict(run=run, stage=s, **r))
    aucs = pd.DataFrame(rows)
    wide = aucs.pivot_table(index=["run", "case", "lesion"], columns="stage", values="auc").reset_index()
    wide.columns = [c if isinstance(c, str) else f"auc_{c}" for c in wide.columns]
    for s in range(N_STAGES):
        if f"auc_{s}" not in wide:
            wide[f"auc_{s}"] = np.nan
    t = meta.merge(wide, on=["run", "case", "lesion"], how="left")
    t["group"] = t.run.str.split("_s").str[0]
    return t


def group_mean(t, group):
    """同一病灶的 AUC 与覆盖率在该组各 seed 间取平均（任一 seed 无效则该 stage 为 NaN）。"""
    g = t[t.group == group]
    cols = [f"auc_{s}" for s in range(N_STAGES)] + ["coverage"]
    keys = ["case", "lesion", "voxels", "cnr", "cnr_stratum", "diameter_mm", "size_stratum", "ims_auc"]
    spec = {c: (c, lambda v: v.mean(skipna=False)) for c in cols}
    return g.groupby(keys, dropna=False).agg(n_seeds=("run", "size"), **spec).reset_index()


def md(df) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(rows)


def cmd_summarize(a):
    import pandas as pd
    import common as C

    out = Path(a.out or REPO / "results" / OUT_DIR / a.cohort)
    meta, feats, ctrl = load(out)
    t = lesion_table(meta, feats)
    t.to_csv(out / "lesion_auc_per_run.csv", index=False)
    groups = sorted(t.group.unique())
    per_group = {g: group_mean(t, g) for g in groups}
    for g, d in per_group.items():
        d.to_csv(out / f"lesion_auc_{g}.csv", index=False)

    res = dict(cohort=a.cohort, decision_cohort=a.cohort == DECISION_COHORT, groups=groups,
               n_cases=int(meta.case.nunique()), n_lesions=int(meta.drop_duplicates(["case", "lesion"]).shape[0]))
    if DECISION_GROUP in per_group:
        res["F_W"] = decide_w(per_group[DECISION_GROUP])
        res["F_S"] = decide_s(per_group[DECISION_GROUP])
        if not res["decision_cohort"]:
            res["F_W"]["GO"] = res["F_S"]["GO"] = None   # 描述性队列不判定

    # 描述性：各组 stage × 大小 × CNR 的 AUC 中位数
    desc = []
    for g, d in per_group.items():
        for (sz, cn), x in d.groupby(["size_stratum", "cnr_stratum"]):
            row = dict(group=g, size=sz, cnr=cn, n=len(x))
            row.update({f"auc_{s}": _median(x[f"auc_{s}"]) for s in range(N_STAGES)})
            row["ims_auc"] = _median(x["ims_auc"])
            row["coverage"] = _median(x["coverage"])
            desc.append(row)
    desc = pd.DataFrame(desc)
    desc.to_csv(out / "auc_by_stratum.csv", index=False)
    # 描述性：高 CNR 小病灶上 A13b 与 A0 的 drop
    if {"A13b", "A0"} <= set(per_group):
        def drop(d):
            x = d[(d.size_stratum == "<10") & (d.cnr_stratum == "high")]
            return _median(x[[f"auc_{s}" for s in W_SHALLOW]].max(axis=1, skipna=False) - x[f"auc_{W_DEEP}"])
        res["desc_high_cnr_small_drop"] = dict(A13b=drop(per_group["A13b"]), A0=drop(per_group["A0"]))
    # 描述性：大而淡病灶按块内覆盖率分组的 gap
    if DECISION_GROUP in per_group:
        d = per_group[DECISION_GROUP]
        d = d[(d.cnr_stratum == "low") & (d.size_stratum == ">20")]
        gap = d["ims_auc"] - d[[f"auc_{s}" for s in S_DEEP]].max(axis=1, skipna=True)
        res["desc_gap_by_coverage"] = {"coverage<0.5": _median(gap[d.coverage < 0.5]),
                                       "coverage>=0.5": _median(gap[d.coverage >= 0.5]),
                                       "n<0.5": int((d.coverage < 0.5).sum()), "n>=0.5": int((d.coverage >= 0.5).sum())}
    res["stage_valid_fraction"] = {g: {s: float(np.isfinite(d[f"auc_{s}"]).mean()) for s in range(N_STAGES)}
                                   for g, d in per_group.items()}
    # 位置对照：远处肝实质块的"中心 vs 四周"，各 run 的病例 AUC 中位数，再在组内取平均
    folds = case_folds(meta.case.unique())
    crow = []
    for (run, s), recs in sorted(ctrl.items()):
        crow.append(dict(group=run.split("_s")[0], run=run, stage=s,
                         auc=_median([r["auc"] for r in probe_stage(recs, folds)])))
    cdf = pd.DataFrame(crow)
    if not cdf.empty:
        cdf.to_csv(out / "position_control.csv", index=False)
        res["position_control"] = {g: {int(s): float(v) for s, v in x.groupby("stage")["auc"].mean().items()}
                                   for g, x in cdf.groupby("group")}
    C.write_json(res, out / "summary.json")

    lines = [f"# F0 特征探针 v2（位置匹配负样本）：{a.cohort}", "",
             f"{res['n_cases']} 例，{res['n_lesions']} 个病灶（≥ {MIN_VOXELS} 体素）。组：{', '.join(groups)}。"
             f"判据见 docs/频域模块计划.md §3.5；{'本队列为判据队列' if res['decision_cohort'] else '本队列只作描述'}。", ""]
    if "F_W" in res:
        w, s = res["F_W"], res["F_S"]
        lines += ["## 判据（A13b，3 个 seed 平均）", "",
                  f"- F-W：分层 W = {w['W']:.3f}，95% CI [{w['ci'][0]:.3f}, {w['ci'][1]:.3f}]"
                  f"（不分层 {w['W_unstratified_v1']:.3f}）；"
                  f"<10 mm drop 中位数 {w['median_drop_small']:.3f}（{w['n_small']} 个），"
                  f">20 mm {w['median_drop_large']:.3f}（{w['n_large']} 个）；"
                  f"<10 mm 浅层 AUC 中位数 {w['median_shallow_auc_small']:.3f} → **{w['GO']}**",
                  f"- F-S：S = {s['S']:.3f}，95% CI [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]；"
                  f"{s['n']} 个低 CNR > 20 mm 病灶，有效 {s['valid_fraction']:.0%} → **{s['GO']}**", ""]
    if "position_control" in res:
        pc = res["position_control"]
        lines += ["## 位置对照（远处肝实质块：中心 vs 四周，病例 AUC 中位数）", "",
                  md(pd.DataFrame([dict(group=g, **{f"stage_{s}": v for s, v in d.items()}) for g, d in pc.items()])),
                  "", "接近 0.5 说明该层特征不含块内位置信息；接近 1 说明 v1 式采样会被位置混杂。", ""]
    lines += ["## 各层 AUC 中位数（描述性）", "", md(desc), ""]
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines[:8]))


# ---------------------------------------------------------------------------
# 自检（合成数据，不需要模型或数据集）
# ---------------------------------------------------------------------------

def selftest():
    import pandas as pd
    rng = np.random.default_rng(0)
    # 1 AUC：与逐对比较一致（含并列）、单个正样本、负样本不足
    p, n = rng.integers(0, 5, 7).astype(float), rng.integers(0, 5, 9).astype(float)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n])
    assert abs(auc(p, n) - brute) < 1e-12
    assert auc([3.0], [1, 2, 3, 4, 5]) == 0.5 and np.isnan(auc([1.0], [0, 0]))
    # 2 池化与采样：小病灶在深层退化为单个体素；负样本排除病灶
    les = np.zeros((16, 16, 16), bool)
    les[5:7, 5:7, 5:7] = True
    bg = np.zeros_like(les)
    bg[8:, :, :] = True
    lf, bf = block_fraction(les, (2, 2, 2)), block_fraction(bg, (2, 2, 2))
    assert abs(lf.sum() * 512 - les.sum()) < 1e-6
    feat = rng.normal(size=(4, 2, 2, 2)).astype(np.float32)
    pos, neg = stage_samples(feat, lf, bf, 4, rng) or (None, None)
    assert pos is None  # 2³ 网格上负样本只有 4 个 < MIN_NEG
    feat = rng.normal(size=(4, 8, 8, 8)).astype(np.float32)
    lf8, bf8 = block_fraction(les, (8, 8, 8)), block_fraction(bg, (8, 8, 8))
    pos, neg = stage_samples(feat, lf8, bf8, 1, rng)
    assert pos.shape == (1, 4) and neg.shape == (NEG_CAP, 4)   # 病灶占比都 < 0.5 → 取最大的 1 个；负样本截断
    # 3 Ims 与 s46 一致
    import s46_detectability_probe as S
    shp, sp = (56, 72, 72), (1.0, 0.78, 0.78)
    seg = np.zeros(shp, np.int8)
    seg[2:54, 2:70, 2:70] = 1
    z, y, x = np.ogrid[:56, :72, :72]
    tum = (z - 28) ** 2 + ((y - 36) * 0.78) ** 2 + ((x - 36) * 0.78) ** 2 <= 36
    seg[tum] = 2
    img = (rng.normal(0, 10, shp) + 60 - 15 * tum).astype(np.float32)
    far = far_liver(seg, sp)
    assert far.any() and not (far & (seg == 2)).any()
    ref = S.feature_maps(img, seg > 0, far, sp)["Ims"]
    mine = ims_map(img, seg > 0, far, sp)
    ok = np.isfinite(ref)
    assert ok.sum() > 1000 and (ok == np.isfinite(mine)).all()
    assert np.abs(ref[ok] - mine[ok]).max() < 1e-4
    full = ims_for_case(img, seg, sp, far)       # 裁剪后放回整图，病灶内与直接计算一致
    assert np.isfinite(full[tum]).all() and auc(full[tum], full[far]) > 0.9
    # 4 探针：可分特征 AUC ≈ 1，纯噪声 AUC ≈ 0.5；按病例分折
    def recs(sep):
        out = []
        for c in range(20):
            for l in range(3):
                pos = rng.normal(sep, 1, (8, 6))
                out.append(dict(case=f"c{c}", lesion=l, pos=pos.astype(np.float16),
                                neg=rng.normal(0, 1, (30, 6)).astype(np.float16)))
        return out
    folds = case_folds([f"c{c}" for c in range(20)])
    assert len(set(folds.values())) == N_FOLDS
    sep = np.median([r["auc"] for r in probe_stage(recs(3.0), folds)])
    null = np.median([r["auc"] for r in probe_stage(recs(0.0), folds)])
    assert sep > 0.95 and abs(null - 0.5) < 0.1, (sep, null)
    # 5 判据：构造 GO 与 NO-GO
    def table(drop_small, gap):
        rows = []
        # 浅层 AUC 0.95；大病灶 AUC_3 = AUC_4 = 0.80（drop 0.15），小病灶 drop = drop_small；Ims = 0.80 + gap
        for c in range(30):
            for sz in ("<10", ">20"):
                a3 = 0.95 - drop_small if sz == "<10" else 0.80
                rows.append(dict(case=f"c{c}", lesion=len(rows), size_stratum=sz, cnr_stratum="low",
                                 auc_0=0.95, auc_1=0.93, auc_2=0.9, auc_3=a3 + rng.normal(0, 0.02),
                                 auc_4=0.80, auc_5=np.nan, ims_auc=0.80 + gap + rng.normal(0, 0.02)))
        return pd.DataFrame(rows)
    assert decide_w(table(0.30, 0.0))["GO"] and not decide_w(table(0.15, 0.0))["GO"]
    # v2：大病灶深层 AUC 上升造成的相对差不算 GO（小病灶自身 drop 须 ≥ 0.05）
    t = table(0.0, 0.0)
    big = t.size_stratum == ">20"
    t.loc[big, ["auc_0", "auc_1"]] = 0.85
    t.loc[big, "auc_3"] = 1.0
    r = decide_w(t)
    assert r["W"] >= W_MIN_EFFECT and not r["GO"]
    # v2：CNR 构成不同造成的差异被分层消除。同一 CNR 层内小、大病灶 drop 相同（高 CNR 0，低 CNR −0.3），
    #     但小病灶 80% 为高 CNR、大病灶 80% 为低 CNR：不分层 W ≈ 0.3，分层 W ≈ 0
    rows = []
    for c in range(30):
        for sz, cn in (("<10", "high" if c < 24 else "low"), (">20", "low" if c < 24 else "high")):
            sh = 0.95 if cn == "high" else 0.65
            rows.append(dict(case=f"c{c}", lesion=len(rows), size_stratum=sz, cnr_stratum=cn,
                             auc_0=sh, auc_1=sh, auc_2=0.9, auc_3=0.95 + rng.normal(0, 0.01),
                             auc_4=0.9, auc_5=np.nan, ims_auc=0.8))
    r = decide_w(pd.DataFrame(rows))
    assert r["W_unstratified_v1"] > 0.2, r["W_unstratified_v1"]
    assert abs(r["W"]) < 0.05 and not r["GO"], r["W"]
    low = table(0.30, 0.0)
    low[["auc_0", "auc_1"]] = 0.7                  # 浅层本来就弱：即使 drop 大也不 GO
    low["auc_3"] = np.where(low.size_stratum == "<10", 0.4, 0.8)
    assert not decide_w(low)["GO"]
    assert decide_s(table(0.0, 0.15))["GO"] and not decide_s(table(0.0, 0.0))["GO"]
    t = table(0.0, 0.15)
    t["auc_3"] = t["auc_4"] = np.nan
    r = decide_s(t)
    assert r["inconclusive"] and not r["GO"]
    # 5b 位置混杂：特征只编码块内位置（坐标平方 + 噪声）。v1 采样（同块中心 vs 四周）AUC ≈ 1，
    #    v2 位置匹配采样（其他块同一位置）AUC ≈ 0.5；位置对照也应 ≈ 1
    n8 = 16
    gz, gy, gx = np.indices((n8,) * 3) - (n8 - 1) / 2

    def posfeat():
        return np.stack([gz ** 2, gy ** 2, gx ** 2, rng.normal(0, 1, gz.shape)]).astype(np.float32) / 50

    lesm = np.zeros((n8,) * 3, bool)
    lesm[7:9, 7:9, 7:9] = True
    far8 = (np.abs(gz) > 4) | (np.abs(gy) > 4) | (np.abs(gx) > 4)
    v1, v2, ct = [], [], []
    ok_all = np.ones(n8 ** 3, bool)
    for c in range(20):
        f = posfeat() + rng.normal(0, 0.01, (4,) + (n8,) * 3).astype(np.float32)
        smp = stage_samples(f, lesm.astype(np.float32), far8.astype(np.float32), 0, rng)
        v1.append(dict(case=f"c{c}", lesion=0, pos=smp[0], neg=smp[1]))
        negs = [posfeat() for _ in range(3)]
        smp = stage_samples_v2(f, lesm.astype(np.float32), negs, [ok_all] * 3, rng)
        v2.append(dict(case=f"c{c}", lesion=0, pos=smp[0], neg=smp[1]))
        smp = control_samples(negs[0], ok_all, rng)
        ct.append(dict(case=f"c{c}", lesion=-1, pos=smp[0], neg=smp[1]))
    fo = case_folds([f"c{c}" for c in range(20)])
    a1 = np.median([r["auc"] for r in probe_stage(v1, fo)])
    a2 = np.median([r["auc"] for r in probe_stage(v2, fo)])
    a3 = np.median([r["auc"] for r in probe_stage(ct, fo)])
    assert a1 > 0.95 and abs(a2 - 0.5) < 0.2 and a3 > 0.95, (a1, a2, a3)
    # 5c 负样本块中心：在远处肝实质内、两两相距 ≥ 30 mm、最多 8 个
    farm = np.zeros((60, 120, 120), bool)
    farm[5:55, 10:110, 10:110] = True
    cs = pick_neg_centers(farm, (1.0, 0.78, 0.78))
    assert 1 <= len(cs) <= NEG_PATCHES and all(farm[c] for c in cs)
    for i in range(len(cs)):
        for j in range(i):
            dd = np.sqrt(sum(((a - b) * sp) ** 2 for a, b, sp in zip(cs[i], cs[j], (1.0, 0.78, 0.78))))
            assert dd >= NEG_CENTER_SEP_MM - 1e-6, dd
    assert pick_neg_centers(np.zeros((5, 5, 5), bool), (1, 1, 1)) == []
    # 6 编码器特征抽取：hook 拿到各 stage 输出，深监督列表输出取第一个
    import torch

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv3d(1, 4, 3, padding=1)

        def forward(self, x):
            f = self.c(x)
            return [f, torch.nn.functional.avg_pool3d(f, 2)]

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.head = Enc(), torch.nn.Conv3d(4, 3, 1)

        def forward(self, x):
            return [self.head(self.encoder(x)[0])]

    enc, pred = encoder_features(Net().eval(), np.zeros((1, 8, 8, 8), np.float32), torch.device("cpu"))
    assert [e.shape for e in enc] == [(4, 8, 8, 8), (4, 4, 4, 4)] and pred.shape == (8, 8, 8)
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("extract", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--cohort", choices=sorted(COHORTS), required=True)
        p.add_argument("--out", default=None)
    e = sub.choices["extract"]
    e.add_argument("--runs", nargs="+", default=["A13b:0,1,2", "A0:0,1,2"])
    e.add_argument("--image-dir", default=None)
    e.add_argument("--label-dir", default=None)
    e.add_argument("--cases-file", default=None)
    e.add_argument("--results-root", default=None)
    e.add_argument("--shard", default=None, help="i/n")
    e.add_argument("--ims", choices=("sgf", "all"), default="sgf",
                   help="sgf：只对低 CNR 且 > 20 mm 的病灶算 Ims（判据需要的范围）")
    e.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.selftest or a.cmd is None:
        selftest()
        if a.cmd is None:
            return
    {"extract": cmd_extract, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
