#!/usr/bin/env python3
"""
D0 检出探针：冻结的 A13b 深层特征上的线性病灶检出头，能否找回 A13b 漏检的病灶，
且明显优于直接降低分割阈值。计划与事先登记的判据：docs/解码端病灶检出计划.md §3。

流程
----
1 extract --split train   LiTS fold-0 训练集 83 例。整卷滑窗（只跑编码器，块与高斯权重同 nnU-Net），
                          第 3 层（1/8，256 通道）与上采样的第 4 层（1/16 → 1/8，320 通道）拼成 576 通道，
                          对照为第 1 层（1/2，64 通道）平均池化到 1/8；各块按高斯权重在 1/8 格子上加权平均。
                          正样本：肿瘤占比 ≥ 0.5 的格子（每个 ≥ 50 体素的病灶没有这样的格子时取其占比最大的格子），
                          每个病灶最多 64 个；负样本：肝实质占比 ≥ 0.5 且无肿瘤的格子，每例最多 2000 个。
2 fit                     每个 seed、每种特征（deep / shallow）一个 L2 逻辑回归（同 freq_feature_probe.fit_logreg）。
3 extract --split msd08   整卷：nnU-Net 原生滑窗 logits → 导出到原始网格（同 missed_lesion_probability.py），
                          argmax、阈值基线与病灶匹配调用 missed_lesion_probability.analyse_case；
                          检出头概率图按导出概率的路径映射到原始网格，在参照肝外扩 6 mm 内取 26 连通域。
4 summarize --split msd08 完整性核对（与归档病灶表逐 seed 比较）与 §3.4 判据。

判据（计划 §3.4）
  工作点：每个 seed 取新增假阳性 ≤ 0.5 个/例的最低阈值（头部 t*、基线 t'*）。
  R_head / R_thr = 被找回的 A13b 漏检病灶（体积 ≥ 30.4 mm³）比例，3 个 seed 平均。
  GO：R_head ≥ 0.25，且 R_head − R_thr ≥ 0.10、病例 bootstrap 95% CI 下限 > 0。

用法（服务器）
  python experiments/analysis/det_head_probe.py --selftest
  python experiments/analysis/det_head_probe.py extract --split train --shard 0/6   # …每片一个进程
  python experiments/analysis/det_head_probe.py fit
  python experiments/analysis/det_head_probe.py extract --split msd08 --shard 0/6
  python experiments/analysis/det_head_probe.py summarize --split msd08
只读：不写 nnUNet_data 下任何文件；产出写入 results/det_probe/。
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

PLAN = REPO / "docs" / "解码端病灶检出计划.md"
OUT_ROOT = REPO / "results" / "det_probe"
CELL = 8
DEEP_STAGES = (3, 4)
SHALLOW_STAGE = 1
POS_PER_LESION = 64
NEG_PER_CASE = 2000
MIN_LESION_PLAN_VOX = 50
PLAN_VOX_MM3 = 1.0 * 0.7791720032691956 ** 2
MIN_LESION_MM3 = MIN_LESION_PLAN_VOX * PLAN_VOX_MM3          # 30.4 mm³
ROI_DILATE_MM = 6.0
HEAD_T = (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
FP_BUDGET = 0.5
FP_BUDGET_DESC = 1.0
R_MIN, DIFF_MIN = 0.25, 0.10
N_BOOT, BOOT_SEED = 2000, 0
INTEGRITY_TOL_PP = 2.0
HEADS = ("deep", "shallow")

SPLITS = {
    "train": dict(image_dir="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/imagesTr",
                  label_dir="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/labelsTr",
                  splits="/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset102_LiTS/splits_final.json"),
    "msd08": dict(image_dir="/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08/imagesTs",
                  label_dir="/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08/labelsTs",
                  cases_file="/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08/cases.txt",
                  archive=str(REPO / "results" / "external_msd08" / "tables" / "lesions_pp4mm.csv")),
    "lits_val": dict(image_dir="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/imagesTr",
                     label_dir="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/labelsTr",
                     cases_file=str(REPO / "experiments" / "eval" / "val_cases.txt")),
}
DECISION_SPLIT = "msd08"


# ---------------------------------------------------------------------------
# 纯函数（单元测试覆盖）
# ---------------------------------------------------------------------------

def cell_gaussian(patch) -> np.ndarray:
    """nnU-Net 的滑窗高斯（compute_gaussian：中心 1、σ = 块长 / 8、以最大值归一）在格子上取块平均。"""
    tmp = np.zeros(patch)
    tmp[tuple(p // 2 for p in patch)] = 1
    g = ndi.gaussian_filter(tmp, [p / 8 for p in patch], 0, mode="constant", cval=0)
    g = g / g.max()
    g[g == 0] = g[g > 0].min()
    s = [p // CELL for p in patch]
    return g.reshape(s[0], CELL, s[1], CELL, s[2], CELL).mean(axis=(1, 3, 5)).astype(np.float32)


def dilated_roi(liver: np.ndarray, spacing, mm: float = ROI_DILATE_MM) -> np.ndarray:
    """肝脏掩膜外扩 mm（欧氏距离，按体素间距）；只在包围盒加边距内算距离变换。"""
    out = np.zeros(liver.shape, bool)
    if not liver.any():
        return out
    idx = np.argwhere(liver)
    pad = [int(np.ceil(mm / sp)) + 1 for sp in spacing]
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + np.array(pad) + 1, np.array(liver.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    out[sl] = ndi.distance_transform_edt(~liver[sl], sampling=spacing) <= mm
    return out


def cell_grid(padded_shape) -> tuple:
    return tuple(int(np.ceil(s / CELL)) + 1 for s in padded_shape)


def tile_cell_offset(slicer) -> tuple:
    """nnU-Net 块切片（第 0 维为通道）→ 该块在 1/8 格子上的起点 round(start / 8)。"""
    return tuple(int(round(sl.start / CELL)) for sl in slicer[1:])


def block_fraction_grid(mask: np.ndarray, grid) -> np.ndarray:
    """体素掩膜（以填充后的网格为原点）→ 1/8 格子上的占比；先补零到 grid × 8。"""
    full = np.zeros(tuple(g * CELL for g in grid), np.float32)
    full[tuple(slice(0, s) for s in mask.shape)] = mask
    return full.reshape(grid[0], CELL, grid[1], CELL, grid[2], CELL).mean(axis=(1, 3, 5))


def tile_features(enc):
    """编码器各 stage 输出（torch，[1, C, *sp]）→ (deep [576, n, n, n], shallow [64, n, n, n])，n = 第 3 层边长。"""
    import torch
    import torch.nn.functional as F
    d3, d4, s1 = enc[DEEP_STAGES[0]], enc[DEEP_STAGES[1]], enc[SHALLOW_STAGE]
    tgt = d3.shape[2:]
    d4u = F.interpolate(d4.float(), size=tgt, mode="nearest")
    k = s1.shape[2] // tgt[0]
    s1p = F.avg_pool3d(s1.float(), kernel_size=k)
    return torch.cat([d3.float(), d4u], 1)[0], s1p[0]


def train_samples(deep, shallow, seg_cells_t, seg_cells_l, lesion_cells, rng):
    """deep [Cd, *g]、shallow [Cs, *g]；seg_cells_t / seg_cells_l：肿瘤 / 肝实质占比（*g）；
    lesion_cells：[(扁平下标数组, 占比数组)]（每个 ≥ 50 体素的病灶覆盖的格子）。返回 dict 的 fp16 数组。"""
    t, l = seg_cells_t.ravel(), seg_cells_l.ravel()
    pos = []
    for idx, frac in lesion_cells:
        sel = idx[frac >= 0.5]
        if sel.size == 0:
            sel = idx[[int(np.argmax(frac))]]
        if sel.size > POS_PER_LESION:
            sel = rng.choice(sel, POS_PER_LESION, replace=False)
        pos.append(sel)
    pos = np.unique(np.concatenate(pos)) if pos else np.zeros(0, int)
    neg = np.flatnonzero((l >= 0.5) & (t == 0))
    if neg.size > NEG_PER_CASE:
        neg = np.sort(rng.choice(neg, NEG_PER_CASE, replace=False))
    out = {}
    for name, f in (("deep", deep), ("shallow", shallow)):
        flat = f.reshape(f.shape[0], -1)
        out[f"{name}_pos"] = flat[:, pos].T.astype(np.float16)
        out[f"{name}_neg"] = flat[:, neg].T.astype(np.float16)
    return out


def lesion_cell_list(tumor_lab: np.ndarray, grid, min_voxels=MIN_LESION_PLAN_VOX):
    """26 连通病灶（≥ min_voxels）→ [(覆盖格子的扁平下标, 每格该病灶占比)]；只在包围盒内计算。"""
    out = []
    for i, sl in enumerate(ndi.find_objects(tumor_lab), start=1):
        if sl is None:
            continue
        m = tumor_lab[sl] == i
        if m.sum() < min_voxels:
            continue
        lo = [s.start // CELL for s in sl]
        hi = [-(-s.stop // CELL) for s in sl]
        box = np.zeros(tuple((h - o) * CELL for o, h in zip(lo, hi)), np.float32)
        box[tuple(slice(s.start - o * CELL, s.stop - o * CELL) for s, o in zip(sl, lo))] = m
        n = [h - o for o, h in zip(lo, hi)]
        frac = box.reshape(n[0], CELL, n[1], CELL, n[2], CELL).mean(axis=(1, 3, 5))
        ii = np.argwhere(frac > 0)
        flat = np.ravel_multi_index(tuple((ii + np.array(lo)).T), grid)
        out.append((flat, frac[frac > 0]))
    return out


def apply_head(F: np.ndarray, head: dict) -> np.ndarray:
    """F [C, *g] → 概率 [*g]。"""
    c = F.shape[0]
    z = ((F.reshape(c, -1).T - head["mu"]) / head["sd"]) @ head["w"] + head["b"]
    return (1 / (1 + np.exp(-z))).reshape(F.shape[1:]).astype(np.float32)


def cells_to_voxels(p_cells: np.ndarray, padded_shape, revert) -> np.ndarray:
    """1/8 格子 → 预处理网格（每格复制为 8³，裁到填充形状，再去掉填充）。"""
    up = p_cells.repeat(CELL, 0).repeat(CELL, 1).repeat(CELL, 2)
    up = up[tuple(slice(0, s) for s in padded_shape)]
    return up[tuple(revert)]


def head_detection(head_raw: np.ndarray, roi: np.ndarray, gt_lab: np.ndarray, block: np.ndarray, thresholds):
    """每个阈值：头部连通域（26 连通，roi 内）；被命中的病灶编号集合；与 block（金标准肿瘤 ∪ A13b 预测）
    不重叠的连通域数 = 新增假阳性。"""
    import evaluate as E
    if roi.any():   # ROI 外不计，裁到 ROI 包围盒内以减少连通域标记的开销
        sl = tuple(slice(int(np.flatnonzero(np.any(roi, axis=tuple(j for j in range(3) if j != i))).min()),
                         int(np.flatnonzero(np.any(roi, axis=tuple(j for j in range(3) if j != i))).max()) + 1)
                   for i in range(3))
        head_raw, roi, gt_lab, block = head_raw[sl], roi[sl], gt_lab[sl], block[sl]
    out = {}
    for t in thresholds:
        lab, n = E.components((head_raw > t) & roi, 26)
        if n == 0:
            out[t] = (set(), 0, 0)
            continue
        on = lab > 0
        hit = set(int(v) for v in np.unique(gt_lab[on & (gt_lab > 0)]))
        touching = np.unique(lab[on & block])
        out[t] = (hit, int(n - touching.size), int(n))
    return out


def pick_threshold(fp_by_t: dict, budget: float):
    """fp_by_t：{阈值: 每例平均新增假阳性}，阈值越低检出越多；取满足预算的最低阈值，没有则 None。"""
    ok = [t for t, v in fp_by_t.items() if v <= budget + 1e-12]
    return min(ok) if ok else None


def recovery_rates(les, cases, budget=FP_BUDGET):
    """les：每个 run × 病灶一行（run, case, missed, head_<h>_<t>, det_pp@<t'>）；
    cases：每个 run × 病例一行（run, case, xfp_<h>_<t>, xfp_thr_<t'>）。
    返回 {run: dict(t_head, t_thr, R_head, R_thr, R_shallow, n_missed)}。"""
    res = {}
    for run, c in cases.groupby("run"):
        l = les[(les.run == run) & les.missed]
        r = dict(n_missed=int(len(l)))
        for h in HEADS:
            t = pick_threshold({t: c[f"xfp_{h}_{t:g}"].mean() for t in HEAD_T}, budget)
            r[f"t_{h}"] = t
            r[f"R_{h}"] = float(l[f"head_{h}_{t:g}"].mean()) if (t is not None and len(l)) else 0.0
        tt = pick_threshold({t: c[f"xfp_thr_{t:g}"].mean() for t in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)}, budget)
        r["t_thr"] = tt
        r["R_thr"] = float(l[f"det_pp@{tt:g}"].mean()) if (tt is not None and len(l)) else 0.0
        res[run] = r
    return res


def rates_at(les, picks, run_cases=None):
    """固定工作点下的平均找回率（bootstrap 用）。picks = recovery_rates 的输出。"""
    rh, rt = [], []
    for run, p in picks.items():
        l = les[(les.run == run) & les.missed]
        if not len(l):
            continue
        rh.append(l[f"head_deep_{p['t_deep']:g}"].mean() if p["t_deep"] is not None else 0.0)
        rt.append(l[f"det_pp@{p['t_thr']:g}"].mean() if p["t_thr"] is not None else 0.0)
    return float(np.mean(rh)), float(np.mean(rt))


def decide(les, cases):
    picks = recovery_rates(les, cases, FP_BUDGET)
    r_head = float(np.mean([p["R_deep"] for p in picks.values()]))
    r_thr = float(np.mean([p["R_thr"] for p in picks.values()]))
    groups = {c: idx.to_numpy() for c, idx in les.reset_index(drop=True).groupby("case").groups.items()}
    L = les.reset_index(drop=True)
    case_ids = np.array(sorted(groups))
    rng = np.random.default_rng(BOOT_SEED)
    diffs = []
    for _ in range(N_BOOT):
        pick = rng.choice(case_ids, case_ids.size, replace=True)
        sub = L.iloc[np.concatenate([groups[c] for c in pick])]
        h, t = rates_at(sub, picks)
        diffs.append(h - t)
    lo, hi = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    go = bool(r_head >= R_MIN and (r_head - r_thr) >= DIFF_MIN and lo > 0)
    return dict(R_head=r_head, R_thr=r_thr, diff=r_head - r_thr, ci=[lo, hi], per_run=picks, GO=go)


# ---------------------------------------------------------------------------
# 推理（服务器）
# ---------------------------------------------------------------------------

def window_features(m, data: np.ndarray, device):
    """只跑编码器的整卷滑窗（块、步长、填充与 nnU-Net 相同）；返回 (deep, shallow, 填充后形状, 去填充切片)。"""
    import torch
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    pred, net = m.predictor, m.network
    patch = tuple(int(v) for v in pred.configuration_manager.patch_size)
    dp, revert = pad_nd_image(torch.from_numpy(data), patch, "constant", {"value": 0}, True, None)
    slicers = pred._internal_get_sliding_window_slicers(dp.shape[1:])
    g = cell_gaussian(patch)
    grid = cell_grid(dp.shape[1:])
    acc_d = acc_s = None
    wsum = np.zeros(grid, np.float32)
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for sl in slicers:
            enc = net.encoder(dp[sl][None].to(device))
            d, s = tile_features(enc)
            d, s = d.cpu().numpy(), s.cpu().numpy()
            if acc_d is None:
                acc_d = np.zeros((d.shape[0], *grid), np.float32)
                acc_s = np.zeros((s.shape[0], *grid), np.float32)
            o = tile_cell_offset(sl)
            reg = tuple(slice(oi, oi + n) for oi, n in zip(o, g.shape))
            acc_d[(slice(None), *reg)] += d * g
            acc_s[(slice(None), *reg)] += s * g
            wsum[reg] += g
    w = np.maximum(wsum, 1e-6)
    return acc_d / w, acc_s / w, tuple(dp.shape[1:]), list(revert)[1:]


def export_map(map_pp: np.ndarray, pred, props: dict) -> np.ndarray:
    """单通道图（预处理网格）按 nnU-Net 导出概率的路径还原到原始网格：重采样、去裁剪、转置。"""
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
    cm, pm = pred.configuration_manager, pred.plans_manager
    cur = cm.spacing if len(cm.spacing) == len(props["shape_after_cropping_and_before_resampling"]) \
        else [props["spacing"][0], *cm.spacing]
    r = cm.resampling_fn_probabilities(map_pp[None].astype(np.float32),
                                       props["shape_after_cropping_and_before_resampling"], cur, props["spacing"])
    r = np.asarray(r.cpu().numpy() if hasattr(r, "cpu") else r)[0]
    out = np.zeros(props["shape_before_cropping"], np.float32)
    out[bounding_box_to_slice(props["bbox_used_for_cropping"])] = r
    return out.transpose(pm.transpose_backward)


def case_rng(case: str):
    return np.random.default_rng(int(hashlib.sha256(case.encode()).hexdigest()[:8], 16))


def split_cases(split: str, cfg: dict, image_dir: Path, C, cases_file=None) -> list:
    if cases_file:
        return C.read_cases(cases_file, image_dir)
    if split == "train":
        return sorted(json.loads(Path(cfg["splits"]).read_text())[0]["train"])
    return C.read_cases(cfg["cases_file"], image_dir)


def manifest(out: Path, a, models, C):
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout
    C.write_json(dict(split=a.split, runs=[m.name for m in models], git_head=head.strip(), git_dirty=bool(dirty.strip()),
                      plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      started=time.strftime("%F %T")), out / "run_manifest.json")


def cmd_extract(a):
    import torch
    import common as C
    import evaluate as E
    import missed_lesion_probability as M
    from nnunetv2.inference.export_prediction import \
        convert_predicted_logits_to_segmentation_with_correct_shape

    cfg = SPLITS[a.split]
    out = Path(a.out or OUT_ROOT) / a.split
    image_dir, label_dir = Path(cfg["image_dir"]), Path(cfg["label_dir"])
    device = torch.device(a.device)
    models = [C.load_model(g, s, device, a.results_root) for g, s in C.parse_runs(a.runs)]
    ref = models[0].predictor
    heads = None
    if a.split != "train":
        hp = Path(a.out or OUT_ROOT) / "heads.npz"
        if not hp.is_file():
            sys.exit(f"{hp} missing: run extract --split train and fit first")
        z = np.load(hp)
        heads = {(m.name, h): {k: z[f"{m.name}/{h}/{k}"] for k in ("w", "b", "mu", "sd")}
                 for m in models for h in HEADS}
    cases = C.shard(split_cases(a.split, cfg, image_dir, C, a.cases_file), a.shard)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    if not a.shard or a.shard.startswith("0/"):
        manifest(out, a, models, C)
    for ci, case in enumerate(cases):
        done = out / "cases" / (f"{case}.npz" if a.split == "train" else f"{case}.json")
        if done.is_file():
            print(f"[{ci + 1}/{len(cases)}] {case}: done, skipped", flush=True)
            continue
        t0 = time.time()
        rng = case_rng(case)
        if a.split == "train":
            raw, props, seg_raw = C.read_raw(ref, case, image_dir, label_dir)
            data, seg, _ = C.preprocess(ref, raw, props, seg_raw)
            # 分割按与图像相同的方式填充，格子原点与特征一致；各 seed 共用
            # 与 window_features 中的图像走同一条 torch 填充路径（numpy 输入不接受 value 参数）
            from acvl_utils.cropping_and_padding.padding import pad_nd_image
            sp, _ = pad_nd_image(torch.from_numpy(seg[0][None].astype(np.int16)),
                                 ref.configuration_manager.patch_size, "constant", {"value": 0}, True, None)
            s0 = sp.numpy()[0]
            grid = cell_grid(s0.shape)
            tf, lf = block_fraction_grid(s0 == 2, grid), block_fraction_grid(s0 == 1, grid)
            tl, _ = E.components(s0 == 2, 26)
            lcells = lesion_cell_list(tl, grid)
            arrays = {}
            for m in models:
                deep, shallow, pshape, _ = window_features(m, data, device)
                if deep.shape[1:] != grid:
                    raise RuntimeError(f"{case}: feature grid {deep.shape[1:]} vs label grid {grid}")
                smp = train_samples(deep, shallow, tf, lf, lcells, case_rng(case))
                for k, v in smp.items():
                    arrays[f"{m.name}/{k}"] = v
            tmp = done.with_name(done.stem + ".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            tmp.replace(done)      # 原子替换：中断不会留下被当作"已完成"的半个文件
            print(f"[{ci + 1}/{len(cases)}] {case}: {len(lcells)} lesions, "
                  f"{arrays[f'{models[0].name}/deep_pos'].shape[0]} pos / "
                  f"{arrays[f'{models[0].name}/deep_neg'].shape[0]} neg cells, {time.time() - t0:.0f}s", flush=True)
            continue

        # 评估队列：原始网格上的分割、阈值基线与检出头
        gt_img, gt = E.read(label_dir / f"{case}.nii.gz")
        spacing_zyx = tuple(float(s) for s in gt_img.GetSpacing()[::-1])
        vox_mm3 = float(np.prod(spacing_zyx))
        gt_lab, n_gt = E.components(gt == 2, 26)
        cnr = {}
        if n_gt:
            cnr = E.raw_lesion_cnr(gt_lab, n_gt, gt_img, E.sitk.ReadImage(str(E.find_image(image_dir, case))))
        roi = dilated_roi(gt > 0, spacing_zyx)
        raw, props, _ = C.read_raw(ref, case, image_dir, None)
        data, _, props_pp = C.preprocess(ref, raw, props)
        rec = dict(case=case, n_gt=int(n_gt), runs={})
        for m in models:
            logits = C.predict_logits(m, data, preserve_dtype=True)
            seg, probs = convert_predicted_logits_to_segmentation_with_correct_shape(
                logits, m.predictor.plans_manager, m.predictor.configuration_manager,
                m.predictor.label_manager, props_pp, return_probabilities=True)
            del logits
            if seg.shape != gt.shape:
                raise RuntimeError(f"{case}: prediction {seg.shape} vs reference {gt.shape}")
            res = M.analyse_case(probs[2].astype(np.float32), probs[1].astype(np.float32), seg, gt, spacing_zyx, cnr)
            del probs
            pred_pp = M._drop_small(seg == 2, vox_mm3, M.MIN_DIAM_PP_MM)
            block = (gt == 2) | pred_pp
            deep, shallow, pshape, revert = window_features(m, data, device)
            run = dict(rules=res["rules"], lesions=res["lesions"], head={})
            for h, F in (("deep", deep), ("shallow", shallow)):
                pc = apply_head(F, heads[(m.name, h)])
                hr = export_map(cells_to_voxels(pc, pshape, revert), m.predictor, props_pp)
                if hr.shape != gt.shape:
                    raise RuntimeError(f"{case}: head map {hr.shape} vs reference {gt.shape}")
                det = head_detection(hr, roi, gt_lab, block, HEAD_T)
                run["head"][h] = {f"{t:g}": dict(hits=sorted(v[0]), xfp=v[1], n=v[2]) for t, v in det.items()}
            for les in run["lesions"]:
                les["volume_mm3"] = les["voxels"] * vox_mm3
            rec["runs"][m.name] = run
        tmp = done.with_name(done.stem + ".tmp.json")
        C.write_json(rec, tmp)
        tmp.replace(done)
        print(f"[{ci + 1}/{len(cases)}] {case}: {n_gt} lesions, {time.time() - t0:.0f}s", flush=True)


def cmd_check(a):
    """导出路径核对：预处理网格上的金标准（肿瘤、肝+肿瘤）经 export_map 回到原始网格，与原始金标准比 Dice。"""
    import torch
    import common as C
    import evaluate as E
    cfg = SPLITS[a.split]
    image_dir, label_dir = Path(cfg["image_dir"]), Path(cfg["label_dir"])
    m = C.load_model(*C.parse_runs(["A13b:0"])[0], torch.device("cpu"), a.results_root)
    for case in split_cases(a.split, cfg, image_dir, C, a.cases_file):
        raw, props, seg_raw = C.read_raw(m.predictor, case, image_dir, label_dir)
        data, seg, props_pp = C.preprocess(m.predictor, raw, props, seg_raw)
        _, gt = E.read(label_dir / f"{case}.nii.gz")
        msg = []
        for name, pp, ref in (("tumor", seg[0] == 2, gt == 2), ("liver+tumor", seg[0] > 0, gt > 0)):
            r = export_map(pp.astype(np.float32), m.predictor, props_pp) > 0.5
            inter = (r & ref).sum()
            msg.append(f"{name} Dice {2 * inter / max(r.sum() + ref.sum(), 1):.4f}")
        print(case, data.shape, "->", gt.shape, "; ".join(msg), flush=True)


def cmd_fit(a):
    import common as C
    from freq_feature_probe import fit_logreg
    out = Path(a.out or OUT_ROOT)
    files = sorted((out / "train" / "cases").glob("*.npz"))
    if not files:
        sys.exit("no training samples")
    runs = sorted({k.split("/")[0] for k in np.load(files[0]).files})
    heads, info = {}, {}
    for run in runs:
        for h in HEADS:
            P = np.concatenate([np.load(f)[f"{run}/{h}_pos"] for f in files]).astype(np.float32)
            N = np.concatenate([np.load(f)[f"{run}/{h}_neg"] for f in files]).astype(np.float32)
            X = np.concatenate([P, N])
            y = np.r_[np.ones(len(P)), np.zeros(len(N))]
            mu, sd = X.mean(0), X.std(0) + 1e-6
            w, b = fit_logreg((X - mu) / sd, y)
            heads.update({f"{run}/{h}/w": w, f"{run}/{h}/b": np.float64(b), f"{run}/{h}/mu": mu, f"{run}/{h}/sd": sd})
            z = ((X - mu) / sd) @ w + b
            from freq_feature_probe import auc
            info[f"{run}/{h}"] = dict(n_pos=int(len(P)), n_neg=int(len(N)), train_auc=auc(z[y == 1], z[y == 0]))
            print(run, h, info[f"{run}/{h}"], flush=True)
    np.savez(out / "heads.npz", **heads)
    C.write_json(dict(n_cases=len(files), heads=info), out / "heads_info.json")


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def tables(out: Path):
    import pandas as pd
    L, Cs = [], []
    for js in sorted((out / "cases").glob("*.json")):
        rec = json.loads(js.read_text())
        for run, r in rec["runs"].items():
            crow = dict(run=run, case=rec["case"])
            arg = r["rules"]["argmax_pp"]["fp"]
            for t in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
                crow[f"xfp_thr_{t:g}"] = r["rules"][f"{t:g}_pp"]["fp"] - arg
            for h in HEADS:
                for t in HEAD_T:
                    crow[f"xfp_{h}_{t:g}"] = r["head"][h][f"{t:g}"]["xfp"]
            Cs.append(crow)
            for les in r["lesions"]:
                row = dict(run=run, case=rec["case"], lesion=les["lesion"], volume_mm3=les["volume_mm3"],
                           diameter_mm=les["diameter_mm"], size_stratum=les["size_stratum"], cnr=les["cnr"],
                           cnr_stratum=les["cnr_stratum"], p_max=les["p_max"],
                           detected=bool(les["det_pp@argmax"]))
                for t in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
                    row[f"det_pp@{t:g}"] = bool(les[f"det_pp@{t:g}"])
                for h in HEADS:
                    for t in HEAD_T:
                        row[f"head_{h}_{t:g}"] = les["lesion"] in r["head"][h][f"{t:g}"]["hits"]
                L.append(row)
    les, cases = pd.DataFrame(L), pd.DataFrame(Cs)
    les["missed"] = (~les.detected) & (les.volume_mm3 >= MIN_LESION_MM3)
    return les, cases


def integrity(les, archive: Path):
    import pandas as pd
    if not archive.is_file():
        return None
    arc = pd.read_csv(archive)
    rows, ok = [], True
    for run, l in les.groupby("run"):
        a = arc[arc.run == run]
        if a.empty:
            continue
        mine_all, arc_all = 100 * l.detected.mean(), 100 * a.detected.mean()
        mine_low = 100 * l[l.cnr_stratum == "low"].detected.mean()
        arc_low = 100 * a[a.cnr_stratum == "low"].detected.mean()
        good = abs(mine_all - arc_all) <= INTEGRITY_TOL_PP and abs(mine_low - arc_low) <= INTEGRITY_TOL_PP
        ok &= good
        rows.append(dict(run=run, n_lesions=len(l), n_archive=len(a), recall_all=mine_all, archive_all=arc_all,
                         recall_low=mine_low, archive_low=arc_low, ok=good))
    return dict(ok=bool(ok), rows=rows)


def cmd_summarize(a):
    import pandas as pd
    import common as C
    out = Path(a.out or OUT_ROOT) / a.split
    les, cases = tables(out)
    les.to_csv(out / "lesions.csv", index=False)
    cases.to_csv(out / "cases.csv", index=False)
    cfg = SPLITS[a.split]
    integ = integrity(les, Path(cfg["archive"])) if "archive" in cfg else None
    res = dict(split=a.split, decision_split=a.split == DECISION_SPLIT, n_cases=int(cases.case.nunique()),
               n_lesions=int(les.drop_duplicates(["case", "lesion"]).shape[0]), integrity=integ)
    lines = [f"# D0 检出探针：{a.split}", "",
             f"{res['n_cases']} 例，{res['n_lesions']} 个病灶。判据见 docs/解码端病灶检出计划.md §3.4；"
             f"{'本队列为判据队列' if res['decision_split'] else '本队列只作描述'}。", ""]
    if integ is not None:
        lines += ["## 完整性核对（与归档病灶表，容差 2 个百分点）", "",
                  "| run | 病灶 | 归档 | 全部召回 | 归档 | 低 CNR 召回 | 归档 | 通过 |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['run']} | {r['n_lesions']} | {r['n_archive']} | {r['recall_all']:.1f} | {r['archive_all']:.1f} | "
                  f"{r['recall_low']:.1f} | {r['archive_low']:.1f} | {r['ok']} |" for r in integ["rows"]]
        lines.append("")
    if integ is not None and not integ["ok"]:
        res["decision"] = None
        lines += ["**完整性核对未通过：按计划 §3.3 不进入判定。**", ""]
    else:
        d = decide(les, cases)
        if not res["decision_split"]:
            d["GO"] = None
        res["decision"] = d
        lines += ["## 判据（新增假阳性 ≤ 0.5 个/例，A13b 3 个 seed 平均）", "",
                  f"- R_head = {d['R_head']:.3f}，R_thr = {d['R_thr']:.3f}，差 {d['diff']:.3f}，"
                  f"95% CI [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] → **{d['GO']}**", "",
                  "| run | 漏检病灶 | t* 头部 | R 头部 | t* 浅层对照 | R 浅层 | t'* 基线 | R 基线 |",
                  "|---|---|---|---|---|---|---|---|"]
        for run, p in d["per_run"].items():
            lines.append(f"| {run} | {p['n_missed']} | {p['t_deep']} | {p['R_deep']:.3f} | {p['t_shallow']} | "
                         f"{p['R_shallow']:.3f} | {p['t_thr']} | {p['R_thr']:.3f} |")
        lines.append("")
        # 描述性
        desc = {}
        for name, sel in (("low_cnr", les.cnr_stratum == "low"), ("partially_lit", les.p_max >= 0.5),
                          ("unlit", les.p_max < 0.5)):
            sub = les[sel]
            p = recovery_rates(sub, cases, FP_BUDGET)
            desc[name] = dict(n_missed=int(sub.missed.sum()),
                              R_deep=float(np.mean([v["R_deep"] for v in p.values()])),
                              R_thr=float(np.mean([v["R_thr"] for v in p.values()])))
        p1 = recovery_rates(les, cases, FP_BUDGET_DESC)
        desc["budget_1.0"] = dict(R_deep=float(np.mean([v["R_deep"] for v in p1.values()])),
                                  R_shallow=float(np.mean([v["R_shallow"] for v in p1.values()])),
                                  R_thr=float(np.mean([v["R_thr"] for v in p1.values()])))
        res["descriptive"] = desc
        lines += ["## 描述性", "", "| 子集 | 漏检数（3 seed 合计） | R 头部 | R 基线 |", "|---|---|---|---|"]
        for k, v in desc.items():
            lines.append(f"| {k} | {v.get('n_missed', '')} | {v['R_deep']:.3f} | {v['R_thr']:.3f} |")
        lines.append("")
    C.write_json(res, out / "summary.json")
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# 自检（合成数据，不需要模型）
# ---------------------------------------------------------------------------

def selftest():
    import pandas as pd
    import torch
    rng = np.random.default_rng(0)
    # 1 高斯与格子
    g = cell_gaussian((128, 128, 128))
    assert g.shape == (16, 16, 16) and g.min() > 0
    assert all(v in (7, 8) for v in np.unravel_index(int(g.argmax()), g.shape))
    lv = np.zeros((30, 40, 40), bool)
    lv[10:20, 10:30, 10:30] = True
    r = dilated_roi(lv, (2.0, 1.0, 1.0), 4.0)
    assert r[10:20, 10:30, 10:30].all() and r[8, 15, 15] and not r[7, 15, 15] and r[15, 6, 15] and not r[15, 5, 15]
    assert cell_grid((128, 200, 256)) == (17, 26, 33)
    assert tile_cell_offset((slice(None), slice(0, 128), slice(36, 164), slice(72, 200))) == (0, 4, 9)
    m = np.zeros((20, 20, 20), bool)
    m[0:8, 0:8, 0:4] = True
    f = block_fraction_grid(m, (3, 3, 3))
    assert f[0, 0, 0] == 0.5 and f.sum() * 512 == m.sum()
    # 2 编码器输出 → 格子特征
    enc = [torch.zeros(1, 32, 128, 128, 128), torch.ones(1, 64, 64, 64, 64), torch.zeros(1, 128, 32, 32, 32),
           torch.ones(1, 256, 16, 16, 16) * 2, torch.arange(8.).view(1, 1, 8, 1, 1).expand(1, 320, 8, 8, 8),
           torch.zeros(1, 320, 4, 4, 4)]
    d, s = tile_features(enc)
    assert d.shape == (576, 16, 16, 16) and s.shape == (64, 16, 16, 16)
    assert float(d[0].mean()) == 2 and float(d[300, 3].mean()) == 1 and float(s.mean()) == 1
    # 3 病灶格子与训练样本：大病灶取占比 ≥ 0.5 的格子，小病灶取占比最大的格子，负样本不含肿瘤
    seg = np.ones((64, 64, 64), np.int8)
    seg[8:40, 8:40, 8:40] = 2           # 大病灶
    seg[50:54, 50:54, 50:54] = 2        # 64 体素的小病灶，跨不到半格
    grid = cell_grid(seg.shape)
    import evaluate as E
    tl, n = E.components(seg == 2, 26)
    lc = lesion_cell_list(tl, grid)
    assert n == 2 and len(lc) == 2
    tf, lf = block_fraction_grid(seg == 2, grid), block_fraction_grid(seg == 1, grid)
    F = rng.normal(size=(5,) + grid).astype(np.float32)
    smp = train_samples(F, F[:2], tf, lf, lc, rng)
    assert smp["deep_pos"].shape[0] == POS_PER_LESION + 1 and smp["shallow_pos"].shape[1] == 2
    neg_idx = np.flatnonzero((lf.ravel() >= 0.5) & (tf.ravel() == 0))
    assert smp["deep_neg"].shape[0] == min(NEG_PER_CASE, neg_idx.size)
    # 4 头部与格子 → 体素
    head = dict(w=np.array([1.0, 0, 0, 0, 0]), b=0.0, mu=np.zeros(5), sd=np.ones(5))
    p = apply_head(F, head)
    assert p.shape == grid and np.allclose(p, 1 / (1 + np.exp(-F[0])), atol=1e-6)
    vox = cells_to_voxels(p, (64, 64, 64), [slice(2, 62)] * 3)
    assert vox.shape == (60, 60, 60) and vox[0, 0, 0] == p[0, 0, 0] and vox[-1, -1, -1] == p[7, 7, 7]
    # 5 头部检出：命中、新增假阳性（与金标准或 A13b 预测重叠的不算）
    gt_lab = np.zeros((40, 40, 40), np.int32)
    gt_lab[5:10, 5:10, 5:10] = 1
    gt_lab[20:25, 20:25, 20:25] = 2
    pred = np.zeros_like(gt_lab, bool)
    pred[30:33, 5:8, 5:8] = True
    hr = np.zeros(gt_lab.shape, np.float32)
    hr[6:9, 6:9, 6:9] = 0.9            # 命中病灶 1
    hr[30:34, 4:9, 4:9] = 0.9          # 与 A13b 预测重叠：不算新增
    hr[12:15, 30:33, 30:33] = 0.9      # 新增假阳性
    hr[35:38, 35:38, 35:38] = 0.3      # 低于 0.5 的阈值
    roi = np.ones(gt_lab.shape, bool)
    det = head_detection(hr, roi, gt_lab, (gt_lab > 0) | pred, (0.5, 0.2))
    assert det[0.5] == ({1}, 1, 3) and det[0.2] == ({1}, 2, 4), det
    roi[12:15] = False
    assert head_detection(hr, roi, gt_lab, (gt_lab > 0) | pred, (0.5,))[0.5][1] == 0
    roi2 = np.zeros(gt_lab.shape, bool)
    roi2[3:28, 3:35, 3:35] = True     # 裁剪后：病灶 1 与假阳性仍在，A13b 预测那块被裁掉
    assert head_detection(hr, roi2, gt_lab, (gt_lab > 0) | pred, (0.5,))[0.5] == ({1}, 1, 2)
    # 6 工作点与判据
    assert pick_threshold({0.9: 0.1, 0.5: 0.4, 0.3: 0.6}, 0.5) == 0.5
    assert pick_threshold({0.9: 0.7}, 0.5) is None

    def make(r_head, r_thr, n_case=40):
        L, Cs = [], []
        for run in ("A13b_s0", "A13b_s1", "A13b_s2"):
            for c in range(n_case):
                Cs.append(dict(run=run, case=f"c{c}", **{f"xfp_deep_{t:g}": (0.3 if t >= 0.5 else 0.9) for t in HEAD_T},
                               **{f"xfp_shallow_{t:g}": 0.2 for t in HEAD_T},
                               **{f"xfp_thr_{t:g}": (0.4 if t >= 0.2 else 0.8) for t in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)}))
                for k in range(3):
                    u = rng.random()
                    row = dict(run=run, case=f"c{c}", lesion=k, missed=True, volume_mm3=100.0,
                               **{f"head_deep_{t:g}": bool(u < r_head) for t in HEAD_T},
                               **{f"head_shallow_{t:g}": False for t in HEAD_T},
                               **{f"det_pp@{t:g}": bool(u < r_thr) for t in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)})
                    L.append(row)
        return pd.DataFrame(L), pd.DataFrame(Cs)

    l, c = make(0.5, 0.1)
    r = decide(l, c)
    assert r["GO"] and abs(r["R_head"] - 0.5) < 0.1 and r["per_run"]["A13b_s0"]["t_deep"] == 0.5 \
        and r["per_run"]["A13b_s0"]["t_thr"] == 0.2, r
    l, c = make(0.3, 0.25)
    assert not decide(l, c)["GO"]
    l, c = make(0.2, 0.0)
    assert not decide(l, c)["GO"]           # R_head < 0.25
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    e = sub.add_parser("extract")
    e.add_argument("--split", choices=sorted(SPLITS), required=True)
    e.add_argument("--runs", nargs="+", default=["A13b:0,1,2"])
    e.add_argument("--results-root", default=None)
    e.add_argument("--shard", default=None, help="i/n")
    e.add_argument("--device", default="cuda")
    e.add_argument("--out", default=None, help=f"default {OUT_ROOT}")
    e.add_argument("--cases-file", default=None, help="覆盖默认病例（冒烟测试用）")
    k = sub.add_parser("check", help="导出路径核对（预处理网格 → 原始网格）")
    k.add_argument("--split", choices=sorted(SPLITS), required=True)
    k.add_argument("--cases-file", default=None)
    k.add_argument("--results-root", default=None)
    f = sub.add_parser("fit")
    f.add_argument("--out", default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--split", choices=sorted(SPLITS), required=True)
    s.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest or a.cmd is None:
        selftest()
        if a.cmd is None:
            return
    {"extract": cmd_extract, "fit": cmd_fit, "summarize": cmd_summarize, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
