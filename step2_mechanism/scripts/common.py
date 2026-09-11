"""
Step 2 机制验证 — 公共工具。

关键约定
--------
* 所有实验只读，不写任何 nnUNet_data 下的文件；产出统一写入 step2_mechanism/{out,figs}。
* 机制验证一律使用 **fold 0 验证集（21 例）**，测试集 27 例保持锁定（validation_plan §2.1）。
* PC 算子直接 import 仓库代码，保证与训练/推理路径一致。
* |∇f| 对照同样使用仓库的 spatial_gradient_magnitude（PCSF grad 路径）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJ = Path("/root/autodl-tmp/Downstream_2")
NNUNET = PROJ / "nnUNet"
DATA = Path("/root/autodl-tmp/nnUNet_data")
PRE = DATA / "nnUNet_preprocessed" / "Dataset102_LiTS"
RAW = DATA / "nnUNet_raw" / "Dataset102_LiTS"
GTSEG = PRE / "gt_segmentations"
PLANS_JSON = PRE / "nnUNetPlans.json"
DATASET_JSON = PRE / "dataset.json"
OUT = PROJ / "step2_mechanism" / "out"
FIGS = PROJ / "step2_mechanism" / "figs"

# 让仓库 phaser 代码可被 import（与训练时 PYTHONPATH 一致）
if str(NNUNET) not in sys.path:
    sys.path.insert(0, str(NNUNET))

OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# 与训练一致的 PFCR 内部 PC 配置（PHASER_PC_SCALES=3）
PC_N_SCALES = 3
# 模型实际使用的 FFT 体素上限；仅用于"模型真实行为"对照
PC_MAX_VOXELS_MODEL = 1 << 20
# 机制分析用"不降采样"上限，避免 PC 分辨率被砍混淆结论
PC_MAX_VOXELS_FULL = 1 << 26

CKPTS = {
    "copy2_ep48": DATA / "nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/_old_arch_backup/checkpoint_best-Copy2.pth",
    "copy1": DATA / "nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/_old_arch_backup/checkpoint_best-Copy1.pth",
    "run_best_overwritten": DATA / "nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
    "retrain_ema_best": DATA / "nnUNet_results_retrain/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
    "retrain_best_tumor": DATA / "nnUNet_results_retrain/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/checkpoint_best_tumor_dice.pth",
    "scratch_ema_best": DATA / "nnUNet_results_scratch/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
    "repro0905_ema_best": DATA / "nnUNet_results_repro_20260905/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
}


def splits():
    d = json.load(open(PRE / "splits_final.json"))[0]
    test = json.load(open(PRE / "test_set.json"))["test"]
    return d["train"], d["val"], test


def load_preprocessed(case: str):
    """返回 (data_cf [D,H,W] float32, seg [D,H,W] int16)。seg 来自 gt_segmentations。

    注意：本仓库 preprocessed 下的 `gt_segmentations` 是 **原始分辨率** 的 GT
    （与 nnUNet_raw 的 labelsTr 逐体素一致，未经重采样），而 `data_cf` 已重采样到
    计划间距 1.0 / 0.779 / 0.779 mm。二者网格不一致，**不要用于联合分析**；
    需要同网格数据请用 `load_raw_resampled`。本函数保留仅用于形状自检。
    """
    import SimpleITK as sitk

    arr = np.load(PRE / f"nnUNetPlans_3d_fullres/{case}.npy").astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(GTSEG / f"{case}.nii.gz"))).astype(np.int16)
    if arr.shape != seg.shape:
        raise RuntimeError(f"{case} 形状不一致 data={arr.shape} seg={seg.shape}")
    return arr, seg


# nnUNetPlans 3d_fullres 的计划间距（SimpleITK 顺序 x, y, z）
PLAN_SPACING_SITK = (0.7791720032691956, 0.7791720032691956, 1.0)


def load_raw_resampled(case: str, want_img: bool = True):
    """把**原始** CT 与 GT 重采样到计划间距，返回 (img [D,H,W] HU float32, seg int16)。

    为什么不用 preprocessed 目录：
      data_cf(.npy)          已重采样（网络输入），但伴随的 gt_segmentations 未重采样；
      二者网格不一致，无法做"输入 ↔ 标签"的区域统计。
    本函数用 SimpleITK 按计划间距重建同网格数据，并使 img 保持 HU 量纲
    （CNR = ΔHU/σ_HU 可直接解释；nnU-Net 的 CTNormalization 是全局仿射变换，CNR 不变）。
    重采样后 numpy 轴序为 (z, y, x)。
    """
    import SimpleITK as sitk

    im = sitk.ReadImage(str(RAW / "imagesTr" / f"{case}_0000.nii.gz"))
    lb = sitk.ReadImage(str(RAW / "labelsTr" / f"{case}.nii.gz"))
    sz, osp = im.GetSize(), im.GetSpacing()
    new = [int(round(sz[i] * osp[i] / PLAN_SPACING_SITK[i])) for i in range(3)]

    def _rs(interp, img):
        f = sitk.ResampleImageFilter()
        f.SetOutputSpacing(PLAN_SPACING_SITK)
        f.SetSize(new)
        f.SetOutputDirection(im.GetDirection())
        f.SetOutputOrigin(im.GetOrigin())
        f.SetInterpolator(interp)
        f.SetDefaultPixelValue(0)
        return f.Execute(img)

    img = None
    if want_img:
        img = sitk.GetArrayFromImage(_rs(sitk.sitkLinear, im)).astype(np.float32)
    seg = sitk.GetArrayFromImage(_rs(sitk.sitkNearestNeighbor, lb)).astype(np.int16)
    return img, seg


def load_input_and_seg(case: str):
    """返回 (data_cf [D,H,W] float32 — 精确网络输入, seg_resampled int16 — 同网格标签)。

    seg 由原始 GT 重采样得到；先断言其网格与 data_cf 一致。
    """
    arr = np.load(PRE / f"nnUNetPlans_3d_fullres/{case}.npy").astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    _, seg = load_raw_resampled(case, want_img=False)
    if arr.shape != seg.shape:
        raise RuntimeError(f"{case} 重采样后仍不一致 data={arr.shape} seg={seg.shape}")
    return arr, seg


def preprocessed_spacing():
    """预处理后的体素间距（mm），用于把体素距离换算成物理距离。"""
    plans = json.load(open(PLANS_JSON))
    cfg = plans["configurations"]["3d_fullres"]
    sp = cfg["spacing"]
    # nnU-Net 内部 (x, y, z) -> numpy 数组轴序 (z, y, x)
    return float(sp[2]), float(sp[1]), float(sp[0])


def boundary_shell(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """形态学边界壳层（mask 减去其腐蚀）。"""
    from scipy import ndimage as ndi

    er = ndi.binary_erosion(mask, iterations=iters, border_value=0)
    return mask & ~er


def dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.binary_dilation(mask, iterations=iters, border_value=0)


def erode(mask: np.ndarray, iters: int) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.binary_erosion(mask, iterations=iters, border_value=0)


def lesions_from_seg(seg: np.ndarray, min_voxels: int = 50):
    """返回 [(label_id, mask, size), ...]，按体素数降序。"""
    from scipy import ndimage as ndi

    lab, n = ndi.label(seg == 2)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        s = int(m.sum())
        if s >= min_voxels:
            out.append((i, m, s))
    out.sort(key=lambda t: -t[2])
    return out


def crop_bbox(mask: np.ndarray, margin: int, max_side: int | None = None):
    """以 mask 为前景取 bbox 并加 margin，返回 (slices, 是否被截断)。"""
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, np.array(mask.shape))
    if max_side is not None:
        center = (lo + hi) // 2
        half = max_side // 2
        lo = np.maximum(center - half, 0)
        hi = np.minimum(lo + max_side, np.array(mask.shape))
        lo = np.maximum(hi - max_side, 0)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return sl


def minmax01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def dumpload_json(p: Path):
    return json.load(open(p))


def save_json(obj, name: str):
    p = OUT / name
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    print(f"[write] {p}")
    return p


def banner(title: str):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
