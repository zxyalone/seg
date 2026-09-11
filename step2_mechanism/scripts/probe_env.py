#!/usr/bin/env python3
"""Step 2 机制验证 — 环境与资产盘点探针（只读）。"""
import json
import os
import sys
from pathlib import Path

print("=" * 70)
print("PYTHON:", sys.version.split()[0], sys.executable)

print("-" * 70)
print("依赖版本")
for m in ["numpy", "scipy", "matplotlib", "torch", "SimpleITK", "skimage", "pandas"]:
    try:
        mod = __import__(m)
        print(f"  {m:12s} {getattr(mod, '__version__', 'n/a')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:12s} MISSING ({e})")
import matplotlib  # noqa: E402
print("  matplotlib backend:", matplotlib.get_backend())
import torch  # noqa: E402
print("  torch.cuda.is_available:", torch.cuda.is_available())

DATA = Path("/root/autodl-tmp/nnUNet_data")
RAW = DATA / "nnUNet_raw"
PRE = DATA / "nnUNet_preprocessed"

print("-" * 70)
print("数据资产")
for p in [
    RAW / "Dataset102_LiTS" / "imagesTr",
    RAW / "Dataset102_LiTS" / "labelsTr",
    RAW / "Dataset102_LiTS" / "imagesTs",
    RAW / "Dataset101_LiTS" / "labelsTr",
    PRE / "Dataset102_LiTS" / "nnUNetPlans_3d_fullres",
    PRE / "Dataset102_LiTS" / "gt_segmentations",
]:
    if p.is_dir():
        n = len(list(p.glob("*.nii.gz")))
        print(f"  [dir ] {p}  ({n} nii.gz)")
    elif p.is_file():
        print(f"  [file] {p}")
    else:
        print(f"  [MISS] {p}")

splits = json.load(open(PRE / "Dataset102_LiTS" / "splits_final.json"))
test = json.load(open(PRE / "Dataset102_LiTS" / "test_set.json"))["test"]
tr, va = splits[0]["train"], splits[0]["val"]
print(f"  split: train={len(tr)} val={len(va)} test={len(test)}")
print(f"  overlap train∩val={len(set(tr) & set(va))} "
      f"train∩test={len(set(tr) & set(test))} val∩test={len(set(va) & set(test))}")

print("-" * 70)
print("checkpoint 清单（PFCR / 相关）")
CKPTS = {
    "Copy2 (epoch48, README 主表)": DATA / "nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/_old_arch_backup/checkpoint_best-Copy2.pth",
    "Copy1": DATA / "nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/_old_arch_backup/checkpoint_best-Copy1.pth",
    "run checkpoint_best (被覆盖)": DATA / "nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
    "retrain EMA-best (~ep101)": DATA / "nnUNet_results_retrain/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
    "retrain best_tumor_dice": DATA / "nnUNet_results_retrain/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/checkpoint_best_tumor_dice.pth",
    "scratch (A8 等价)": DATA / "nnUNet_results_scratch/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
}
for name, p in CKPTS.items():
    if p.is_file():
        print(f"  [ok] {name:34s} {p.stat().st_size/1e6:.1f} MB")
        print(f"       {p}")
    else:
        print(f"  [--] {name:34s} NOT FOUND")

print("-" * 70)
print("其它 results 目录")
for d in sorted(DATA.glob("nnUNet_results*")):
    if d.is_dir():
        subs = list(d.glob("Dataset102_LiTS/*/fold_0"))
        print(f"  {d.name}: {len(subs)} fold_0")

print("=" * 70)
