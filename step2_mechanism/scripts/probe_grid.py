#!/usr/bin/env python3
"""探查 Dataset102 原始/预处理数据的网格与 spacing，确定 4.2/4.4 的正确取数方式。"""
import json
import pickle
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PRE = Path("/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset102_LiTS")
RAW = Path("/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS")
PLANS = json.load(open(PRE / "nnUNetPlans.json"))
print("plan spacing (x,y,z):", PLANS["configurations"]["3d_fullres"]["spacing"])
print("plan median_image_size_in_voxels:", PLANS["configurations"]["3d_fullres"]["median_image_size_in_voxels"])
print("plan original_median_shape_after_transp:", PLANS.get("original_median_shape_after_transp"))
print()

for case in ["LiTS_001", "LiTS_006", "LiTS_009"]:
    print("=" * 60)
    print(case)
    npy = np.load(PRE / f"nnUNetPlans_3d_fullres/{case}.npy")
    print("  data_cf .npy shape:", npy.shape, npy.dtype)
    with open(PRE / f"nnUNetPlans_3d_fullres/{case}.pkl", "rb") as f:
        props = pickle.load(f)
    print("  pkl keys:", list(props.keys()))
    print("  shape_before_cropping:", props.get("shape_before_cropping"))
    print("  bbox_used_for_cropping:", props.get("bbox_used_for_cropping"))
    print("  spacing:", props.get("spacing"))
    g = sitk.ReadImage(str(PRE / f"gt_segmentations/{case}.nii.gz"))
    print("  gt_segmentations size:", g.GetSize(), "spacing:", tuple(round(v, 4) for v in g.GetSpacing()))
    im = sitk.ReadImage(str(RAW / "imagesTr" / f"{case}_0000.nii.gz"))
    print("  raw image size:", im.GetSize(), "spacing:", tuple(round(v, 4) for v in im.GetSpacing()))
    lb = sitk.ReadImage(str(RAW / "labelsTr" / f"{case}.nii.gz"))
    print("  raw label size:", lb.GetSize(), "spacing:", tuple(round(v, 4) for v in lb.GetSpacing()))
    a = sitk.GetArrayFromImage(lb)
    print("  raw label values:", np.unique(a), "tumor voxels:", int((a == 2).sum()))
    ga = sitk.GetArrayFromImage(g)
    print("  gt_seg values:", np.unique(ga), "tumor voxels:", int((ga == 2).sum()))
