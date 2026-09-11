#!/usr/bin/env python3
"""核实 PFCR 各注入点的特征/PC/route 实际分辨率（确认 pc_max_voxels 是否触发降采样）。"""
import json
import os
import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import DATASET_JSON, PLANS_JSON, banner  # noqa: E402

os.environ.update({
    "PHASER_PFCR": "1", "PHASER_PFCR_USE_PRIOR": "1", "PHASER_SMALL_HEAD": "1",
    "PHASER_PC_SCALES": "3", "PHASER_PC_MAX_VOXELS": str(1 << 20),
    "PHASER_PCSF": "0", "PHASER_LWFD": "0", "PHASER_ACG": "0",
    "PHASER_T_RAF": "none", "PHASER_LA_ACG": "0", "PHASER_FUSION": "none",
    "VOCOMNI_NNUNET_CKPT": "",
})

banner("PFCR 分辨率核实")
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_phaser import nnUNetTrainer_phaser  # noqa: E402
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402

plans = json.load(open(PLANS_JSON))
djs = json.load(open(DATASET_JSON))
pm = PlansManager(plans)
cm = pm.get_configuration("3d_fullres")
model = nnUNetTrainer_phaser.build_network_architecture(pm, djs, cm, 1, False).eval()

for i, m in enumerate(model.pfcr_modules):
    print(f"pfcr_modules[{i}]: pc.max_voxels={m.pc.max_voxels}  n_scales={m.pc.n_scales}  "
          f"use_prior={m.use_prior}")

x = torch.randn(1, 1, 128, 128, 128)
enc_shapes = []
hooks = []
for i, st in enumerate(model.encoder.stages):
    hooks.append(st.register_forward_hook(
        lambda mod, inp, out, i=i: enc_shapes.append((i, tuple(out.shape[2:])))))
with torch.no_grad():
    model(x)
for h in hooks:
    h.remove()
print("\nencoder stage 输出空间尺寸：")
for i, s in enc_shapes:
    print(f"  stage {i}: {s}")
print("\nPFCR route 实际尺寸：")
for i, r in enumerate(model._last_pfcr_routes):
    print(f"  pfcr_modules[{i}] route: {tuple(r.shape)}   (PC 分辨率 = route 分辨率)")
print("\nPC map 尺寸：")
for i, p in enumerate(model._last_pc_maps):
    print(f"  [{i}] {None if p is None else tuple(p.shape)}")
print(f"\n2^20 = {1 << 20};  128^3 = {128**3};  64^3 = {64**3}")
print(f"128^3 > 2^20 ? {128**3 > (1 << 20)}  → 若 route 仍为 128^3 则降采样未触发")
