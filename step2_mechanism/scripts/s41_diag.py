#!/usr/bin/env python3
"""4.1 诊断：无噪声阶梯边缘下 PC 沿传播距离的实际行为。

目的：S3 显示无噪声时 PC_interior ≈ 0.78（远超预期），需要确认
      "内部/外周参考区是否真的没有边缘能量"，否则比值指标的解释会失真。
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import PC_MAX_VOXELS_FULL, banner  # noqa: E402
from nnunetv2.training.nnUNetTrainer.phaser.dim_utils import spatial_gradient_magnitude  # noqa: E402
from nnunetv2.training.nnUNetTrainer.phaser.modules import PhaseCongruency  # noqa: E402

N, BASE, SOFT, R = 160, 60.0, 1.0, 60.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pc_mod = PhaseCongruency(dims=3, n_scales=3, max_voxels=PC_MAX_VOXELS_FULL).to(DEV).eval()


def pc_of(x):
    with torch.no_grad():
        return pc_mod(torch.from_numpy(x)[None, None].to(DEV).float())[0, 0].cpu().numpy()


def grad_of(x):
    with torch.no_grad():
        t = torch.from_numpy(x)[None, None].to(DEV).float()
        return spatial_gradient_magnitude(t, 3)[0, 0].cpu().numpy()


banner("4.1 诊断：无噪声阶梯边缘的 PC 空间剖面")
z = np.arange(N, dtype=np.float32) - (N - 1) / 2
profile = 1.0 / (1.0 + np.exp((np.abs(z) - R) / SOFT))
obj = np.broadcast_to(profile[:, None, None], (N, N, N)).copy()

print(f"{'A':>6} | " + " ".join(f"z={v:>6}" for v in
      [0, 2, 5, 10, 20, 30, 40, 60, 79]))
for A in (20.0, 5.0):
    x = (BASE + A * obj).astype(np.float32)
    pc = pc_of(x)
    gr = grad_of(x)
    # 沿 z 取中心 (y,x) 的剖面
    cy = cx = N // 2
    pcv = [pc[int(v + N // 2), cy, cx] for v in [0, 2, 5, 10, 20, 30, 40, 60, 79]]
    print(f"A={A:5.1f} | PC   " + " ".join(f"{v:7.4f}" for v in pcv))
    grv = [gr[int(v + N // 2), cy, cx] for v in [0, 2, 5, 10, 20, 30, 40, 60, 79]]
    print(f"      | |∇f| " + " ".join(f"{v:7.4f}" for v in grv))

# 关键的 Kovesi 噪声阈值与各scale幅度
print("\n各 scale 的幅度统计（无噪声 A=20）：")
x = (BASE + 20.0 * obj).astype(np.float32)
with torch.no_grad():
    t = torch.from_numpy(x)[None, None].to(DEV).float()
    xf = t.float()
    F = torch.fft.rfftn(xf, dim=(-3, -2, -1))
    log_rho, unit, dc = pc_mod._freq_grid((N, N, N), xf.device)
    rho_s = 0.5 * torch.sigmoid(pc_mod.rho_raw)
    sig_s = 0.05 + 0.9 * torch.sigmoid(pc_mod.sigma_raw)
    G = []
    for s in range(3):
        shp = (-1,) + (1,) * 3
        lnr = log_rho.unsqueeze(0) - torch.log(rho_s[s]).view(*shp)
        den = 2.0 * (torch.log(sig_s[s]) ** 2).view(*shp)
        g = torch.exp(-(lnr ** 2) / den).masked_fill(dc.unsqueeze(0), 0.0)
        G.append(g[0])
    print(f"  learned rho = {rho_s.tolist()}  (λ = {[round(1/v,3) for v in rho_s.tolist()]})")
    print(f"  learned sigma_onf = {sig_s.tolist()}")
    for s, g in enumerate(G):
        K = torch.complex(g, torch.zeros_like(g))
        resp = torch.fft.irfftn(F * K.unsqueeze(0), s=(N, N, N), dim=(-3, -2, -1))
        a = resp.abs()[0, 0]
        print(f"  scale {s}: |response| mean={a.mean():.4e} median={a.median():.4e} "
              f"max={a.max():.4e} min={a.min():.4e}")
    A0 = resp.abs()  # 最后一行的 scale=2
print("\n结论参考：若各 scale 的 median|response| 在远离边缘处不为 0，"
      "则 PC 的分子/分母在该区仍非零，PC_interior 不为 0 属实现本身的性质。")
