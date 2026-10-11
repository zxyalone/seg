#!/usr/bin/env python3
"""Unit tests for the LMC v0 module (A9 / A10). CPU, synthetic data, about 1 minute.

    PYTHONPATH=nnUNet python experiments/test_lmc.py

Checks
  1  contrast maps: a faint hypodense sphere (CNR about 0.3 per voxel) gives
     clearly negative large-scale z inside the lesion; tumor-free liver stays
     near 0; maps are invariant to x -> 2x + 50; 'zero' and 'shuffle'
     interventions behave as documented.
  2  PhaserUNet with LMC is an exact identity at initialisation (same logits as
     the bare PlainConvUNet), gamma receives gradient at step 0, the mixing
     weights receive gradient once gamma != 0, and the liver head is trained by
     its BCE loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nnUNet"))
from dynamic_network_architectures.architectures.unet import PlainConvUNet  # noqa: E402

from nnunetv2.training.nnUNetTrainer.phaser.modules import LiverReferencedContrast  # noqa: E402
from nnunetv2.training.nnUNetTrainer.phaser.unet import PhaserUNet  # noqa: E402

SPACING = (1.0, 0.78, 0.78)


def synthetic(shape=(64, 96, 96), seed=0):
    g = torch.Generator().manual_seed(seed)
    z, y, x = torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in shape], indexing="ij")
    liver = torch.zeros(shape, dtype=torch.bool)
    liver[4:60, 6:90, 6:90] = True
    img = torch.where(liver, torch.tensor(100.0), torch.tensor(-50.0)) + 12.0 * torch.randn(shape, generator=g)
    c = [n // 2 for n in shape]
    lesion = ((z - c[0]) ** 2 + ((y - c[1]) * 0.78) ** 2 + ((x - c[2]) * 0.78) ** 2) <= 9.0 ** 2
    img[lesion] -= 4.0                                   # CNR about 0.33 per voxel
    return img[None, None], liver[None, None].float(), lesion


def test_maps():
    img, liver, lesion = synthetic()
    lmc = LiverReferencedContrast(SPACING)
    maps = lmc(img, liver)
    assert maps.shape == (1, 10) + tuple(img.shape[2:])
    mu4 = maps[0, 4]                                     # z(mu_c) at sigma_c = 4 mm
    inside, far = mu4[lesion].median().item(), mu4[(liver[0, 0] > 0) & ~lesion].median().item()
    fine_inside = maps[0, 0][lesion].median().item()     # sigma_c = 1 mm
    dog4 = maps[0, 5][lesion].median().item()
    print(f"  z(mu_c) median: 4 mm inside {inside:+.2f}, liver {far:+.2f}; 1 mm inside {fine_inside:+.2f}; "
          f"4 mm centre-surround inside {dog4:+.2f}")
    assert inside < -2.0 and abs(far) < 0.5 and inside < fine_inside and dog4 < -1.0
    maps2 = lmc(2 * img + 50, liver)
    d = (maps2 - maps).abs().max().item()
    print(f"  affine invariance: max |Δ| = {d:.2e}")
    assert d < 1e-2
    lmc.intervene = "zero"
    assert lmc(img, liver).abs().max() == 0
    lmc.intervene = "shuffle"
    sh = lmc(img, liver)
    assert torch.allclose(sh.flatten(2).sort(-1).values, maps.flatten(2).sort(-1).values)
    assert not torch.allclose(sh, maps)
    print("✓ 1 contrast maps")


def small_unet(seed=0):
    torch.manual_seed(seed)
    return PlainConvUNet(
        input_channels=1, n_stages=4, features_per_stage=[8, 16, 32, 32], conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 4, strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[1, 1, 1, 1], num_classes=3, n_conv_per_stage_decoder=[1, 1, 1],
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=True)


def test_network():
    img, liver, _ = synthetic(shape=(32, 48, 48))
    liver = torch.zeros_like(liver); liver[..., 2:30, 4:44, 4:44] = 1
    x = (img - 60) / 60
    base = small_unet()
    ref = [o.detach().clone() for o in base(x)]
    net = PhaserUNet(small_unet(), [8, 16, 32, 32], enable_lmc=True, lmc_spacing=list(SPACING))
    net.train()
    out = net(x)
    diff = max((a - b).abs().max().item() for a, b in zip(out, ref))
    print(f"  identity at init: max |Δ logits| = {diff:.2e}")
    assert diff < 1e-5
    loss = out[0].square().mean() + F.binary_cross_entropy_with_logits(
        net._last_lmc_liver_logits.float(), F.interpolate(liver, size=net._last_lmc_liver_logits.shape[2:]))
    loss.backward()
    g_gamma = [m.gamma.grad.abs().item() for m in net.lmc_gates]
    g_mix = [m.mix.weight.grad.abs().max().item() for m in net.lmc_gates]
    g_liver = max(p.grad.abs().max().item() for p in net.lmc_liver_branch.parameters() if p.grad is not None)
    print(f"  step 0 grads: gamma {['%.1e' % g for g in g_gamma]}, mix max {max(g_mix):.1e}, liver head {g_liver:.1e}")
    assert all(g > 0 for g in g_gamma) and max(g_mix) == 0 and g_liver > 0
    net.zero_grad()
    for m in net.lmc_gates:
        m.gamma.data.fill_(0.5)
    net(x)[0].square().mean().backward()
    g_mix = [m.mix.weight.grad.abs().max().item() for m in net.lmc_gates]
    print(f"  gamma = 0.5: mix grads {['%.1e' % g for g in g_mix]}")
    assert all(g > 0 for g in g_mix)
    n_new = sum(p.numel() for p in net.get_phaser_params())
    print(f"  LMC parameters: {n_new}")
    print("✓ 2 PhaserUNet integration")


if __name__ == "__main__":
    test_maps()
    test_network()
    print("全部通过")
