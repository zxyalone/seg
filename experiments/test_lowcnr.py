#!/usr/bin/env python3
"""Unit tests for A11 (lesion contrast augmentation) and A12 (component-wise DiceCE).

    PYTHONPATH=nnUNet python experiments/test_lowcnr.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nnUNet"))
from nnunetv2.training.nnUNetTrainer.phaser.lowcnr import (  # noqa: E402
    component_dicece_loss, lesion_contrast_augment, voronoi_partition)

SHAPE = (48, 64, 64)


def ball(c, r):
    z, y, x = torch.meshgrid(*[torch.arange(n) for n in SHAPE], indexing="ij")
    return ((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) <= r * r


def volume(seed=0):
    g = torch.Generator().manual_seed(seed)
    lab = torch.zeros(SHAPE, dtype=torch.long)
    lab[4:44, 6:58, 6:58] = 1
    big, small = ball((20, 24, 24), 8), ball((30, 46, 44), 4)
    lab[big] = 2
    lab[small] = 2
    x = 1.0 + 0.2 * torch.randn(SHAPE, generator=g)
    x[lab == 0] = -1.0
    x[big] -= 0.5
    x[small] -= 0.4
    return x[None, None], lab[None, None], big, small


def ring_mean(x, m, lab):
    d6 = F.max_pool3d(m.float()[None, None], 13, 1, 6)[0, 0] > 0
    d1 = F.max_pool3d(m.float()[None, None], 3, 1, 1)[0, 0] > 0
    return x[d6 & ~d1 & (lab == 1)].mean()


def test_augment():
    x, lab, big, small = volume()
    same, applied = lesion_contrast_augment(x, lab, p=0.0)
    assert torch.equal(same, x) and applied == []
    torch.manual_seed(1)
    y, applied = lesion_contrast_augment(x, lab, p=1.0, alpha_min=0.5)
    (b, alpha), = applied
    for name, m in (("big", big), ("small", small)):
        c0 = (x[0, 0][m].mean() - ring_mean(x[0, 0], m, lab[0, 0])).item()
        c1 = (y[0, 0][m].mean() - ring_mean(y[0, 0], m, lab[0, 0])).item()
        sd0, sd1 = x[0, 0][m].std().item(), y[0, 0][m].std().item()
        print(f"  {name}: contrast {c0:+.3f} -> {c1:+.3f} (ratio {c1 / c0:.3f}, alpha {alpha:.3f}); "
              f"texture sd {sd0:.3f} -> {sd1:.3f}")
        assert abs(c1 / c0 - alpha) < 0.06 and abs(sd1 - sd0) < 0.02
    far = lab[0, 0] == 0
    assert torch.equal(y[0, 0][far], x[0, 0][far])
    print("✓ A11 lesion contrast augmentation")


def test_cc_loss():
    _, lab, big, small = volume()
    part, counts = voronoi_partition(lab)
    assert counts == [2] and set(part.unique().tolist()) == {1, 2}
    def logits_for(pred_tumor):
        lg = torch.full((1, 3) + SHAPE, -12.0)
        lg[0, 1] = 0.0
        lg[0, 2][pred_tumor] = 12.0
        return lg.requires_grad_()
    perfect = component_dicece_loss(logits_for(lab[0, 0] == 2), lab)
    miss_small = logits_for(big)
    l_miss = component_dicece_loss(miss_small, lab)
    l_miss.backward()
    pt = torch.softmax(miss_small.detach(), 1)[0, 2]
    gt = (lab[0, 0] == 2).float()
    global_dice_loss = 1 - 2 * (pt * gt).sum() / (pt.sum() + gt.sum())
    print(f"  perfect {perfect.item():.3f}; small lesion missed: component loss {l_miss.item():.3f} "
          f"vs global Dice loss {global_dice_loss.item():.3f}")
    assert perfect < 0.05 and l_miss > 0.5 > global_dice_loss
    assert miss_small.grad[0, 2][small].abs().sum() > 0
    no_tumor = torch.zeros_like(lab)
    assert component_dicece_loss(logits_for(big), no_tumor).item() == 0.0
    print("✓ A12 component-wise DiceCE")


def test_speed():
    lab = torch.zeros((2, 1, 128, 128, 128), dtype=torch.long)
    lab[:, :, 10:118, 10:118, 10:118] = 1
    for c in ((40, 40, 40), (80, 90, 70), (60, 30, 100)):
        z, y, x = torch.meshgrid(*[torch.arange(128)] * 3, indexing="ij")
        lab[:, 0][:, ((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) <= 64] = 2
    x = torch.randn((2, 1, 128, 128, 128))
    t0 = time.time(); lesion_contrast_augment(x, lab, p=1.0); t1 = time.time()
    component_dicece_loss(torch.randn((2, 3, 128, 128, 128)), lab); t2 = time.time()
    print(f"  CPU, 2 x 128^3, 3 lesions: augmentation {t1 - t0:.2f}s, component loss {t2 - t1:.2f}s")


if __name__ == "__main__":
    test_augment()
    test_cc_loss()
    test_speed()
    print("全部通过")
