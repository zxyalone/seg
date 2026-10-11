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
    LOW_CNR_QUANTILES, component_dicece_loss, contrast_consistency_loss, lesion_contrast_augment,
    sample_target_cnr, voronoi_partition)

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


def test_boundary_and_nearby_lesions():
    lab = torch.zeros(SHAPE, dtype=torch.long)
    lab[4:44, 6:58, 6:58] = 1
    first = ball((24, 30, 6), 5) & (lab == 1)
    second = ball((24, 30, 18), 5) & (lab == 1)
    lab[first | second] = 2
    g = torch.Generator().manual_seed(0)
    x = 1 + .2 * torch.randn(SHAPE, generator=g)
    x[lab == 0] = -1
    x[first] -= .5
    x[second] -= .4
    torch.manual_seed(1)
    y, applied = lesion_contrast_augment(x[None, None], lab[None, None], p=1.0)
    assert applied
    y = y[0, 0]
    assert torch.equal(y[lab == 0], x[lab == 0])
    for m in (first, second):
        # A uniform shift must preserve every relative voxel intensity, including
        # when another component's smoothing kernel reaches this component.
        delta = (y - x)[m]
        assert (delta.max() - delta.min()).item() < 1e-6
    print("✓ A11 boundary protection and nearby-lesion texture preservation")


def ring_cnr(x, m, lab):
    d6 = F.max_pool3d(m.float()[None, None], 13, 1, 6)[0, 0] > 0
    d1 = F.max_pool3d(m.float()[None, None], 3, 1, 1)[0, 0] > 0
    r = x[d6 & ~d1 & (lab == 1)]
    return ((x[m].mean() - r.mean()).abs() / r.std(unbiased=False)).item()


def test_target_cnr():
    g = torch.Generator().manual_seed(0)
    t = torch.tensor([sample_target_cnr(g) for _ in range(20000)])
    q = torch.tensor(LOW_CNR_QUANTILES)
    emp = torch.quantile(t, torch.tensor([0.1, 0.5, 0.9]))
    print(f"  target CNR draws: range [{t.min():.3f}, {t.max():.3f}], deciles 10/50/90 "
          f"{emp[0]:.3f}/{emp[1]:.3f}/{emp[2]:.3f} vs table {q[2]:.3f}/{q[10]:.3f}/{q[18]:.3f}")
    assert t.min() >= q[0] and t.max() <= q[-1]
    assert (emp - q[[2, 10, 18]]).abs().max() < 0.02
    x, lab, big, small = volume()
    c_big, c_small = ring_cnr(x[0, 0], big, lab[0, 0]), ring_cnr(x[0, 0], small, lab[0, 0])
    after = {"big": [], "small": []}
    for seed in range(8):
        region = torch.zeros_like(lab, dtype=torch.float32)
        y, applied = lesion_contrast_augment(x, lab, p=1.0, target_cnr=True, region_out=region,
                                             generator=torch.Generator().manual_seed(seed))
        assert len(applied) == 1 and 0 < applied[0][1] < 1
        after["big"].append(ring_cnr(y[0, 0], big, lab[0, 0]))
        after["small"].append(ring_cnr(y[0, 0], small, lab[0, 0]))
        assert region[0, 0][big | small].min() == 1 and region.sum() > 2 * (big | small).sum()
        assert region[0, 0][:, :, :3].sum() == 0               # far from both lesions
        assert torch.equal(y[0, 0][lab[0, 0] == 0], x[0, 0][lab[0, 0] == 0])
    for name, c0 in (("big", c_big), ("small", c_small)):
        a = torch.tensor(after[name])
        print(f"  {name}: CNR {c0:.2f} -> {a.min():.2f}..{a.max():.2f} over 8 draws")
        assert c0 > 1.5 and a.max() < 1.2785 + 0.1
    # Lesions already fainter than every target are left untouched.
    faint = x.clone()
    faint[0, 0][big] += 0.5 - 0.004
    faint[0, 0][small] += 0.4 - 0.004
    y, applied = lesion_contrast_augment(faint, lab, p=1.0, target_cnr=True)
    assert applied == [] and torch.equal(y, faint)
    print("✓ A13a target-CNR attenuation")


def test_consistency():
    torch.manual_seed(0)
    region = torch.zeros((2, 1, 16, 16, 16))
    region[0, 0, 4:10, 4:10, 4:10] = 1
    t = [torch.randn(2, 8, 16, 16, 16), torch.randn(2, 16, 8, 8, 8), torch.randn(2, 32, 4, 4, 4)]
    same = [f.clone().requires_grad_() for f in t]
    assert contrast_consistency_loss(same, t, region).abs().item() < 1e-5
    s = [torch.randn_like(f).requires_grad_() for f in t]
    loss = contrast_consistency_loss(s, t, region)
    loss.backward()
    assert 0.5 < loss.item() < 1.5
    assert s[0].grad[1].abs().sum() == 0 and s[0].grad[0].abs().sum() > 0      # only inside region
    assert s[2].grad is None or s[2].grad.abs().sum() == 0                     # third stage unused
    scaled = [(3.0 * f).detach().requires_grad_() for f in t]
    assert contrast_consistency_loss(scaled, t, region).abs().item() < 1e-5     # scale-free
    assert contrast_consistency_loss(s, t, torch.zeros_like(region)).item() == 0.0
    print(f"  random features: loss {loss.item():.3f}; identical / rescaled features: 0")
    print("✓ A13b contrast consistency loss")


def test_consistency_through_network():
    """A13b inside PhaserUNet: teacher pass without gradient, student pass with
    every decoder stage kept in the graph (as the trainer sets it); the loss must
    reach the encoder."""
    import torch.nn as nn
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from nnunetv2.training.nnUNetTrainer.phaser.unet import PhaserUNet
    torch.manual_seed(0)
    base = PlainConvUNet(
        input_channels=1, n_stages=4, features_per_stage=[8, 16, 32, 32], conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 4, strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[1, 1, 1, 1], num_classes=3, n_conv_per_stage_decoder=[1, 1, 1],
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=True)
    net = PhaserUNet(base, [8, 16, 32, 32]).train()
    x, lab, big, small = volume()
    region = torch.zeros_like(lab, dtype=torch.float32)
    y, applied = lesion_contrast_augment(x, lab, p=1.0, target_cnr=True, region_out=region,
                                         generator=torch.Generator().manual_seed(0))
    assert applied
    net._spectral_grad_scales = 10 ** 6
    with torch.no_grad():
        net(x)
    teacher = [f.detach() for f in net._last_dec_features[:2]]
    net(y)
    student = net._last_dec_features[:2]
    assert student[0].shape[2:] == x.shape[2:] and all(f.requires_grad for f in student)
    loss = contrast_consistency_loss(student, teacher, region, 2)
    loss.backward()
    g = net.encoder.stages[0][0].convs[0].conv.weight.grad
    print(f"  PhaserUNet: consistency loss {loss.item():.4f}, encoder stage-0 |grad| {g.abs().max().item():.2e}")
    assert loss.item() > 0 and g.abs().max() > 0
    print("✓ A13b gradient reaches the encoder through the finest decoder features")


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
    test_boundary_and_nearby_lesions()
    test_cc_loss()
    test_target_cnr()
    test_consistency()
    test_consistency_through_network()
    test_speed()
    print("全部通过")
