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
    LOW_CNR_QUANTILES, _gauss_blur, band_beta, component_dicece_loss, contrast_consistency_loss,
    lesion_contrast_augment, relative_mse_consistency_loss, sample_target_cnr, voronoi_partition)

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


BAND = (2.0, 2.0 / 0.779172, 2.0 / 0.779172)   # sigma_b = 2 mm on the plan grid, in voxels


def test_bdca():
    """B-DCA (B13, docs/训练侧频域计划.md §3.1)."""
    x, lab, big, small = volume()
    # 1 sigma_b -> 0 reduces to the plain operator (same target draws)
    plain, _ = lesion_contrast_augment(x, lab, p=1.0, target_cnr=True, generator=torch.Generator().manual_seed(3))
    tiny, _ = lesion_contrast_augment(x, lab, p=1.0, target_cnr=True, band_sigma=(1e-4,) * 3,
                                      generator=torch.Generator().manual_seed(3))
    assert (plain - tiny).abs().max() <= 1e-5, (plain - tiny).abs().max()
    # 2 the field stays in the allowed region (liver); nothing changes outside the liver
    rep_ = []
    out, _ = lesion_contrast_augment(x, lab, p=1.0, target_cnr=True, band_sigma=BAND,
                                     generator=torch.Generator().manual_seed(3), report=rep_)
    d = (out - x)[0, 0]
    assert d[lab[0, 0] == 0].abs().max() == 0 and d[big].abs().max() > 0
    # 3 texture: the change inside the lesion is smooth (no high-frequency part)
    core = (F.max_pool3d((~big).float()[None, None], 5, 1, 2)[0, 0] == 0)
    hp = (d - _gauss_blur(d[None, None], (1.0,) * 3)[0, 0])[core]
    assert hp.abs().max() <= 0.05 * d[core].abs().mean(), (hp.abs().max(), d[core].abs().mean())
    # 5 small lesions are attenuated less than with the plain field (mean contrast kept higher)
    xs, labs, _, sm = volume(seed=2)
    po, _ = lesion_contrast_augment(xs, labs, p=1.0, target_cnr=True, fixed_target=0.4)
    bo, _ = lesion_contrast_augment(xs, labs, p=1.0, target_cnr=True, fixed_target=0.4, band_sigma=BAND)
    cp, cb = ring_cnr(po[0, 0], sm, labs[0, 0]), ring_cnr(bo[0, 0], sm, labs[0, 0])
    assert cb > cp, (cb, cp)
    print(f"  small lesion (r = 4 voxels) at target 0.4: plain DCA CNR {cp:.2f}, B-DCA {cb:.2f}")
    # 6 calibration helper: no attenuation when the low-passed field is not larger in the lesion; cap at 1
    assert band_beta(0.5, 1.0, 0.1, 0.2, 0.3) is None and band_beta(0.5, 1.0, 0.1, 0.3, 0.3) is None
    assert band_beta(0.1, 1.0, 0.0, 0.3, 0.0) == 1.0
    assert abs(band_beta(0.5, 1.0, 0.0, 0.9, 0.0) - 0.5 / 0.9) < 1e-12
    assert rep_ and all(r["beta"] is not None for r in rep_)
    print("✓ B-DCA band-limited contrast attenuation")


def test_bdca_calibration():
    """§3.1 calibration test of B-DCA (target within 5% on a large synthetic lesion).
    KNOWN TO FAIL with the registered design (docs/训练侧频域计划.md §9, 2026-10-16): the
    low-passed field also lifts the liver ring, the calibrated amplitude exceeds the cap of 1
    and high-CNR lesions stay above the target. Not part of the passing suite; run on its own."""
    g = torch.Generator().manual_seed(1)
    lab2 = torch.zeros(SHAPE, dtype=torch.long)
    lab2[2:46, 2:62, 2:62] = 1
    les = ball((24, 32, 32), 13)
    lab2[les] = 2
    x2 = 1.0 + 0.1 * torch.randn(SHAPE, generator=g)
    x2[lab2 == 0] = -1.0
    x2[les] -= 0.5
    c0 = ring_cnr(x2, les, lab2)
    ok = True
    for t in (0.3, 0.6, 1.0):
        o, _ = lesion_contrast_augment(x2[None, None], lab2[None, None], p=1.0, target_cnr=True,
                                       band_sigma=BAND, fixed_target=t)
        c1 = ring_cnr(o[0, 0], les, lab2)
        ok &= abs(c1 - t) <= 0.05 * t
        print(f"  B-DCA large lesion (r = 13 voxels): CNR {c0:.2f} -> {c1:.3f} (target {t})")
    print("✓ B-DCA calibration" if ok else "✗ B-DCA calibration (known failure, plan §9)")
    return ok


def test_ecfc():
    """E-CFC (E13, docs/训练侧频域计划.md §3.1)."""
    g = torch.Generator().manual_seed(0)
    t = [torch.randn((1, 8, 12, 12, 12), generator=g), torch.randn((1, 6, 6, 6, 6), generator=g)]
    raw = [torch.randn(f.shape, generator=g) for f in t]
    # rescale the student to the teacher's norm voxel by voxel
    s = [r * (f.norm(dim=1, keepdim=True) / r.norm(dim=1, keepdim=True)) for r, f in zip(raw, t)]
    region = torch.zeros((1, 1, 12, 12, 12))
    region[..., 2:9, 2:9, 2:9] = 1
    rel = relative_mse_consistency_loss(s, t, region, 2)
    cos = contrast_consistency_loss(s, t, region, 2)
    assert abs(0.5 * rel.item() - cos.item()) < 1e-6, (0.5 * rel.item(), cos.item())
    # magnitude sensitivity: cosine ignores a rescaled student, relative MSE does not
    s2 = [2.0 * f for f in t]
    assert contrast_consistency_loss(s2, t, region, 2).item() < 1e-6
    assert abs(relative_mse_consistency_loss(s2, t, region, 2).item() - 1.0) < 1e-5
    # no gradient outside the region
    sg = [f.clone().requires_grad_(True) for f in raw]
    relative_mse_consistency_loss(sg, t, region, 1).backward()
    gmap = sg[0].grad.abs().sum(1)[0]
    assert gmap[region[0, 0] == 0].max() == 0 and gmap[region[0, 0] > 0].max() > 0
    # EMA teacher: copy at start, no gradients, m-weighted update
    from types import SimpleNamespace
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_phaser import nnUNetTrainer_phaser as Tr
    net = torch.nn.Sequential(torch.nn.Conv3d(1, 4, 3), torch.nn.InstanceNorm3d(4, affine=True))
    ns = SimpleNamespace(network=net, is_ddp=False, lca_ema_m=0.9, _ema_teacher=None,
                         print_to_log_file=lambda *a, **k: None)
    ns._student_module = lambda: Tr._student_module(ns)
    Tr._init_ema_teacher(ns)
    tp = list(ns._ema_teacher.parameters())
    assert all(torch.equal(a, b) for a, b in zip(tp, net.parameters())) and not any(p.requires_grad for p in tp)
    old = [p.detach().clone() for p in tp]
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    Tr._ema_update(ns)
    for o, p, q in zip(old, tp, net.parameters()):
        assert torch.allclose(p, 0.9 * o + 0.1 * q, atol=1e-6)
    print("✓ E-CFC relative MSE and EMA teacher")


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
    test_bdca()
    test_ecfc()
    test_speed()
    print("全部通过")
    # Known failure of the registered B-DCA design, reported separately (plan §9):
    test_bdca_calibration()
