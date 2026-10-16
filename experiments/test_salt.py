#!/usr/bin/env python3
"""Unit tests for SALT (docs/SALT模型设计.md §3.1): spectral scale-space anchors, spectral moment descriptor,
FAF-3D, lesion query decoder, set loss, fusion and the PhaserUNet integration. CPU only.

    PYTHONPATH=nnUNet python experiments/test_salt.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nnUNet"))
from nnunetv2.training.nnUNetTrainer.phaser.freq_fusion3d import FreqAwareFusion3D, local_filter3d  # noqa: E402
from nnunetv2.training.nnUNetTrainer.phaser.lesion_query import (  # noqa: E402
    DEFAULT_RADII_MM, LesionQueryDecoder, fuse_tumour, lesion_instances, lesion_map, lesion_query_loss,
    log_bank_response, sample_points, scale_space_peaks, spectral_anchors, spectral_moment_descriptor)

SPACING = (1.0, 0.7791720032691956, 0.7791720032691956)


def ball_mm(shape, centre, r_mm, spacing=SPACING):
    grids = torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in shape], indexing="ij")
    d2 = sum(((g - c) * s) ** 2 for g, c, s in zip(grids, centre, spacing))
    return d2 <= r_mm ** 2


def test_log_bank_matches_spatial_log():
    """Fourier-domain bank == -s^2 lap(G_s * v) with per-axis sigmas s / spacing (scipy), away from borders."""
    g = torch.Generator().manual_seed(0)
    v = torch.randn((1, 48, 56, 56), generator=g)
    radii = (3.0, 6.0)
    r = log_bank_response(v, SPACING, radii)
    for k, rad in enumerate(radii):
        s = rad / math.sqrt(3)
        # Laplacian in mm: second derivative along each axis (voxel units) divided by spacing^2.
        vv = v[0].numpy().astype(np.float64)
        sig_vox = [s / sp for sp in SPACING]
        lap = sum(ndi.gaussian_filter(vv, sigma=sig_vox, order=[2 if a == b else 0 for b in range(3)],
                                      mode="constant", truncate=6.0) / SPACING[a] ** 2 for a in range(3))
        ref = -s ** 2 * lap
        inner = (slice(12, -12),) * 3
        err = np.abs(r[0, k].numpy()[inner] - ref[inner]).max() / np.abs(ref[inner]).max()
        print(f"  r = {rad} mm: max relative difference to scipy gaussian_laplace {err:.2e}")
        assert err < 2e-2
    print("✓ Fourier-domain scale-normalised LoG bank (anisotropic spacing)")


def test_scale_selection():
    """For a ball of radius r the centre response over scale peaks at s ~ r / sqrt(3)."""
    shape = (64, 80, 80)
    for r_mm in (4.0, 9.0):
        m = ball_mm(shape, (32, 40, 40), r_mm).float()[None]
        radii = tuple(np.geomspace(1.5, 25, 25))
        resp = log_bank_response(m, SPACING, radii)[0, :, 32, 40, 40].numpy()
        best = radii[int(resp.argmax())]
        print(f"  ball r = {r_mm} mm: best bank radius {best:.2f} mm")
        assert abs(best / r_mm - 1) < 0.2
    print("✓ automatic scale selection (sigma = r / sqrt(3))")


def test_spectral_anchors():
    shape = (64, 96, 96)
    centres = [(20, 30, 30), (40, 60, 64), (30, 70, 25)]
    radii = [3.0, 6.0, 12.0]
    sem = torch.zeros(shape)
    for c, r in zip(centres, radii):
        sem[ball_mm(shape, c, r)] = 0.4                      # faint semantic evidence (partially lit)
    g = torch.Generator().manual_seed(1)
    img = 0.8 + 0.1 * torch.randn(shape, generator=g)
    for c, r in zip(centres, radii):
        img[ball_mm(shape, c, r)] -= 0.5                     # hypodense lesions
    fg = torch.zeros(shape, dtype=torch.bool)
    fg[4:60, 6:90, 6:90] = True
    a = spectral_anchors(sem[None], img[None], fg[None], SPACING, n_sem=6, n_img=6)
    v = a["valid"][0]
    sem_c = a["centre"][0, :6][v[:6]]
    sem_r = a["radius"][0, :6][v[:6]]
    for c, r in zip(centres, radii):
        d = ((sem_c - torch.tensor(c, dtype=torch.float32)) * torch.tensor(SPACING)).norm(dim=1)
        j = int(d.argmin())
        print(f"  lesion r = {r} mm: semantic anchor at {d[j]:.2f} mm, radius {float(sem_r[j]):.1f} mm")
        assert d[j] < 1.5 and 0.5 * r <= float(sem_r[j]) <= 1.6 * r
        di = ((a["centre"][0, 6:][v[6:]] - torch.tensor(c, dtype=torch.float32)) * torch.tensor(SPACING)).norm(dim=1)
        assert di.min() < 2.5, "image anchors miss a hypodense lesion"
    assert set(a["source"][0].tolist()) == {0, 1}
    # Peaks outside the foreground are rejected for the image source.
    out = scale_space_peaks(log_bank_response(img[None], SPACING)[0].abs(), DEFAULT_RADII_MM, SPACING, 50,
                            valid=torch.zeros(shape, dtype=torch.bool))
    assert out == []
    print("✓ spectral scale-space anchors: centres and radii of semantic and image blobs")


def test_spectral_moments():
    """Parseval: gradient moments / 4 pi^2 == power-spectrum moments; blob / tube / sheet signatures."""
    shape = (40, 48, 48)
    grids = torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in shape], indexing="ij")
    mm = [(gg - n / 2) * s for gg, n, s in zip(grids, shape, SPACING)]
    sig = (3.0, 4.0, 2.5)
    blob = torch.exp(-0.5 * sum((x / s) ** 2 for x, s in zip(mm, sig)))
    # FFT-based second moments (cycles / mm)
    M = torch.fft.fftn(blob.double())
    f = torch.meshgrid(*[torch.fft.fftfreq(n, d=s, dtype=torch.float64) for n, s in zip(shape, SPACING)], indexing="ij")
    P = M.abs() ** 2
    T = torch.tensor([[float((f[a] * f[b] * P).sum() / P.sum()) for b in range(3)] for a in range(3)])
    lam_fft = torch.linalg.eigvalsh(T).flip(0)
    desc = spectral_moment_descriptor(blob[None, None], SPACING)[0, 0]
    lam_tr = math.exp(float(desc[0]))
    print(f"  Gaussian blob: trace from gradients {lam_tr:.5f}, from FFT {float(lam_fft.sum()):.5f}")
    assert abs(lam_tr / float(lam_fft.sum()) - 1) < 0.05
    # Shapes: isotropic ball, tube along z, sheet normal to z (soft masks).
    def soft(d_mm):
        return torch.sigmoid((2.5 - d_mm) / 0.5)
    ball = soft(torch.sqrt(sum(x ** 2 for x in mm)))
    tube = soft(torch.sqrt(mm[1] ** 2 + mm[2] ** 2))
    sheet = soft(mm[0].abs())
    d = spectral_moment_descriptor(torch.stack([ball, tube, sheet])[None], SPACING)[0]
    for name, row in zip(("ball", "tube", "sheet"), d):
        print(f"  {name}: l2/l1 {float(row[1]):.2f}, l3/l1 {float(row[2]):.2f}")
    assert d[0, 1] > 0.8 and d[0, 2] > 0.8                    # blob: isotropic spectrum
    assert d[1, 1] > 0.8 and d[1, 2] < 0.15                   # tube: energy on a plane
    assert d[2, 1] < 0.15 and d[2, 2] < 0.15                  # sheet: energy on a line
    print("✓ spectral moment descriptor (Parseval; blob / tube / sheet)")


def test_faf():
    g = torch.Generator().manual_seed(2)
    up, skip = torch.randn((2, 8, 10, 12, 12), generator=g), torch.randn((2, 8, 10, 12, 12), generator=g)
    faf = FreqAwareFusion3D(8, 8, compressed=4, k_low=3, k_high=3)
    u2, s2 = faf(up, skip)
    assert torch.equal(u2, up) and torch.equal(s2, skip), "FAF must be the identity at initialisation"
    wl, wh = faf.kernels(up, skip)
    assert torch.allclose(wl.sum(1), torch.ones_like(wl[:, 0]), atol=1e-6) and (wl >= 0).all()
    # local_filter3d == unfold reference
    x = torch.randn((1, 3, 6, 7, 8), generator=g)
    w = torch.softmax(torch.randn((1, 27, 6, 7, 8), generator=g), 1)
    xp = F.pad(x, (1,) * 6, mode="replicate")
    cols = torch.stack([xp[:, :, a:a + 6, b:b + 7, c:c + 8] for a in range(3) for b in range(3) for c in range(3)], 2)
    ref = (cols * w[:, None]).sum(2)
    assert torch.allclose(local_filter3d(x, w, 3), ref, atol=1e-6)
    # Low-pass: with alpha = 1 the energy at the Nyquist frequency of a checkerboard drops.
    with torch.no_grad():
        faf.alpha.fill_(1.0)
    zz, yy, xx = torch.meshgrid(*[torch.arange(n) for n in (10, 12, 12)], indexing="ij")
    cb = ((-1.0) ** (zz + yy + xx)).float()[None, None].repeat(1, 8, 1, 1, 1)
    u3, _ = faf(cb, torch.zeros_like(cb))
    ratio = float(u3.detach()[..., 2:-2, 2:-2, 2:-2].abs().mean() / cb.abs().mean())
    print(f"  checkerboard amplitude after the adaptive low-pass: {ratio:.3f}")
    assert ratio < 0.2
    print("✓ FAF-3D: identity at init, normalised kernels, memory-light local filter, low-pass behaviour")


def _toy_decoder(**kw):
    torch.manual_seed(0)
    return LesionQueryDecoder([16, 8], mask_channels=8, spacing=(2.0, 1.56, 1.56), hidden=24, heads=2, layers=2,
                              ffn=32, mask_dim=8, **kw)


def test_decoder_and_loss():
    B, full = 2, (16, 20, 20)
    lab = torch.zeros((B, 1) + full, dtype=torch.long)
    lab[0, 0, 2:8, 3:10, 3:10] = 2
    lab[0, 0, 10:14, 12:17, 12:17] = 2
    lab[1, 0, 0, 0, 0] = 2                                    # 1 voxel: below min_voxels, not an instance
    inst, counts = lesion_instances(lab, min_voxels=20)
    # Pixel features that carry the lesion identity (as learned decoder features would) plus noise; the
    # coarser memory levels are pooled from them.
    gm = torch.Generator().manual_seed(5)
    onehot = torch.stack([torch.from_numpy(inst[b]) for b in range(B)])
    onehot = F.one_hot(onehot.long(), 3).permute(0, 4, 1, 2, 3).float()
    mask_feat = torch.cat([onehot, 0.3 * torch.randn((B, 5) + full, generator=gm)], 1)
    levels = [F.adaptive_avg_pool3d(mask_feat, (4, 5, 5)).repeat(1, 2, 1, 1, 1),
              F.adaptive_avg_pool3d(mask_feat, (8, 10, 10))]
    assert counts == [2, 0] and inst[0].max() == 2 and inst[1].max() == 0
    idx, lbl = sample_points(inst[0], n_per_instance=64, n_uniform=32, rng=np.random.default_rng(0))
    assert len(idx) == 32 + 2 * 64 and ((lbl == 1).sum() >= 1) and ((lbl == 2).sum() >= 1)
    for kw in (dict(n_sem=0, n_img=0, n_free=6), dict(n_sem=3, n_img=3, n_free=2, shape_descriptor=True)):
        dec = _toy_decoder(**kw)
        anchors = None
        if dec.n_anchor:
            sem = torch.softmax(torch.randn(B, 3, *full), 1)[:, 2]
            anchors = spectral_anchors(sem, torch.randn(B, *full), torch.ones((B,) + full, dtype=torch.bool),
                                       (1.0, 0.78, 0.78), dec.n_sem, dec.n_img)
        out = dec(levels, mask_feat, anchors)
        assert len(out["logits"]) == dec.n_layers + 1 and out["logits"][0].shape == (B, dec.n_queries)
        opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
        losses = []
        for _ in range(100):
            out = dec(levels, mask_feat, anchors)
            loss = lesion_query_loss(out, inst, counts, rng=np.random.default_rng(1), n_per_instance=64,
                                     n_uniform=64)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        p = lesion_map(dec(levels, mask_feat, anchors))
        inside = float(p[0][torch.from_numpy(inst[0] > 0)].mean())
        outside = float(p[0][torch.from_numpy(inst[0] == 0)].mean())
        print(f"  {kw}: set loss {losses[0]:.2f} -> {losses[-1]:.2f}; lesion map inside {inside:.2f}, "
              f"outside {outside:.2f}")
        assert losses[-1] < 0.5 * losses[0] and inside > 0.5 > outside
    print("✓ lesion query decoder: shapes, Hungarian set loss, overfits two lesions")


def test_fusion():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn((2, 3, 4, 5, 6), generator=g)
    P = torch.rand((2, 4, 5, 6), generator=g)
    for w in (0.0, 0.5, 1.0):
        p = torch.softmax(fuse_tumour(logits, P, w), 1)
        assert torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-4) and (p >= 0).all()
    ref = torch.softmax(logits, 1)
    assert torch.allclose(torch.softmax(fuse_tumour(logits, P, 1.0), 1), ref, atol=1e-5), "w = 1 is the semantic head"
    p0 = torch.softmax(fuse_tumour(logits, torch.zeros_like(P), 0.0), 1)
    assert float(p0[:, 2].max()) < 1e-4 and torch.allclose(p0[:, 0], ref[:, 0], atol=1e-5)
    print("✓ fusion: valid distributions; w = 1 reproduces the semantic head; background untouched")


def _unet(**kw):
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from nnunetv2.training.nnUNetTrainer.phaser.unet import PhaserUNet
    torch.manual_seed(0)
    base = PlainConvUNet(1, 4, [8, 16, 32, 64], torch.nn.Conv3d, 3, [1, 2, 2, 2], 2, 3, 2, conv_bias=True,
                         norm_op=torch.nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
                         nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=False)
    return PhaserUNet(base, [8, 16, 32, 64], spacing=SPACING, **kw)


def test_unet_integration():
    x = torch.randn(1, 1, 32, 32, 32)
    plain = _unet().eval()
    with torch.no_grad():
        y0 = plain(x)
    # FAF on, gates at zero: same output and the pretrained keys load unchanged.
    faf = _unet(enable_faf=True).eval()
    missing, unexpected = faf.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("faf.") for k in missing)
    with torch.no_grad():
        assert torch.allclose(faf(x), y0, atol=1e-6)
    # LQD on, fusion weight 1: inference equals the semantic head of the same weights.
    for w in (1.0, 0.5):
        q = _unet(enable_lqd=True, lqd_kwargs=dict(n_sem=4, n_img=4, n_free=2, hidden=24, heads=2, layers=2,
                                                   ffn=32, mask_dim=8, shape_descriptor=True),
                  lqd_fuse_w=w).eval()
        q.load_state_dict(plain.state_dict(), strict=False)
        with torch.no_grad():
            y = q(x)
        p = torch.softmax(y, 1)
        assert torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-4)
        if w == 1.0:
            assert torch.allclose(p, torch.softmax(y0, 1), atol=1e-5)
    # Training mode caches the decoder outputs and returns the semantic logits unchanged.
    q.train()
    y_tr = q(x)
    assert q._last_lqd is not None and y_tr.shape == y0.shape
    print("✓ PhaserUNet: FAF identity at init, LQD fusion w = 1 == semantic head, training cache")


def test_speed():
    g = torch.Generator().manual_seed(4)
    sem = torch.rand((2, 128, 128, 128), generator=g) * 0.1
    img = torch.randn((2, 128, 128, 128), generator=g)
    fg = torch.ones((2, 128, 128, 128), dtype=torch.bool)
    t0 = time.time()
    spectral_anchors(sem, img, fg, SPACING, 16, 16)
    print(f"✓ spectral anchors for 2 x 128^3 on CPU: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    test_log_bank_matches_spatial_log()
    test_scale_selection()
    test_spectral_anchors()
    test_spectral_moments()
    test_faf()
    test_decoder_and_loss()
    test_fusion()
    test_unet_integration()
    test_speed()
    print("all SALT tests passed")
