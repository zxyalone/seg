#!/usr/bin/env python3
"""B-DCA design variants on synthetic spherical lesions (docs/训练侧频域计划.md §9, 2026-10-16).
R = registered (low-passed field, mean-contrast calibration, beta <= 1); C = field confined to the lesion
(min(G*w, w)), beta = 1 - alpha; D = confined field, mean-contrast calibration, beta <= 1 / max(f).
Columns: achieved CNR at target 0.6 (plain DCA, R, C, D) and boundary-shell gradient ratio vs plain.
Descriptive; motivates why the registered operator is expected to fail the §3.2 check."""
import sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nnUNet"))
from nnunetv2.training.nnUNetTrainer.phaser import lowcnr as L
SP = (1.0, 0.779172, 0.779172); BAND = tuple(2.0 / s for s in SP)
SH = (64, 80, 80)
def ring_mask(m):
    d6 = F.max_pool3d(m.float()[None, None], 13, 1, 6)[0, 0] > 0
    d1 = F.max_pool3d(m.float()[None, None], 3, 1, 1)[0, 0] > 0
    return d6 & ~d1
def cnr(x, m, lab):
    r = x[ring_mask(m) & (lab == 1)]
    return ((x[m].mean() - r.mean()).abs() / r.std(unbiased=False)).item()
def edge_grad(x, m):
    d = F.max_pool3d(m.float()[None, None], 3, 1, 1)[0, 0] > 0
    e = -F.max_pool3d(-m.float()[None, None], 3, 1, 1)[0, 0] > 0
    sh = d & ~e
    g = torch.gradient(x, spacing=SP)
    mag = torch.sqrt(sum(gi ** 2 for gi in g))
    return mag[sh].mean().item()
def make(r_mm, cnr0, seed=0):
    g = torch.Generator().manual_seed(seed)
    z, y, xx = torch.meshgrid(*[torch.arange(n) for n in SH], indexing="ij")
    m = ((z - 32) * SP[0]) ** 2 + ((y - 40) * SP[1]) ** 2 + ((xx - 40) * SP[2]) ** 2 <= r_mm ** 2
    lab = torch.ones(SH, dtype=torch.long); lab[:2] = 0; lab[-2:] = 0; lab[m] = 2
    x = 1.0 + 0.1 * torch.randn(SH, generator=g); x[m] -= 0.1 * cnr0; x[lab == 0] = -1
    return x, lab, m
def apply(x, lab, m, t, mode):
    xb = x[None, None].clone(); mm = m.float()[None, None]
    ring = ring_mask(m)[None, None].float() * (lab == 1).float()[None, None]
    mu_l = (xb * mm).sum() / mm.sum(); mu_r = (xb * ring).sum() / ring.sum()
    sd = (((xb - mu_r) ** 2 * ring).sum() / ring.sum()).sqrt()
    alpha = min(1.0, t / ((mu_l - mu_r).abs() / sd).item())
    allowed = (lab > 0).float()[None, None]
    w = L._soften(mm) * allowed
    if mode == "plain":
        return (xb - (1 - alpha) * (mu_l - mu_r) * w)[0, 0]
    G = L._gauss_blur(L._soften(mm), BAND)
    f = G * allowed if mode == "R" else torch.minimum(G, L._soften(mm)) * allowed
    a = lambda fld, msk: ((fld * msk).sum() / msk.sum()).item()
    if mode == "C":
        beta = 1 - alpha
    else:
        beta = (1 - alpha) * (a(w, mm) - a(w, ring)) / (a(f, mm) - a(f, ring))
        cap = 1.0 if mode == "R" else 1.0 / f[mm > 0].max().item()
        beta = min(cap, beta)
    return (xb - beta * (mu_l - mu_r) * f)[0, 0]
t = 0.6
print("r_mm cnr0 | achieved CNR: plain   R     C     D  | edge grad ratio vs plain: R    C    D")
for cnr0 in (2.0, 5.0):
    for r in (3, 4, 5, 7, 10, 15):
        x, lab, m = make(r, cnr0)
        out = {k: apply(x, lab, m, t, k) for k in ("plain", "R", "C", "D")}
        c = {k: cnr(v, m, lab) for k, v in out.items()}
        e = {k: edge_grad(v, m) for k, v in out.items()}
        print(f"{r:4d} {cnr0:4.1f} | {c['plain']:5.2f} {c['R']:5.2f} {c['C']:5.2f} {c['D']:5.2f} | "
              f"{e['R']/e['plain']:5.2f} {e['C']/e['plain']:5.2f} {e['D']/e['plain']:5.2f}")
