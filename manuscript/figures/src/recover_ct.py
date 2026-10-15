"""Recover grayscale CT and GT mask from the overlay panels of fig44 (alpha 0.45 blend)."""
import sys
import numpy as np
from PIL import Image
src, outdir = sys.argv[1], sys.argv[2]
im = np.array(Image.open(src).convert("RGB")).astype(float)
dark = (im[:, :560].sum(2) < 200)
rows = np.where(dark[:, 23:476].sum(1) > 300)[0]
# group consecutive rows -> frame top/bottom lines
groups = np.split(rows, np.where(np.diff(rows) > 5)[0] + 1)
print([ (g[0], g[-1]) for g in groups])
a = 0.45
cols = {1: np.array([0x4C, 0x9E, 0xE8], float), 2: np.array([0xE2, 0x4B, 0x4A], float)}
for k, (y0, y1) in enumerate([(135, 584), (676, 1126), (1218, 1668)]):
    k *= 2
    p = im[y0:y1, 25:474]
    spread = p.max(2) - p.min(2)
    lab = np.zeros(p.shape[:2], np.uint8)
    over = spread > 12
    lab[over & (p[..., 2] > p[..., 0])] = 1
    lab[over & (p[..., 0] >= p[..., 2])] = 2
    g = p.mean(2)
    for l, c in cols.items():
        m = lab == l
        g[m] = ((p[m] - a * c) / (1 - a)).mean(1)
    g = np.clip(g, 0, 255)
    Image.fromarray(g.astype(np.uint8)).save(f"{outdir}/ct_{k//2}.png")
    Image.fromarray(lab * 120).save(f"{outdir}/gt_{k//2}.png")
    np.savez(f"{outdir}/case_{k//2}.npz", ct=g, gt=lab)
    print(k // 2, p.shape, (lab == 1).mean().round(3), (lab == 2).mean().round(3))
