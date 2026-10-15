"""Compute illustrative thumbnails: 2D monogenic PC (Eqs. 3-11, initial params), Haar sub-bands."""
import sys
import numpy as np
from scipy.ndimage import zoom, gaussian_filter, binary_dilation, binary_erosion

d = np.load(sys.argv[1]); out = sys.argv[2]; f = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
ct, gt = d["ct"], d["gt"]


def pc2d(p, lam=(3.0, 6.3, 13.2), sig=0.55, k=2.0):
    H, W = p.shape
    uy = np.fft.fftfreq(H)[:, None]; ux = np.fft.fftfreq(W)[None, :]
    rho = np.sqrt(uy ** 2 + ux ** 2); rho[0, 0] = 1
    P = np.fft.fft2(p)
    vs, A = [], []
    for l in lam:
        G = np.exp(-np.log(rho * l) ** 2 / (2 * np.log(sig) ** 2)); G[0, 0] = 0
        e = np.fft.ifft2(G * P).real
        oy = np.fft.ifft2(1j * uy / rho * G * P).real
        ox = np.fft.ifft2(1j * ux / rho * G * P).real
        v = np.stack([e, oy, ox]); vs.append(v); A.append(np.linalg.norm(v, axis=0))
    E = np.linalg.norm(sum(vs), axis=0); SA = sum(A)
    tau = np.median(A[0]) / np.sqrt(np.log(4))
    rhos = [1 / l for l in lam]; g = sum(r / max(rhos) for r in rhos)
    T = tau * g * (np.sqrt(np.pi / 2) + k * np.sqrt((4 - np.pi) / 2))
    return np.clip(np.maximum(E - T, 0) / (SA + 1e-3 * SA.mean() + 1e-12), 0, 1)


small = zoom(ct, f, order=1)
pc = pc2d(small)
print("PC>0 fraction", (pc > 0).mean().round(3), "max", pc.max().round(3))
# Haar (2D, one level) on the CT
x = small[: small.shape[0] // 2 * 2, : small.shape[1] // 2 * 2]
a, b, c, dd = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
LL = (a + b + c + dd) / 4; LH = (a - b + c - dd) / 4; HL = (a + b - c - dd) / 4; HH = (a - b - c + dd) / 4
les = gt == 2
prior = gaussian_filter(les.astype(float), 6)
prior = prior / prior.max()
# loss regions (in the full-res mask, then crop around lesion)
r = 3
B = binary_dilation(les, iterations=r) & ~binary_erosion(les, iterations=r)
I = binary_erosion(les, iterations=r)
O = binary_dilation(les, iterations=3 * r) & ~(les | B)
np.savez(out, ct=ct, gt=gt, small=small, pc=pc, LL=LL, LH=LH, HL=HL, HH=HH, prior=prior, B=B, I=I, O=O)
