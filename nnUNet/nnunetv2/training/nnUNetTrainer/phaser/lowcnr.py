"""
Training-time tools aimed at low-CNR liver lesions (docs/方法重设计_可检出性计划.md §5b).

lesion_contrast_augment  (A11)
    Per training sample (probability p) and per tumor component, move the lesion
    intensities towards the mean of its liver ring:
        x' = x - (1 - alpha) * (mu_lesion - mu_ring) * w,   alpha ~ U(alpha_min, 1)
    w = max(G_1 * mask, mask) softens the lesion border (the E12 operator). The
    shift keeps the lesion texture and the ring statistics, so the lesion CNR
    becomes alpha * CNR. The ring is liver (label 1) within 2..6 voxels
    (Chebyshev, by max-pooling) outside the lesion and outside other lesions.
    Works on the normalised network input; nnU-Net's CT normalisation is a
    global affine map, so CNR is unchanged by it.

component_dicece_loss  (A12)
    Component-wise DiceCE in the spirit of CC-DiceCE (Bouteille et al., ISBI
    2026, arXiv 2511.17146): every voxel is assigned to its nearest ground-truth
    tumor component (Voronoi partition, computed at half resolution), and a
    tumor-vs-rest soft Dice + BCE is evaluated inside each region and averaged
    over components, so every lesion weighs the same whatever its size. Added to
    the standard deep-supervised DiceCE; zero for patches without tumor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

_STRUCT26 = ndi.generate_binary_structure(3, 3)


def _components(mask: np.ndarray):
    return ndi.label(mask, structure=_STRUCT26)


def _dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    return F.max_pool3d(m, kernel_size=2 * r + 1, stride=1, padding=r)


def _soften(m: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    r = max(int(np.ceil(3 * sigma)), 1)
    t = torch.arange(-r, r + 1, device=m.device, dtype=m.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    out = m
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = k.numel()
        pad = [0, 0, 0]
        pad[ax] = r
        out = F.conv3d(out, k.view(shape), padding=tuple(pad))
    return torch.maximum(out, m)


@torch.no_grad()
def lesion_contrast_augment(data: torch.Tensor, target: torch.Tensor, p: float = 0.5,
                            alpha_min: float = 0.3, min_voxels: int = 20,
                            ring=(2, 6), generator: torch.Generator = None):
    """data [B, 1, D, H, W] (modified copy returned); target [B, 1, D, H, W] labels.
    Returns (augmented data, list of applied (sample, alpha) pairs)."""
    if data.dim() != 5:
        return data, []
    out = data.clone()
    applied = []
    lab_np = target[:, 0].detach().cpu().numpy()
    for b in range(data.shape[0]):
        if torch.rand(1, generator=generator).item() >= p or not (lab_np[b] == 2).any():
            continue
        alpha = alpha_min + (1.0 - alpha_min) * torch.rand(1, generator=generator).item()
        comp, n = _components(lab_np[b] == 2)
        if n == 0:
            continue
        comp_t = torch.from_numpy(comp).to(data.device)
        x = out[b:b + 1].float()
        margin = ring[1] + 4
        for k, sl in enumerate(ndi.find_objects(comp), start=1):
            if sl is None:
                continue
            box = (slice(None), slice(None)) + tuple(
                slice(max(s.start - margin, 0), min(s.stop + margin, n))
                for s, n in zip(sl, comp.shape))
            c = comp_t[None, None][box]
            m = (c == k).float()
            if m.sum() < min_voxels:
                continue
            others = ((c > 0) & (c != k)).float()
            liver = (target[b:b + 1][box] == 1).float()
            ring_m = (_dilate(m, ring[1]) - _dilate(m, ring[0] - 1)) * liver * (1 - _dilate(others, 1))
            if ring_m.sum() < 10:
                continue
            xb = x[box]
            mu_l = (xb * m).sum() / m.sum()
            mu_r = (xb * ring_m).sum() / ring_m.sum()
            x[box] = xb - (1.0 - alpha) * (mu_l - mu_r) * _soften(m)
        out[b:b + 1] = x.to(out.dtype)
        applied.append((b, alpha))
    return out, applied


def voronoi_partition(target: torch.Tensor, factor: int = 2):
    """Nearest-tumor-component label per voxel ([B, D, H, W] int64; 0 where the
    sample has no tumor), computed on a `factor`-times coarser grid."""
    lab = target[:, 0].detach().cpu().numpy()
    parts, counts = [], []
    for b in range(lab.shape[0]):
        comp, n = _components(lab[b] == 2)
        counts.append(n)
        if n == 0:
            parts.append(np.zeros(lab[b].shape, np.int64))
            continue
        small = comp[::factor, ::factor, ::factor]
        if not (small > 0).any():
            small = ndi.maximum_filter(comp, size=factor)[::factor, ::factor, ::factor]
        _, idx = ndi.distance_transform_edt(small == 0, return_indices=True)
        vor = small[tuple(idx)]
        vor = np.repeat(np.repeat(np.repeat(vor, factor, 0), factor, 1), factor, 2)
        vor = vor[:lab.shape[1], :lab.shape[2], :lab.shape[3]]
        parts.append(np.where(comp > 0, comp, vor).astype(np.int64))
    return torch.from_numpy(np.stack(parts)).to(target.device), counts


def component_dicece_loss(logits: torch.Tensor, target: torch.Tensor, tumor_label: int = 2,
                          factor: int = 2, smooth: float = 1e-5) -> torch.Tensor:
    """logits [B, C, D, H, W] (full resolution), target [B, 1, D, H, W]."""
    part, counts = voronoi_partition(target, factor)
    prob_t = torch.softmax(logits.float(), dim=1)[:, tumor_label]
    logit_t = torch.logsumexp(logits.float()[:, tumor_label:tumor_label + 1], 1) - \
        torch.logsumexp(torch.cat([logits.float()[:, :tumor_label], logits.float()[:, tumor_label + 1:]], 1), 1)
    gt_t = (target[:, 0] == tumor_label).float()
    losses = []
    for b, n in enumerate(counts):
        for k in range(1, n + 1):
            region = part[b] == k
            if not region.any():
                continue
            p, g, z = prob_t[b][region], gt_t[b][region], logit_t[b][region]
            dice = 1 - (2 * (p * g).sum() + smooth) / (p.sum() + g.sum() + smooth)
            bce = F.binary_cross_entropy_with_logits(z, g)
            losses.append(dice + bce)
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()
