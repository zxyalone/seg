"""
Training-time tools aimed at low-CNR liver lesions (docs/方法重设计_可检出性计划.md §5b).

lesion_contrast_augment  (A11)
    Per training sample (probability p) and per tumor component, move the lesion
    intensities towards the mean of its liver ring:
        x' = x - (1 - alpha) * (mu_lesion - mu_ring) * w,   alpha ~ U(alpha_min, 1)
    w = max(G_1 * mask, mask) softens the lesion border (the E12 operator). The
    shift keeps each lesion's texture; soft boundary tails can slightly change
    the ring statistics, so its CNR is approximately alpha * CNR. The ring is liver (label 1) within 2..6 voxels
    (Chebyshev, by max-pooling) outside the lesion and outside other lesions.
    Works on the normalised network input; nnU-Net's CT normalisation is a
    global affine map, so CNR is unchanged by it.

Target-CNR mode  (A13a, plan §6c)
    Same operator, but each lesion gets its own alpha = min(1, t / CNR0), where
    CNR0 = |mu_lesion - mu_ring| / sd_ring is measured in the patch and the
    target t is drawn from the empirical CNR distribution of the low-CNR tier of
    the 518 training lesions (LOW_CNR_QUANTILES; tier bound 1.2785 = frozen
    tertile). Lesions already below t are left unchanged. Every attenuated
    lesion therefore lands in the CNR range where the baseline fails, instead
    of the mild average attenuation of a uniform alpha. No parameter is chosen
    on validation data.

Band-limited mode  (B-DCA, B13; docs/训练侧频域计划.md §1.2)
    The shift field is low-passed before use: f = G_sigma_b * w, restricted to
    the same allowed region (liver, outside other lesions). The lesion's
    low-frequency contrast is attenuated while most of its edge step is kept,
    and small lesions are attenuated less. Calibration keeps the mean-contrast
    reduction of the plain operator: with a_l / a_r the mean of a field over the
    lesion / ring, beta = (1 - alpha) * (a_l(w) - a_r(w)) / (a_l(f) - a_r(f)),
    capped at 1 (a larger beta would invert the contrast); lesions with
    a_l(f) <= a_r(f) are left unchanged. As sigma_b -> 0, f -> w and the operator
    reduces to the plain one.

contrast_consistency_loss  (A13b, plan §6c)
    Feature-level contrast invariance: the decoder features of the attenuated
    patch (with gradient) are pulled towards those of the original patch (same
    network, no gradient) by 1 - cosine similarity, inside the attenuated
    lesions dilated by 6 voxels, at the two finest decoder resolutions.

relative_mse_consistency_loss  (E-CFC, E13)
    Same region and stages as A13b, teacher = EMA of the student (trainer), loss
    ||s - t||^2 / (||t||^2 + eps) per voxel. With ||s|| = ||t|| it equals
    2 (1 - cos), so a weight of 0.5 reproduces the A13b loss; otherwise it also
    pulls the student's feature magnitude towards the (lagging) teacher.

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


# CNR quantiles (0, 5, ..., 100 %) of the 173 training lesions in the low tier
# (CNR < 1.2785) of step2_mechanism/out/s46_lesions.csv (518 training lesions,
# s42 / evaluate.py CNR definition). Used as an inverse CDF (linear
# interpolation) to draw A13 target CNRs.
LOW_CNR_QUANTILES = (0.0223, 0.1985, 0.3378, 0.4206, 0.4786, 0.5271, 0.5882, 0.6487, 0.7108,
                     0.7638, 0.8089, 0.872, 0.9263, 0.971, 1.0284, 1.086, 1.1446, 1.1772,
                     1.2278, 1.2545, 1.2777)


def sample_target_cnr(generator: torch.Generator = None) -> float:
    q = LOW_CNR_QUANTILES
    u = torch.rand(1, generator=generator).item() * (len(q) - 1)
    i = min(int(u), len(q) - 2)
    f = u - i
    return q[i] * (1.0 - f) + q[i + 1] * f


def _components(mask: np.ndarray):
    return ndi.label(mask, structure=_STRUCT26)


def _dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    return F.max_pool3d(m, kernel_size=2 * r + 1, stride=1, padding=r)


def _gauss_blur(m: torch.Tensor, sigmas) -> torch.Tensor:
    """Separable Gaussian blur of [B, C, D, H, W] with one sigma (voxels) per axis,
    truncated at 3 sigma, zero padding."""
    out = m
    for ax, sigma in enumerate(sigmas):
        r = max(int(np.ceil(3 * sigma)), 1)
        t = torch.arange(-r, r + 1, device=m.device, dtype=m.dtype)
        k = torch.exp(-0.5 * (t / max(sigma, 1e-8)) ** 2)
        k = k / k.sum()
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = k.numel()
        pad = [0, 0, 0]
        pad[ax] = r
        out = F.conv3d(out, k.view(shape), padding=tuple(pad))
    return out


def band_beta(alpha: float, a_lw: float, a_rw: float, a_lf: float, a_rf: float):
    """B-DCA calibration: shift amplitude for the low-passed field so that the
    lesion-minus-ring mean contrast drops as much as with the plain field; None
    (no attenuation) when the low-passed field is not larger in the lesion than
    in its ring; capped at 1 so the contrast is never inverted."""
    if a_lf - a_rf <= 0:
        return None
    return min(1.0, (1.0 - alpha) * (a_lw - a_rw) / (a_lf - a_rf))


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
                            ring=(2, 6), generator: torch.Generator = None,
                            target_cnr: bool = False, region_out: torch.Tensor = None,
                            region_dilate: int = 6, band_sigma=None, fixed_target: float = None,
                            report: list = None):
    """data [B, 1, D, H, W] (modified copy returned); target [B, 1, D, H, W] labels.
    Returns (augmented data, list of applied (sample, alpha) pairs).

    target_cnr=False (A11): one alpha ~ U(alpha_min, 1) per sample.
    target_cnr=True (A13a): one alpha = min(1, t / CNR0) per lesion; the pair
    records the mean alpha of the attenuated lesions, and only samples with at
    least one attenuated lesion are listed.
    region_out: optional float tensor shaped like target; set to 1 inside every
    attenuated lesion dilated by `region_dilate` voxels (for A13b).
    band_sigma: per-axis sigma in voxels of the B-DCA low-pass (None = plain DCA).
    fixed_target: use this target CNR instead of sampling one (operator check only).
    report: optional list; one dict per attenuated lesion is appended."""
    if data.dim() != 5:
        return data, []
    out = data.clone()
    applied = []
    lab_np = target[:, 0].detach().cpu().numpy()
    for b in range(data.shape[0]):
        if torch.rand(1, generator=generator).item() >= p or not (lab_np[b] == 2).any():
            continue
        if not target_cnr:
            alpha = alpha_min + (1.0 - alpha_min) * torch.rand(1, generator=generator).item()
        alphas = []
        comp, n = _components(lab_np[b] == 2)
        if n == 0:
            continue
        comp_t = torch.from_numpy(comp).to(data.device)
        x = out[b:b + 1].float()
        margin = ring[1] + 4
        if band_sigma is not None:   # room for the low-pass tails
            margin += int(np.ceil(3 * max(band_sigma)))
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
            cnr0 = t_cnr = None
            if target_cnr:
                sd_r = (((xb - mu_r) ** 2 * ring_m).sum() / ring_m.sum()).sqrt()
                cnr0 = ((mu_l - mu_r).abs() / (sd_r + 1e-8)).item()
                t_cnr = fixed_target if fixed_target is not None else sample_target_cnr(generator)
                alpha = min(1.0, t_cnr / max(cnr0, 1e-8))
                if alpha >= 1.0:
                    continue
            # Preserve tissue outside the liver and every other tumor component.
            # Otherwise smoothing a boundary lesion modifies background, and
            # overlapping tails can change the texture of a neighboring lesion.
            allowed = (target[b:b + 1][box] > 0) & ~(others > 0)
            weight = _soften(m) * allowed
            if band_sigma is None:
                beta = 1.0 - alpha
                x[box] = xb - beta * (mu_l - mu_r) * weight
            else:
                field = _gauss_blur(_soften(m), band_sigma) * allowed
                a_lw, a_rw = (weight * m).sum() / m.sum(), (weight * ring_m).sum() / ring_m.sum()
                a_lf, a_rf = (field * m).sum() / m.sum(), (field * ring_m).sum() / ring_m.sum()
                beta = band_beta(alpha, float(a_lw), float(a_rw), float(a_lf), float(a_rf))
                if beta is None:
                    continue
                x[box] = xb - beta * (mu_l - mu_r) * field
            alphas.append(alpha)
            if report is not None:
                report.append(dict(sample=b, label=k, voxels=int(m.sum().item()), cnr0=cnr0, target=t_cnr,
                                   alpha=alpha, beta=beta))
            if region_out is not None:
                r = region_out[b:b + 1][box]
                region_out[b:b + 1][box] = torch.maximum(r, _dilate(m, region_dilate).to(r.dtype))
        out[b:b + 1] = x.to(out.dtype)
        if not target_cnr:
            applied.append((b, alpha))
        elif alphas:
            applied.append((b, float(np.mean(alphas))))
    return out, applied


def contrast_consistency_loss(student: list, teacher: list, region: torch.Tensor,
                              n_stages: int = 2) -> torch.Tensor:
    """student / teacher: decoder features, finest first ([B, C, ...] each);
    teacher is detached. region: [B, 1, D, H, W] at full resolution (0/1).
    Mean over stages of the region-averaged 1 - cosine similarity."""
    losses = []
    for s_f, t_f in list(zip(student, teacher))[:n_stages]:
        m = region.float()
        if m.shape[2:] != s_f.shape[2:]:
            m = F.adaptive_max_pool3d(m, s_f.shape[2:])
        sel = m[:, 0] > 0
        if not bool(sel.any()):
            continue
        # Only voxels inside Omega enter the loss, so gather them before the
        # float32 cosine (full-resolution float copies do not fit next to a
        # SegResNet on 12 GB). Same value as the masked mean over all voxels.
        s_v = s_f.movedim(1, -1)[sel].float()
        t_v = t_f.detach().movedim(1, -1)[sel].float()
        cos = F.cosine_similarity(s_v, t_v, dim=-1, eps=1e-6)
        losses.append((1.0 - cos).mean())
    if not losses:
        return student[0].new_tensor(0.0, dtype=torch.float32)
    return torch.stack(losses).mean()


def relative_mse_consistency_loss(student: list, teacher: list, region: torch.Tensor,
                                  n_stages: int = 2, eps: float = 1e-6) -> torch.Tensor:
    """E-CFC: mean over stages of the region-averaged ||s - t||^2 / (||t||^2 + eps);
    teacher detached. Same gathering as contrast_consistency_loss."""
    losses = []
    for s_f, t_f in list(zip(student, teacher))[:n_stages]:
        m = region.float()
        if m.shape[2:] != s_f.shape[2:]:
            m = F.adaptive_max_pool3d(m, s_f.shape[2:])
        sel = m[:, 0] > 0
        if not bool(sel.any()):
            continue
        s_v = s_f.movedim(1, -1)[sel].float()
        t_v = t_f.detach().movedim(1, -1)[sel].float()
        rel = ((s_v - t_v) ** 2).sum(-1) / ((t_v ** 2).sum(-1) + eps)
        losses.append(rel.mean())
    if not losses:
        return student[0].new_tensor(0.0, dtype=torch.float32)
    return torch.stack(losses).mean()


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
