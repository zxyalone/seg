"""
PhASER auxiliary loss functions.

1. boundary_dice_loss: Dice loss restricted to morphological boundary voxels
   of the ground truth. Encourages precise boundary delineation.

2. spectral_consistency_loss: Self-supervised regulariser that enforces
   high-frequency consistency between encoder skip features and decoder
   features at matching resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from scipy import ndimage as ndi


def morphological_boundary(target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Extract boundary voxels from a binary label map via morphological
    gradient: boundary = dilation(target) - erosion(target).

    Implemented with max-pooling / -max-pooling for efficiency.

    Args:
        target: [B, 1, D, H, W] or [B, D, H, W] binary (0/1)
        kernel_size: size of the structuring element

    Returns:
        boundary: [B, 1, D, H, W] float, 1 at boundary, 0 elsewhere
    """
    # Ensure contiguous 4D [B, 1, spatial...] or 5D [B, 1, D, H, W]
    if target.dim() == 3:
        # 2D target [B, H, W] → [B, 1, H, W]
        target = target.unsqueeze(1).float()
    elif target.dim() == 4 and target.shape[1] == 1:
        # Already [B, 1, spatial...] (from 2D unsqueeze or 3D single-channel)
        target = target.float()
    elif target.dim() == 4:
        # 3D target [B, D, H, W] without channel → [B, 1, D, H, W]
        target = target.unsqueeze(1).float()
    elif target.dim() == 5:
        if target.size(1) != 1:
            target = target.mean(dim=1, keepdim=True).float()
        else:
            target = target.float()
    else:
        raise ValueError(f"Unsupported target dims: {target.dim()}")

    # Detect actual spatial dims from shape beyond B and C
    ndim = target.dim()
    if ndim == 4:
        # [B, 1, H, W] for 2D or [B, 1, D, H, W] -- look at shape[-3]
        if target.shape[-2] > 1 and target.shape[-1] > 1:
            spatial_ndim = 2 if target.shape[-3] == 1 else 3
        else:
            spatial_ndim = 2
    else:
        spatial_ndim = 3

    if spatial_ndim == 3:
        max_pool = F.max_pool3d
    else:
        max_pool = F.max_pool2d

    pad_size = kernel_size // 2
    if spatial_ndim == 3:
        target_padded = F.pad(target, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        dilated = max_pool(target_padded, kernel_size=kernel_size, stride=1, padding=0)
        eroded = -max_pool(-target_padded, kernel_size=kernel_size, stride=1, padding=0)
    else:
        k = kernel_size if isinstance(kernel_size, int) else kernel_size
        target_padded = F.pad(target, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        dilated = max_pool(target_padded, kernel_size=k, stride=1, padding=0)
        eroded = -max_pool(-target_padded, kernel_size=k, stride=1, padding=0)

    boundary = (dilated - eroded).clamp(min=0.0, max=1.0)
    return boundary


def boundary_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    include_background: bool = False,
    smooth: float = 1e-5,
    ignore_label: Optional[int] = None,
    tumor_label: Optional[int] = None,
) -> torch.Tensor:
    """
    Dice loss computed only on boundary voxels of the ground truth.

    Args:
        pred:  [B, C, D, H, W]  logits (no softmax applied)
        target: [B, D, H, W] (long)  or  [B, 1, D, H, W] (float one-hot channel)
        include_background: whether to include background in Dice
        smooth: Laplace smoothing
        ignore_label: label value to ignore (e.g. -1)
        tumor_label: if set, extract boundary only from this class
                     (e.g., 2 for tumor-only boundary).
                     If None, use all foreground (liver + tumor).

    Returns:
        scalar loss
    """
    spatial_dims = tuple(range(2, pred.dim()))

    # Softmax prediction
    pred_soft = torch.softmax(pred, dim=1)

    # --- Normalize target to [B, 1, D, H, W] binary foreground mask ---
    # Target from nnUNet can be [B, D, H, W] (integer labels, 4D for 3D)
    # [B, H, W] (integer labels, 3D for 2D), or [B, 1, D, H, W] (5D)
    if target.dim() == 3:
        target_int = target  # [B, H, W] integer labels (2D)
    elif target.dim() == 4:
        target_int = target  # [B, D, H, W] integer labels (3D)
    elif target.dim() == 5 and target.shape[1] == 1:
        target_int = target.squeeze(1)  # [B, D, H, W] integer labels
    elif target.dim() == pred.dim():
        target_int = target.argmax(dim=1)  # convert one-hot to int labels
    else:
        raise ValueError(f"Unsupported target shape: {target.shape}")

    # Build one-hot for Dice computation
    target_oh = torch.zeros_like(pred_soft)
    if ignore_label is not None:
        target_safe = target_int.clone()
        target_safe[target_safe == ignore_label] = 0
    else:
        target_safe = target_int
    target_oh.scatter_(1, target_safe.unsqueeze(1).long(), 1)

    # Foreground mask (for boundary extraction)
    if tumor_label is not None:
        # Tumor-specific boundary
        any_fg = (target_int == tumor_label).float().unsqueeze(1)
    else:
        # All foreground (liver + tumor)
        any_fg = (target_int != 0).float().unsqueeze(1)
    if ignore_label is not None:
        any_fg = any_fg * (target_int != ignore_label).float().unsqueeze(1)

    # Ignore label mask
    if ignore_label is not None:
        mask = (target_int != ignore_label).float().unsqueeze(1)
    else:
        mask = 1.0

    # Boundary mask from foreground
    bdry = morphological_boundary(any_fg, kernel_size=3)  # [B, 1, D, H, W]

    # Combined weight: only boundary + non-ignore voxels
    weight = bdry * mask

    # Dice per class
    start_idx = 0 if include_background else 1
    dice_loss = 0.0
    for c in range(start_idx, pred.shape[1]):
        p_c = pred_soft[:, c:c + 1]
        t_c = target_oh[:, c:c + 1]

        intersection = (p_c * t_c * weight).sum(dim=spatial_dims)
        denom = ((p_c * weight).sum(dim=spatial_dims) + (t_c * weight).sum(dim=spatial_dims))

        dice_c = (2.0 * intersection + smooth) / (denom + smooth)
        dice_loss += (1.0 - dice_c).mean()

    n_classes = pred.shape[1] - (0 if include_background else 1)
    return dice_loss / max(n_classes, 1)


def spectral_consistency_loss(
    enc_skips: list,
    dec_features: list,
    cutoff: float = 0.3,
    max_scales: int = 3,
) -> torch.Tensor:
    """
    Self-supervised regulariser: enforce high-frequency consistency between
    encoder skip features and decoder features at matching resolutions.

    The encoder skip features and decoder features at corresponding scales
    should have similar high-frequency content — this encourages the decoder
    to preserve fine details that might otherwise be lost.

    Only the deepest `max_scales` scales are used for efficiency.

    Args:
        enc_skips: encoder skip features, shallowest→deepest [e0, e1, ..., e_N-1]
        dec_features: decoder features after conv stages, shallowest→deepest
                      [d0, d1, ..., d_N-1] (matching enc_skips ordering)
        cutoff: fraction of Nyquist below which to suppress in MSE
        max_scales: number of deepest scales to compare

    Returns:
        scalar loss
    """
    if len(enc_skips) != len(dec_features):
        raise ValueError(
            f"Mismatched lengths: enc_skips={len(enc_skips)}, "
            f"dec_features={len(dec_features)}"
        )

    # Only use the deepest max_scales (cheapest: small spatial dims).
    # Both lists are shallowest→deepest, so [-n:] picks the deepest n
    # features which pair correctly: enc_skips[-1]↔dec_features[-1] etc.
    n = min(len(enc_skips), max_scales)
    paired_enc = enc_skips[-n:]
    paired_dec = dec_features[-n:]

    loss = 0.0
    for enc, dec in zip(paired_enc, paired_dec):
        # Detect spatial dimensionality (4D for 2D, 5D for 3D)
        spatial_ndim = enc.dim() - 2  # 2 or 3

        # Interpolate decoder feature to encoder skip spatial size
        if enc.shape[2:] != dec.shape[2:]:
            mode = 'trilinear' if spatial_ndim == 3 else 'bilinear'
            dec = F.interpolate(dec, size=enc.shape[2:],
                                mode=mode, align_corners=False)

        # Per-channel mean spectra (average over channels)
        enc_mean = enc.float().mean(dim=1, keepdim=True)
        dec_mean = dec.float().mean(dim=1, keepdim=True)

        # RFFT magnitude (operate on spatial dims only)
        if spatial_ndim == 3:
            rfft_dims = (-3, -2, -1)
            mean_dims = (-3, -2, -1)
        else:
            rfft_dims = (-2, -1)
            mean_dims = (-2, -1)

        M_enc = torch.abs(torch.fft.rfftn(enc_mean, dim=rfft_dims))
        M_dec = torch.abs(torch.fft.rfftn(dec_mean, dim=rfft_dims))

        # Normalize by mean magnitude → scale-invariant
        eps = 1e-8
        M_enc = M_enc / (M_enc.mean(dim=mean_dims, keepdim=True) + eps)
        M_dec = M_dec / (M_dec.mean(dim=mean_dims, keepdim=True) + eps)

        # High-pass mask
        high_pass = _make_high_pass_mask(
            M_enc.shape[-spatial_ndim:], cutoff, ndim=spatial_ndim
        ).to(M_enc.device).to(M_enc.dtype)

        diff = (M_enc - M_dec) * high_pass
        loss += (diff ** 2).mean()

    return loss / max(n, 1)


def _make_high_pass_mask(shape, cutoff: float, ndim: int = 3) -> torch.Tensor:
    """Create high-pass mask for RFFT shape. Supports 2D (H, W_last) and 3D (D, H, W_last)."""
    if ndim == 2:
        H, W = shape
        freqs_h = (torch.fft.fftfreq(H, d=1.0).abs() /
                   torch.tensor(0.5 if H > 1 else 1.0))
        freqs_w = (torch.fft.rfftfreq(2 * W - 2 if 2 * W - 2 > 0 else 2, d=1.0).abs() /
                   torch.tensor(0.5 if W > 1 else 1.0))
        if freqs_w.shape[0] != W:
            freqs_w = torch.fft.rfftfreq(2 * (W - 1), d=1.0).abs() / 0.5
        grid = (
            freqs_h.view(-1, 1) ** 2 +
            freqs_w.view(1, -1) ** 2
        ).sqrt()
    else:
        D, H, W = shape
        freqs_d = (torch.fft.fftfreq(D, d=1.0).abs() /
                   torch.tensor(0.5 if D > 1 else 1.0))
        freqs_h = (torch.fft.fftfreq(H, d=1.0).abs() /
                   torch.tensor(0.5 if H > 1 else 1.0))
        freqs_w = (torch.fft.rfftfreq(2 * W - 2 if 2 * W - 2 > 0 else 2, d=1.0).abs() /
                   torch.tensor(0.5 if W > 1 else 1.0))
        if freqs_w.shape[0] != W:
            freqs_w = torch.fft.rfftfreq(2 * (W - 1), d=1.0).abs() / 0.5
        grid = (
            freqs_d.view(-1, 1, 1) ** 2 +
            freqs_h.view(1, -1, 1) ** 2 +
            freqs_w.view(1, 1, -1) ** 2
        ).sqrt()
    return (grid > cutoff).float().unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Phase-congruency boundary supervision
# ---------------------------------------------------------------------------

def _dilate(mask: torch.Tensor, k: int) -> torch.Tensor:
    """Morphological dilation via max-pool. mask: [B, 1, *spatial]."""
    mp = F.max_pool3d if mask.dim() == 5 else F.max_pool2d
    return mp(mask, kernel_size=k, stride=1, padding=k // 2)


def phase_frequency_budget_loss(routes: list, target: torch.Tensor = None,
                                tumor_label: int = 2,
                                background_budget=(0.70, 0.22, 0.08),
                                lesion_budget=(0.20, 0.40, 0.40)) -> torch.Tensor:
    """Regularize PFCR low/mid/high routing budgets by semantic region."""
    if not routes:
        return torch.tensor(0.0)
    total = routes[0].new_tensor(0.0)
    for route in routes:
        spatial = tuple(range(2, route.dim()))
        bg_budget = route.new_tensor(background_budget).view(1, 3, *([1] * len(spatial)))
        if target is None:
            total = total + (route.mean(dim=spatial, keepdim=True) - bg_budget).abs().mean()
            continue
        mask = (target == tumor_label).float()
        if mask.dim() == route.dim() - 1:
            mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=route.shape[2:], mode='nearest')
        lesion_mass = mask.sum(dim=spatial, keepdim=True).clamp_min(1.0)
        background_mass = (1.0 - mask).sum(dim=spatial, keepdim=True).clamp_min(1.0)
        lesion_mean = (route * mask).sum(dim=spatial, keepdim=True) / lesion_mass
        background_mean = (route * (1.0 - mask)).sum(dim=spatial, keepdim=True) / background_mass
        lesion_budget_tensor = route.new_tensor(lesion_budget).view(1, 3, *([1] * len(spatial)))
        # Samples without tumor have no lesion mean; without this indicator they
        # would add a constant |0 - budget| with zero gradient to the logged loss.
        has_lesion = (mask.sum(dim=spatial, keepdim=True) > 0).to(route.dtype)
        total = total + ((lesion_mean - lesion_budget_tensor).abs() * has_lesion).mean()
        total = total + (background_mean - bg_budget).abs().mean()
    return total / len(routes)


def pc_boundary_ranking_loss(pc_maps: list,
                             target: torch.Tensor,
                             tumor_label: int = 2,
                             margin: float = 0.2,
                             outer_k: int = 7) -> torch.Tensor:
    """
    Margin-ranking supervision for PhaseCongruencySkipFusion's PC maps.

    Why this exists
    ---------------
    The PC map feeds a gate of the form  out = x·(1 + tanh(γ)·σ(10·tanh(α)·(PC−τ))).
    Every gradient reaching the log-Gabor filters is therefore scaled by tanh(γ),
    which starts at 0 (exact identity, protecting the VoCo weights). Measured
    consequence without this loss: over 1000 SGD steps γ only reaches 0.13 and
    the centre frequencies move by 4e-4 — i.e. "learnable filters" are learnable
    in name only for the first several epochs. This loss gives the filters a
    gradient path that bypasses γ entirely.

    Why ranking rather than a dense binary target
    ---------------------------------------------
    Phase congruency is a *generic* structure detector: it fires on vessels, the
    liver capsule and every other genuine edge, not only on lesion margins.
    Regressing PC onto a binary lesion-boundary mask would penalise all of those
    correct responses and destroy the operator's meaning. What actually matters
    for segmentation is the *contrast* between lesion margin and its
    surroundings, so that is what is optimised directly:

        L = mean_b  relu( margin − ( mean(PC | boundary) − mean(PC | non-boundary) ) )

    positives = morphological boundary shell of the tumour
    negatives = eroded tumour interior  ∪  parenchyma ring just outside it

    The loss is zero once the boundary response already exceeds its surroundings
    by `margin`, so it shapes the filters without fighting them afterwards.

    Args:
        pc_maps: list of [B, 1, *spatial] PC maps, one per skip (may contain None)
        target:  [B, *spatial] integer labels, or [B, 1, *spatial]
        tumor_label: label whose boundary supplies the positives
        margin: required boundary-minus-surround PC gap
        outer_k: dilation kernel defining the outside parenchyma ring

    Returns:
        scalar loss (0.0 if no sample in the batch contains the tumour class)
    """
    maps = [p for p in pc_maps if p is not None]
    if not maps:
        return torch.zeros((), device=target.device, dtype=torch.float32)

    if target.dim() == 5 and target.shape[1] == 1:
        target = target.squeeze(1)
    tumor = (target.long() == tumor_label).float().unsqueeze(1)  # [B,1,*sp]

    loss = maps[0].new_tensor(0.0)
    n_terms = 0

    for pc in maps:
        sp = pc.shape[2:]
        t = tumor if tumor.shape[2:] == sp else F.interpolate(
            tumor, size=sp, mode='nearest')
        if t.sum() == 0:
            continue

        bdry = morphological_boundary(t, kernel_size=3)          # positives
        interior = (t - bdry).clamp(min=0.0)                     # inside lesion
        outer = (_dilate(t, outer_k) - t - bdry).clamp(min=0.0)  # outside ring
        neg = (interior + outer).clamp(0.0, 1.0)

        pcf = pc.float()
        sp_dims = tuple(range(1, pcf.dim()))
        pos_n = bdry.sum(dim=sp_dims)
        neg_n = neg.sum(dim=sp_dims)
        # Only samples that actually have both populations can be ranked.
        valid = (pos_n > 0) & (neg_n > 0)
        if not valid.any():
            continue

        pos_m = (pcf * bdry).sum(dim=sp_dims) / pos_n.clamp(min=1.0)
        neg_m = (pcf * neg).sum(dim=sp_dims) / neg_n.clamp(min=1.0)
        gap = (pos_m - neg_m)[valid]
        loss = loss + F.relu(margin - gap).mean()
        n_terms += 1

    if n_terms == 0:
        return maps[0].new_tensor(0.0)
    return loss / n_terms


# ---------------------------------------------------------------------------
# Lesion-aware importance weighting for small-tumor focus
# ---------------------------------------------------------------------------

def compute_lesion_importance_map(target: torch.Tensor,
                                  tumor_label: int = 2,
                                  small_thresh: int = 100,
                                  medium_thresh: int = 1000) -> torch.Tensor:
    """
    根据肿瘤大小生成逐体素损失权重图。

    小肿瘤 (体积 < small_thresh)  → 权重 5x      — 重点关注
    中等肿瘤 (small~medium)       → 权重 2x      — 适当加强
    大肿瘤 (> medium)              → 权重 1x      — 正常
    肝脏 (非肿瘤)                  → 权重 0.5
    背景                            → 权重 0.1

    Args:
        target: [B, D, H, W] 整数标签
        tumor_label: 肿瘤的标签编号 (默认 2)
        small_thresh: 小肿瘤的体素阈值 (< 该值 = 小肿瘤)
        medium_thresh: 中等肿瘤的体素阈值
    Returns:
        [B, D, H, W] float 权重图
    """
    if target.dim() == 5 and target.shape[1] == 1:
        target = target.squeeze(1)
    if target.dim() == 4:
        B, D, H, W = target.shape
        is_3d = True
    else:
        B, H, W = target.shape
        D = 1
        target = target.unsqueeze(1)  # [B, 1, H, W]
        is_3d = False
    device = target.device
    weight_map = torch.ones_like(target, dtype=torch.float32)

    # 默认权重
    weight_map[target == 0] = 0.1    # 背景
    weight_map[target == 1] = 0.5    # 肝脏
    # 肿瘤区域先设为 1.0，后面按体积调整
    tumor_mask = (target == tumor_label)
    weight_map[tumor_mask] = 1.0

    # 逐 batch 做连通域分析
    for b in range(B):
        mask_3d = tumor_mask[b].cpu().numpy().astype(np.int8)
        if mask_3d.max() == 0:
            continue
        labeled, num = ndi.label(mask_3d)
        for lid in range(1, num + 1):
            voxel_mask = (labeled == lid)
            volume = voxel_mask.sum()
            if volume < small_thresh:
                weight = 5.0
            elif volume < medium_thresh:
                weight = 2.0
            else:
                weight = 1.0
            # 写回
            idx = torch.from_numpy(voxel_mask).to(device)
            weight_map[b][idx] = weight

    if not is_3d:
        weight_map = weight_map.squeeze(1)  # [B, 1, H, W] → [B, H, W]
    return weight_map


# ---------------------------------------------------------------------------
# Gate supervision loss for SG-CFF
# ---------------------------------------------------------------------------

def compute_gate_target(target: torch.Tensor,
                        target_spatial,
                        liver_label: int = 1,
                        tumor_label: int = 2) -> torch.Tensor:
    """
    Generate gate supervision target for each skip scale.

    Gate target values:
      1.0  - stable liver parenchyma
      0.2  - near tumor boundary
      0.4  - inside small tumor
      0.8  - near liver outer boundary
      0.5  - background

    Args:
        target: [B, D, H, W] or [B, H, W] integer label map
        target_spatial: (D, H, W) or (H, W) target spatial size
        liver_label: label for liver (default 1)
        tumor_label: label for tumor (default 2)

    Returns:
        [B, 1, D, H, W] or [B, 1, H, W] float gate target
    """
    if target.dim() == 5 and target.shape[1] == 1:
        target = target.squeeze(1)

    device = target.device
    gate_t = torch.full_like(target, 0.5, dtype=torch.float32)     # background

    # Liver parenchyma
    liver_mask = (target == liver_label)
    gate_t[liver_mask] = 1.0

    # Tumor region
    tumor_mask = (target == tumor_label)
    gate_t[tumor_mask] = 0.4

    # Detect 2D vs 3D based on target dim
    is_3d = (target.dim() == 4)  # [B, D, H, W]
    max_pool = F.max_pool3d if is_3d else F.max_pool2d

    # Liver boundary (dilation - erosion of liver)
    liver_binary = liver_mask.float().unsqueeze(1)  # [B, 1, ...]
    k, p = (5, 2) if is_3d else ((5, 5), (2, 2))
    liver_dilated = max_pool(liver_binary, kernel_size=k, stride=1, padding=p)
    liver_eroded = -max_pool(-liver_binary, kernel_size=k, stride=1, padding=p)
    liver_boundary = (liver_dilated - liver_eroded).squeeze(1) > 0.5
    gate_t[liver_boundary & ~tumor_mask] = 0.8

    # Tumor boundary
    tumor_binary = tumor_mask.float().unsqueeze(1)
    k, p = (3, 1) if is_3d else ((3, 3), (1, 1))
    tumor_dilated = max_pool(tumor_binary, kernel_size=k, stride=1, padding=p)
    tumor_eroded = -max_pool(-tumor_binary, kernel_size=k, stride=1, padding=p)
    tumor_boundary = (tumor_dilated - tumor_eroded).squeeze(1) > 0.5
    gate_t[tumor_boundary] = 0.2

    # Resize to target spatial size
    if gate_t.dim() == 4:
        gate_t = gate_t.unsqueeze(1)
    elif gate_t.dim() == 3:
        gate_t = gate_t.unsqueeze(1)
    if gate_t.shape[2:] != target_spatial:
        mode = 'trilinear' if is_3d else 'bilinear'
        gate_t = F.interpolate(gate_t, size=target_spatial, mode=mode, align_corners=False)

    return gate_t


def gate_supervision_loss(gates: list,
                          target: torch.Tensor,
                          valid_mask: torch.Tensor = None) -> torch.Tensor:
    """
    BCE loss for supervised gate learning.

    Each gate map g_i is compared against a downsampled gate target
    at the corresponding skip resolution.

    Args:
        gates: list of [B, 1, D_i, H_i, W_i] gate maps from CFF modules
        target: [B, D, H, W] integer label map
        valid_mask: optional [B, 1, D, H, W] mask of valid voxels

    Returns:
        scalar loss
    """
    loss = 0.0
    count = 0
    for i, g in enumerate(gates):
        if g is None:
            continue
        gt = compute_gate_target(target, g.shape[2:])
        with torch.amp.autocast(device_type=g.device.type, enabled=False):
            bce = F.binary_cross_entropy(g.float(), gt.float(), reduction='none')
        if valid_mask is not None:
            vm = F.interpolate(valid_mask, size=g.shape[2:], mode='nearest')
            bce = bce * vm
            loss += bce.sum() / vm.sum().clamp(min=1)
        else:
            loss += bce.mean()
        count += 1
    return loss / max(count, 1)


# ---------------------------------------------------------------------------
# Inside-liver penalty (tumor outside liver → penalty)
# ---------------------------------------------------------------------------

def inside_liver_penalty_loss(seg_logits: torch.Tensor,
                              liver_logits: torch.Tensor,
                              tumor_label: int = 2,
                              margin: float = 0.1) -> torch.Tensor:
    """
    Penalize tumor probability outside estimated liver region.

    Args:
        seg_logits: [B, C, D, H, W] multi-class segmentation logits
        liver_logits: [B, 1, D, H, W] liver prior logits
        tumor_label: index of tumor class
        margin: tolerance outside liver before penalty activates

    Returns:
        scalar loss
    """
    tumor_prob = torch.softmax(seg_logits, dim=1)[:, tumor_label:tumor_label + 1]
    liver_prob = torch.sigmoid(liver_logits)
    # Ensure spatial size match
    if liver_prob.shape[2:] != tumor_prob.shape[2:]:
        is_3d = (liver_prob.dim() == 5)
        mode = 'trilinear' if is_3d else 'bilinear'
        liver_prob = F.interpolate(liver_prob, size=tumor_prob.shape[2:],
                                    mode=mode, align_corners=False)
    outside = torch.clamp(1.0 - liver_prob - margin, min=0.0)
    penalty = (tumor_prob * outside).mean()
    return penalty


# ---------------------------------------------------------------------------
# Boundary-weighted Dice loss (GPU-only morphological boundary)
# ---------------------------------------------------------------------------

def boundary_weighted_dice_loss(pred: torch.Tensor,
                                target: torch.Tensor,
                                label: int = 2,
                                smooth: float = 1e-5) -> torch.Tensor:
    """
    Dice loss weighted by morphological boundary of the target class.
    Voxels near the boundary get higher weight, providing focused
    boundary supervision without the [0,1] vs [-1,1] mismatch.

    Uses GPU-based morphological boundary (no CPU scipy).

    Args:
        pred: [B, C, D, H, W] logits
        target: [B, D, H, W] integer labels or [B, 1, D, H, W]
        label: target class for boundary weighting
        smooth: Laplace smoothing

    Returns:
        scalar loss
    """
    pred_soft = torch.softmax(pred, dim=1)
    pred_c = pred_soft[:, label:label + 1]

    if target.dim() == 5 and target.shape[1] == 1:
        target_int = target.squeeze(1)
    else:
        target_int = target

    target_binary = (target_int == label).float().unsqueeze(1)

    # Morphological boundary on GPU (max-pool gradient)
    bdry = morphological_boundary(target_binary, kernel_size=3)

    # Boundary weight: 3x at boundary, 1x elsewhere
    weight = 1.0 + 2.0 * bdry

    dims = tuple(range(2, pred.dim()))
    intersection = (pred_c * target_binary * weight).sum(dim=dims)
    denom = ((pred_c * weight).sum(dim=dims) + (target_binary * weight).sum(dim=dims))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return (1.0 - dice).mean()


def lesion_weighted_loss(pred: torch.Tensor,
                         target: torch.Tensor,
                         weight_map: torch.Tensor,
                         ignore_label: Optional[int] = None) -> torch.Tensor:
    """
    用重要性权重加权的 Dice + CE 损失。

    只在肿瘤区域（tumor_label=2）计算加权损失，
    肝脏和背景区域用标准损失。

    Args:
        pred:  [B, C, D, H, W] logits
        target: [B, D, H, W] 整数标签
        weight_map: [B, D, H, W] float 权重图 (来自 compute_lesion_importance_map)
        ignore_label: 忽略标签
    Returns:
        标量损失
    """
    # Normalize target to [B, D, H, W] Long
    if target.dim() == 5 and target.shape[1] == 1:
        target = target.squeeze(1)
    target = target.long()

    prob = torch.softmax(pred, dim=1)

    target_oh = torch.zeros_like(prob)
    target_safe = target.clone()
    if ignore_label is not None:
        target_safe[target_safe == ignore_label] = 0
    target_oh.scatter_(1, target_safe.unsqueeze(1).long(), 1)

    # 权重广播到通道维
    expand_shape = (-1, pred.shape[1]) + (-1,) * (pred.dim() - 2)
    w = weight_map.unsqueeze(1).expand(*expand_shape)

    # 加权 Dice
    dims = tuple(range(2, pred.dim()))
    tp = (prob * target_oh * w).sum(dim=dims)
    fp = (prob * (1 - target_oh) * w).sum(dim=dims)
    fn = ((1 - prob) * target_oh * w).sum(dim=dims)

    smooth = 1e-5
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    dice_loss = 1 - dice.mean()

    # 加权 CE
    ce = F.cross_entropy(pred, target_safe, reduction='none') * weight_map
    ce_loss = ce.sum() / weight_map.sum().clamp(min=1)

    return dice_loss + ce_loss
