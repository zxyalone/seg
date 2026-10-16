"""
FAF-3D: frequency-aware feature fusion for one nnU-Net decoder stage (SALT, docs/SALT模型设计.md §2.3).

3D adaptation of FreqFusion (Chen et al., "Frequency-aware Feature Fusion for Dense Image Prediction",
IEEE TPAMI 2024). At every decoder stage nnU-Net concatenates the upsampled deep feature with the encoder
skip. FreqFusion observed that the upsampled deep feature varies quickly inside objects (intra-class
inconsistency: high-frequency content where the class is constant) and that the skip lacks the high
frequencies that place the boundary. It predicts, per voxel and from both inputs,
  * an adaptive low-pass filter (ALPF): k^3 non-negative weights summing to 1 (softmax), i.e. a
    spatially variant low-pass kernel, applied to the upsampled deep feature;
  * an adaptive high-pass filter (AHPF): identity minus another softmax kernel, applied to the skip.

Here both filters enter as gated residuals so that the module is the identity at initialisation and the
pretrained (VoComni) decoder is untouched:
    up'   = up   + alpha * (ALPF(up) - up)
    skip' = skip + beta  * (skip - LPF_h(skip))
alpha, beta are scalars initialised to 0 (no weight decay: names in NO_DECAY_PARAM_NAMES).
The kernel generators start near uniform (weights ~ N(0, 1e-3)), i.e. box filters.

The local filter is computed as a sum over the k^3 shifted views of a replicate-padded input, never
materialising the unfolded [C, k^3, D, H, W] tensor, so memory stays O(|x|).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def local_filter3d(x: torch.Tensor, w: torch.Tensor, k: int) -> torch.Tensor:
    """x [B, C, D, H, W]; w [B, k^3, D, H, W] per-voxel kernel (offsets in z-major order).
    Returns sum_j w_j(v) * x(v + o_j) with replicate padding."""
    r = k // 2
    D, H, W = x.shape[2:]
    xp = F.pad(x, (r, r, r, r, r, r), mode="replicate")
    out = None
    j = 0
    for dz in range(k):
        for dy in range(k):
            for dx in range(k):
                term = w[:, j:j + 1] * xp[:, :, dz:dz + D, dy:dy + H, dx:dx + W]
                out = term if out is None else out + term
                j += 1
    return out


class FreqAwareFusion3D(nn.Module):
    def __init__(self, up_channels: int, skip_channels: int, compressed: int = 32,
                 k_low: int = 3, k_high: int = 3):
        super().__init__()
        assert k_low % 2 == 1 and k_high % 2 == 1
        self.k_low, self.k_high = k_low, k_high
        self.compress_up = nn.Conv3d(up_channels, compressed, 1)
        self.compress_skip = nn.Conv3d(skip_channels, compressed, 1)
        self.low_gen = nn.Conv3d(compressed, k_low ** 3, 3, padding=1)
        self.high_gen = nn.Conv3d(compressed, k_high ** 3, 3, padding=1)
        for m in (self.low_gen, self.high_gen):
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)
        self.alpha = nn.Parameter(torch.zeros(1))   # low-pass gate on the upsampled deep feature
        self.beta = nn.Parameter(torch.zeros(1))    # high-pass gate on the skip

    def kernels(self, up: torch.Tensor, skip: torch.Tensor):
        g = self.compress_up(up) + self.compress_skip(skip)
        w_low = torch.softmax(self.low_gen(g).float(), dim=1).to(up.dtype)
        w_high = torch.softmax(self.high_gen(g).float(), dim=1).to(skip.dtype)
        return w_low, w_high

    def forward(self, up: torch.Tensor, skip: torch.Tensor):
        w_low, w_high = self.kernels(up, skip)
        low = local_filter3d(up, w_low, self.k_low)
        high = skip - local_filter3d(skip, w_high, self.k_high)
        return up + self.alpha * (low - up), skip + self.beta * high
