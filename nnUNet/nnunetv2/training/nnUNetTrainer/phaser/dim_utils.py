"""
Dimension-adaptive factory functions for PhASER modules.

Provides drop-in replacements for Conv3d/InstanceNorm3d/trilinear etc.
that route to 2D or 3D ops based on a `spatial_dims` argument (2 or 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------

def conv_nd(dims: int, in_channels: int, out_channels: int,
            kernel_size: Union[int, Tuple[int, ...]],
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Union[int, Tuple[int, ...]] = 0,
            dilation: Union[int, Tuple[int, ...]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros'):
    """Return nn.Conv2d or nn.Conv3d based on dims."""
    Conv = nn.Conv2d if dims == 2 else nn.Conv3d
    return Conv(in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias, padding_mode)


def conv_nd_fn(dims: int, x: torch.Tensor, weight: torch.Tensor,
               bias: Optional[torch.Tensor] = None,
               stride=1, padding=0, dilation=1, groups=1):
    """Functional conv like F.conv2d or F.conv3d."""
    conv_fn = F.conv2d if dims == 2 else F.conv3d
    return conv_fn(x, weight, bias, stride, padding, dilation, groups)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def instancenorm_nd(dims: int, num_features: int, affine: bool = True,
                    eps: float = 1e-5):
    """Return nn.InstanceNorm2d or nn.InstanceNorm3d."""
    Norm = nn.InstanceNorm2d if dims == 2 else nn.InstanceNorm3d
    return Norm(num_features, affine=affine, eps=eps)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def adaptive_avg_pool_nd(dims: int, output_size=1):
    """Return nn.AdaptiveAvgPool2d or nn.AdaptiveAvgPool3d."""
    Pool = nn.AdaptiveAvgPool2d if dims == 2 else nn.AdaptiveAvgPool3d
    return Pool(output_size)


def max_pool_nd(dims: int, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False, return_indices=False):
    """Functional max_pool2d or max_pool3d."""
    fn = F.max_pool2d if dims == 2 else F.max_pool3d
    return fn


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_nd(dims: int, x: torch.Tensor,
                   size=None, scale_factor=None, mode=None,
                   align_corners=False, **kwargs):
    """
    F.interpolate with automatic mode selection.
    If mode is None, uses 'bilinear' for 2D, 'trilinear' for 3D.
    """
    if mode is None:
        mode = 'bilinear' if dims == 2 else 'trilinear'
    return F.interpolate(x, size=size, scale_factor=scale_factor,
                         mode=mode, align_corners=align_corners, **kwargs)


def interp_mode(dims: int):
    """Return the interpolation mode string for the given dims."""
    return 'bilinear' if dims == 2 else 'trilinear'


# ---------------------------------------------------------------------------
# Spatial reduction
# ---------------------------------------------------------------------------

def spatial_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean over all spatial dims. Works for 4D (B,C,H,W) and 5D (B,C,D,H,W)."""
    ndim = x.dim()
    spatial_dims = tuple(range(-(ndim - 2), 0))
    return x.mean(dim=spatial_dims)


def spatial_unsqueeze(x: torch.Tensor, target_dims: int = 5) -> torch.Tensor:
    """Unsqueeze a tensor to have target_dims total dimensions (last dims)."""
    while x.dim() < target_dims:
        x = x.unsqueeze(-1)
    return x


# ---------------------------------------------------------------------------
# DWT helpers (Haar wavelet)
# ---------------------------------------------------------------------------

def _dwt2d_haar(x: torch.Tensor):
    """
    Single-level 2D Haar DWT.  Input:  [B, C, H, W]
    Returns 4 sub-bands (LL, LH, HL, HH), each [B, C, H/2, W/2].
    """
    lo_h = (x[:, :, :, 0::2] + x[:, :, :, 1::2]) * 0.5
    hi_h = (x[:, :, :, 0::2] - x[:, :, :, 1::2]) * 0.5

    LL = (lo_h[:, :, 0::2, :] + lo_h[:, :, 1::2, :]) * 0.5
    LH = (lo_h[:, :, 0::2, :] - lo_h[:, :, 1::2, :]) * 0.5
    HL = (hi_h[:, :, 0::2, :] + hi_h[:, :, 1::2, :]) * 0.5
    HH = (hi_h[:, :, 0::2, :] - hi_h[:, :, 1::2, :]) * 0.5
    return LL, LH, HL, HH


def _idwt2d_haar(LL, LH, HL, HH):
    """
    Single-level 2D Haar inverse DWT.
    Inputs: each [B, C, H/2, W/2].  Output: [B, C, H, W].
    """
    B, C, Hh, Wh = LL.shape
    H2, W2 = Hh * 2, Wh * 2

    # Reverse step 2 (along H): reconstruct lo_h [B, C, H, W/2]
    lo_h = torch.zeros(B, C, H2, Wh, device=LL.device, dtype=LL.dtype)
    lo_h[:, :, 0::2, :] = LL + LH
    lo_h[:, :, 1::2, :] = LL - LH

    hi_h = torch.zeros(B, C, H2, Wh, device=LL.device, dtype=LL.dtype)
    hi_h[:, :, 0::2, :] = HL + HH
    hi_h[:, :, 1::2, :] = HL - HH

    # Reverse step 1 (along W): reconstruct y [B, C, H, W]
    y = torch.zeros(B, C, H2, W2, device=LL.device, dtype=LL.dtype)
    y[:, :, :, 0::2] = lo_h + hi_h
    y[:, :, :, 1::2] = lo_h - hi_h
    return y


def _dwt3d_haar(x: torch.Tensor):
    """
    Single-level 3D Haar DWT.  Input:  [B, C, D, H, W]
    Returns 8 sub-bands (LLL, LLH, LHL, HLL, LHH, HLH, HHL, HHH),
    each [B, C, D/2, H/2, W/2].
    """
    lo_x = (x[:, :, :, :, 0::2] + x[:, :, :, :, 1::2]) * 0.5
    hi_x = (x[:, :, :, :, 0::2] - x[:, :, :, :, 1::2]) * 0.5

    lo_y = (lo_x[:, :, :, 0::2, :] + lo_x[:, :, :, 1::2, :]) * 0.5
    hi_y = (lo_x[:, :, :, 0::2, :] - lo_x[:, :, :, 1::2, :]) * 0.5
    lo_lo_y = (hi_x[:, :, :, 0::2, :] + hi_x[:, :, :, 1::2, :]) * 0.5
    hi_hi_y = (hi_x[:, :, :, 0::2, :] - hi_x[:, :, :, 1::2, :]) * 0.5

    LLL = (lo_y[:, :, 0::2] + lo_y[:, :, 1::2]) * 0.5
    LLH = (lo_y[:, :, 0::2] - lo_y[:, :, 1::2]) * 0.5
    LHL = (hi_y[:, :, 0::2] + hi_y[:, :, 1::2]) * 0.5
    LHH = (hi_y[:, :, 0::2] - hi_y[:, :, 1::2]) * 0.5
    HLL = (lo_lo_y[:, :, 0::2] + lo_lo_y[:, :, 1::2]) * 0.5
    HLH = (lo_lo_y[:, :, 0::2] - lo_lo_y[:, :, 1::2]) * 0.5
    HHL = (hi_hi_y[:, :, 0::2] + hi_hi_y[:, :, 1::2]) * 0.5
    HHH = (hi_hi_y[:, :, 0::2] - hi_hi_y[:, :, 1::2]) * 0.5
    return LLL, LLH, LHL, HLL, LHH, HLH, HHL, HHH


def _idwt3d_haar(LLL, LLH, LHL, HLL, LHH, HLH, HHL, HHH):
    """
    Single-level 3D Haar inverse DWT.
    Inputs: each [B, C, D/2, H/2, W/2].  Output: [B, C, D, H, W].
    """
    B, C, Dh, Hh, Wh = LLL.shape
    D2, H2, W2 = Dh * 2, Hh * 2, Wh * 2

    # Step 1: reconstruct along D (dim 2)
    lo_v = torch.zeros(B, C, D2, Hh, Wh, device=LLL.device, dtype=LLL.dtype)
    lo_v[:, :, 0::2] = LLL + LLH
    lo_v[:, :, 1::2] = LLL - LLH

    hi_v = torch.zeros(B, C, D2, Hh, Wh, device=LLL.device, dtype=LLL.dtype)
    hi_v[:, :, 0::2] = LHL + LHH
    hi_v[:, :, 1::2] = LHL - LHH

    lo_lo_v = torch.zeros(B, C, D2, Hh, Wh, device=LLL.device, dtype=LLL.dtype)
    lo_lo_v[:, :, 0::2] = HLL + HLH
    lo_lo_v[:, :, 1::2] = HLL - HLH

    hi_hi_v = torch.zeros(B, C, D2, Hh, Wh, device=LLL.device, dtype=LLL.dtype)
    hi_hi_v[:, :, 0::2] = HHL + HHH
    hi_hi_v[:, :, 1::2] = HHL - HHH

    # Step 2: reconstruct along H (dim 3)
    lo_z = torch.zeros(B, C, D2, H2, Wh, device=LLL.device, dtype=LLL.dtype)
    lo_z[:, :, :, 0::2] = lo_v + hi_v
    lo_z[:, :, :, 1::2] = lo_v - hi_v

    hi_z = torch.zeros(B, C, D2, H2, Wh, device=LLL.device, dtype=LLL.dtype)
    hi_z[:, :, :, 0::2] = lo_lo_v + hi_hi_v
    hi_z[:, :, :, 1::2] = lo_lo_v - hi_hi_v

    # Step 3: reconstruct along W (dim 4)
    y = torch.zeros(B, C, D2, H2, W2, device=LLL.device, dtype=LLL.dtype)
    y[:, :, :, :, 0::2] = lo_z + hi_z
    y[:, :, :, :, 1::2] = lo_z - hi_z
    return y


def dwt_haar_nd(dims: int, x: torch.Tensor):
    """Haar DWT. Returns 4 subbands for 2D, 8 for 3D."""
    if dims == 2:
        return _dwt2d_haar(x)
    else:
        return _dwt3d_haar(x)


def idwt_haar_nd(dims: int, *subbands):
    """Haar inverse DWT. Expects 4 subbands for 2D, 8 for 3D."""
    if dims == 2:
        return _idwt2d_haar(*subbands)
    else:
        return _idwt3d_haar(*subbands)


def dwt_subband_count(dims: int) -> int:
    """Number of Haar DWT subbands: 4 for 2D, 8 for 3D."""
    return 4 if dims == 2 else 8


# ---------------------------------------------------------------------------
# Gradients / boundary
# ---------------------------------------------------------------------------

def spatial_gradient_magnitude(x: torch.Tensor, dims: int) -> torch.Tensor:
    """
    Compute spatial gradient magnitude via finite differences.
    2D: diffs along dims 2,3.  3D: along dims 2,3,4.
    Expects x in [B, C, ...].
    """
    if dims == 2:
        g_h = torch.diff(x, dim=2, prepend=x[:, :, :1])
        g_w = torch.diff(x, dim=3, prepend=x[:, :, :, :1])
        return torch.sqrt(g_h ** 2 + g_w ** 2 + 1e-8)
    else:
        g_d = torch.diff(x, dim=2, prepend=x[:, :, :1])
        g_h = torch.diff(x, dim=3, prepend=x[:, :, :, :1])
        g_w = torch.diff(x, dim=4, prepend=x[:, :, :, :, :1])
        return torch.sqrt(g_d ** 2 + g_h ** 2 + g_w ** 2 + 1e-8)


# ---------------------------------------------------------------------------
# Grid sample
# ---------------------------------------------------------------------------

def grid_sample_nd(dims: int, x: torch.Tensor, grid: torch.Tensor,
                   mode='bilinear', padding_mode='zeros', align_corners=False):
    """F.grid_sample. Grid should have (N, H, W, 2) for 2D, (N, D, H, W, 3) for 3D."""
    return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode,
                         align_corners=align_corners)


# ---------------------------------------------------------------------------
# Make nd-weight parameter
# ---------------------------------------------------------------------------

def nd_weight(channels: int, dims: int) -> nn.Parameter:
    """Parameter of shape [C, 1, 1] for 2D or [C, 1, 1, 1] for 3D."""
    shape = (channels, 1, 1) if dims == 2 else (channels, 1, 1, 1)
    return nn.Parameter(torch.ones(*shape))
