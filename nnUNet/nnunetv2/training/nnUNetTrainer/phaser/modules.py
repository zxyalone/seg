"""
PhASER Modules: PCSF, LWFD, ACG, and Content-Frequency Fusion.

All modules use residual initialization to preserve VoCo pretrained features.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from nnunetv2.training.nnUNetTrainer.phaser.dim_utils import (
    conv_nd,
    instancenorm_nd,
    interpolate_nd,
    spatial_gradient_magnitude,
    spatial_mean,
    nd_weight,
    dwt_haar_nd,
    idwt_haar_nd,
    dwt_subband_count,
)


# ---------------------------------------------------------------------------
# Phase-Conditioned Skip Fusion (PCSF)
# ---------------------------------------------------------------------------

class PhaseConditionedSkipFusion(nn.Module):
    """
    Boundary-aware skip connection enhancement.

    Detects tissue boundaries by computing the 3D spatial gradient magnitude
    of the encoder feature map, which highlights rapid feature changes
    (edges/boundaries) independent of absolute intensity.

    Theory:
      Feature gradient magnitude |∇f| = sqrt(|∂f/∂d|²+|∂f/∂h|²+|∂f/∂w|²)
      identifies spatial locations where feature representations change
      abruptly — these correspond to anatomical boundaries. This is equivalent
      to detecting phase edges in the frequency domain (by the Fourier
      differentiation theorem) but computed directly in spatial domain for
      efficiency (O(N) vs O(N log N) for FFT).

    The gradient map is passed through a learnable sigmoid gate and used
    to modulate the skip connection:
      skip' = skip * (1 + γ · attention)

    Initially γ=0, so the module is identity at training start.
    """

    def __init__(self, channels: int, dims: int = 3,
                 gate_sens_init: float = 0.5):
        super().__init__()
        self.dims = dims
        # gate_sens_init must be NONZERO. The effective sensitivity is
        # 10·tanh(alpha), and it multiplies the gate's input, so alpha = 0 makes
        # d(attention)/d(evidence) identically zero and no gradient reaches
        # anything upstream of the gate. That is harmless here (this module has
        # no upstream parameters) but fatal for PhaseCongruencySkipFusion, whose
        # learnable log-Gabor filters sit upstream. Both variants therefore share
        # the same nonzero default so the grad-vs-pc ablation stays matched on
        # initialisation. Set PHASER_PCSF_GATE_SENS=0.0 to recover the historical
        # alpha = 0 behaviour of runs logged before 2026-09.
        self.alpha = nn.Parameter(torch.tensor(float(gate_sens_init)))
        self.beta = nn.Parameter(torch.tensor(0.0))    # gate threshold (sigmoid-ed)
        self.gamma = nn.Parameter(torch.tensor(0.0))   # enhancement strength

    def forward(self, enc_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_feat: [B, C, D, H, W] encoder skip feature (or [B, C, H, W] for 2D)
        Returns:
            modulated feature of same shape
        """
        G = spatial_gradient_magnitude(enc_feat, self.dims)

        # Average over channels → 1-channel boundary map
        attn_spatial = G.mean(dim=1, keepdim=True)

        # Normalise to [0, 1] per sample
        spatial_axes = tuple(range(-self.dims, 0))
        a_max = attn_spatial.amax(dim=spatial_axes, keepdim=True).clamp(min=1e-8)
        a_min = attn_spatial.amin(dim=spatial_axes, keepdim=True)
        attn_norm = (attn_spatial - a_min) / (a_max - a_min + 1e-8)

        # Sigmoid gate
        gate_sensitivity = torch.tanh(self.alpha) * 10.0
        gate_threshold = torch.sigmoid(self.beta)
        attention = torch.sigmoid(gate_sensitivity * (attn_norm - gate_threshold))

        # Broadcast back to C channels
        attention = attention.expand_as(enc_feat)

        # Modulate (gamma=0 → identity at init)
        return enc_feat * (1.0 + torch.tanh(self.gamma) * attention)


# ---------------------------------------------------------------------------
# Phase Congruency (monogenic signal) — contrast-invariant boundary evidence
# ---------------------------------------------------------------------------

class PhaseCongruency(nn.Module):
    """
    Monogenic-signal phase congruency with learnable log-Gabor scales (2D/3D).

    Motivation
    ----------
    Phase congruency (Kovesi, 1999) marks locations where the Fourier components
    of a signal are maximally *in phase*. Its defining property is that it is
    invariant to signal amplitude and contrast:

        PC(k · x) == PC(x)    for any k > 0

    That invariance is the entire reason this module exists. Isoattenuating liver
    lesions are, by definition, lesions whose CT attenuation nearly matches the
    surrounding parenchyma: their *magnitude* spectrum is almost identical to
    healthy tissue. Any amplitude-based edge cue — finite-difference gradient
    magnitude |∇f| as used by `PhaseConditionedSkipFusion`, or a Sobel/Canny
    response — is therefore weakest exactly where the segmentation problem is
    hardest. Local *phase* structure still changes across the lesion margin, and
    phase congruency reads that change directly.

    Formulation
    -----------
    The 1D quadrature pair (even/odd filter) does not generalise to nD as a
    scalar, so this uses the monogenic signal (Felsberg & Sommer, 2001): the
    Riesz transform generalises the 1D Hilbert transform to nD and yields an
    *isotropic* quadrature pair with no orientation filter bank.

    With F = rfftn(x), radial frequency ρ = |u|, and per-scale log-Gabor

        G_s(ρ) = exp( −ln(ρ/ρ_s)² / (2 ln(σ_s)²) ),    G_s(0) = 0

    the even part and the Riesz (odd) vector at scale s are

        e_s     = irfftn( F · G_s(ρ) )
        o_{s,j} = irfftn( F · G_s(ρ) · i·u_j/ρ ),    j = 1..dims

    Stacking the monogenic vector v_s = (e_s, o_{s,1}, …, o_{s,dims}) ∈ R^(1+dims)
    with per-scale amplitude A_s = ‖v_s‖, phase congruency is

        PC(x) = ‖ Σ_s v_s ‖ / ( Σ_s A_s + ε )        ∈ [0, 1]

    PC → 1 where every scale agrees in phase and orientation (a true feature
    point) and PC → 0 under phase-incoherent noise. Both numerator and
    denominator are homogeneous of degree 1 in x, which is precisely where the
    contrast invariance comes from. This is the same PC_1 form used by FSIM.

    ρ_s (centre frequency) and σ_s (bandwidth ratio) are **learnable**, so the
    network tunes which spatial scales of phase coherence matter at each skip
    level rather than inheriting hand-set filter constants.

    Notes
    -----
    - Output is already in [0, 1] and contrast-invariant, so it needs no
      per-sample min-max normalisation. That is a real advantage over the
      gradient path, whose min-max step lets a single outlier voxel set the
      scale of the whole attention map.
    - torch.fft has no half-precision support, so the transform block forces
      fp32 regardless of autocast.
    - `max_voxels` caps FFT cost: above it the proxy is downsampled by powers of
      two, PC is computed there, and the map is upsampled back. The default
      2^20 is below 128^3, so on a standard patch the shallowest skip is
      computed at 64^3. This is not only a cost device: the 2x average
      denoises. Measured (step2_mechanism/p0_fix_report.md): removing the cap
      made boundary localisation worse on a synthetic slab (AUC at CNR 2/4:
      1.00/1.00 capped vs 0.50/0.52 uncapped) but better on one real patch
      (boundary/peri ratio 0.62 vs 0.37). The default is kept; the effective
      `downsample_factor` is reported in _last_diag.
    - `spacing` (mm/voxel per axis) makes the log-Gabor bank isotropic in
      physical space. None (default) keeps cycles/voxel; enabling it
      reinterprets the learnable rho and rescales the bank by up to 1.28x on
      LiTS, with no clear win on one real patch, so it is an ablation only.
    - `noise_rel_kappa` optionally caps the Kovesi threshold at kappa * sum_s A_s.
      Measured and rejected as a default: it stops the threshold from
      sparsifying the output (PC > 0 for 94% of voxels vs 48%) and boundary
      AUC on a synthetic slab fell to 0.39/0.29/0.25 at CNR 0.5/1/2 (uncapped
      0.52/0.67/1.00), because num/den is ~1 at both boundary and interior of
      step-like objects. 0.0 (default) = pure Kovesi threshold.
    """

    def __init__(self, dims: int = 3, n_scales: int = 3,
                 min_wavelength: float = 3.0, mult: float = 2.1,
                 sigma_onf: float = 0.55, noise_k: float = 2.0,
                 max_voxels: int = 1 << 20, eps: float = 1e-3,
                 legacy: bool = False, spacing=None, noise_rel_kappa: float = 0.0):
        super().__init__()
        self.dims = dims
        self.n_scales = n_scales
        self.max_voxels = max_voxels
        self.eps = eps
        # Physical voxel spacing per tensor axis; None keeps the grid in
        # cycles/voxel (d = 1), the behaviour every checkpoint was trained with.
        if spacing is None:
            self.spacing = (1.0,) * dims
        else:
            spacing = tuple(float(s) for s in spacing)
            if len(spacing) != dims or any(s <= 0 for s in spacing):
                raise ValueError(f"spacing must be {dims} positive values, got {spacing}")
            self.spacing = spacing
        self.noise_rel_kappa = float(noise_rel_kappa)
        # Eval-mode diagnostics (threshold, effective resolution, PC sparsity)
        # for the mechanism probes.
        self._last_diag = None
        # legacy=True estimates the noise scale from A[:, 0] (the scale that was
        # *initialised* finest) as pre-V2 checkpoints did. V2 uses the scale
        # whose learned ρ_s is currently highest, because nothing keeps the
        # learned centre frequencies in their initial order.
        self.legacy = legacy

        # Centre frequencies ρ_s = 1/λ_s for λ_s = min_wavelength · mult^s,
        # stored as raw logits so ρ_s = 0.5·σ(raw) stays inside (0, Nyquist).
        wavelengths = [min_wavelength * (mult ** s) for s in range(n_scales)]
        rho = torch.tensor([1.0 / w for w in wavelengths], dtype=torch.float32)
        rho = rho.clamp(1e-4, 0.499)
        self.rho_raw = nn.Parameter(torch.logit(rho / 0.5))

        # Bandwidth ratio σ_onf ∈ (0.05, 0.95); ln(σ)² is the log-Gabor variance,
        # so σ must stay strictly below 1.
        sig = torch.full((n_scales,), float(sigma_onf)).clamp(0.06, 0.94)
        self.sigma_raw = nn.Parameter(torch.logit((sig - 0.05) / 0.9))

        # Noise-threshold multiplier k (Kovesi's default is 2.0). Learnable via
        # softplus so it stays non-negative: k = softplus(raw), raw = ln(e^2 − 1).
        self.noise_k = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(noise_k))))
        )

        # Frequency grids are shape-dependent but parameter-independent, so they
        # are cached per (shape, device) and reused across steps.
        self._grid_cache = {}

    # -- frequency grid -------------------------------------------------
    def _freq_grid(self, spatial, device):
        key = (tuple(spatial), str(device), self.spacing)
        hit = self._grid_cache.get(key)
        if hit is not None:
            return hit

        axes = []
        for i, n in enumerate(spatial):
            # rfftn only halves the LAST transformed axis. d = spacing, so the
            # grid is in cycles/voxel by default and cycles/mm with spacing.
            if i == len(spatial) - 1:
                axes.append(torch.fft.rfftfreq(n, d=self.spacing[i], device=device))
            else:
                axes.append(torch.fft.fftfreq(n, d=self.spacing[i], device=device))
        grids = torch.meshgrid(*axes, indexing='ij')

        rho = torch.sqrt(sum(g ** 2 for g in grids))
        dc = (rho == 0)
        rho_safe = rho.clamp(min=1e-12)
        # Riesz unit vectors u_j/ρ (the i factor is applied when building the
        # complex kernel). Zeroed at DC where the direction is undefined.
        unit = torch.stack([g / rho_safe for g in grids], dim=0)
        unit = unit.masked_fill(dc.unsqueeze(0), 0.0)
        log_rho = torch.log(rho_safe)

        self._grid_cache[key] = (log_rho, unit, dc)
        return self._grid_cache[key]

    # -- forward --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, *spatial] single-channel intensity proxy
        Returns:
            [B, 1, *spatial] phase congruency in [0, 1]
        """
        dims = self.dims
        spatial = tuple(x.shape[2:])
        S = self.n_scales
        if self.training:
            self._last_diag = None

        # torch.fft is fp32-only; opt out of autocast for the whole transform.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()

            # Cost control: shrink by powers of two until under the voxel cap,
            # never below 8 voxels on any axis (log-Gabor needs some support).
            factor = 1
            def _numel(f):
                n = 1
                for s in spatial:
                    n *= max(s // f, 1)
                return n
            while (_numel(factor) > self.max_voxels
                   and all(s // (factor * 2) >= 8 for s in spatial)):
                factor *= 2
            if factor > 1:
                xf = interpolate_nd(dims, xf,
                                    size=tuple(s // factor for s in spatial))
            work = tuple(xf.shape[2:])

            fft_dims = tuple(range(-dims, 0))
            F_x = torch.fft.rfftn(xf, dim=fft_dims)              # [B,1,*rf]
            log_rho, unit, dc = self._freq_grid(work, xf.device)

            # Learnable log-Gabor bank: [S, *rf]
            rho_s = 0.5 * torch.sigmoid(self.rho_raw)
            sig_s = 0.05 + 0.9 * torch.sigmoid(self.sigma_raw)
            shape_s = (-1,) + (1,) * dims
            ln_ratio = log_rho.unsqueeze(0) - torch.log(rho_s).view(*shape_s)
            denom = 2.0 * (torch.log(sig_s) ** 2).view(*shape_s)
            G = torch.exp(-(ln_ratio ** 2) / denom)
            G = G.masked_fill(dc.unsqueeze(0), 0.0)   # log-Gabor carries no DC

            # Complex kernel: 1 even (real) + dims odd/Riesz (imaginary) per scale
            Gk = G.unsqueeze(1)                                  # [S,1,*rf]
            odd_imag = Gk * unit.unsqueeze(0)                    # [S,dims,*rf]
            K = torch.cat([
                torch.complex(Gk, torch.zeros_like(Gk)),
                torch.complex(torch.zeros_like(odd_imag), odd_imag),
            ], dim=1).reshape(S * (1 + dims), *G.shape[1:])

            # One batched inverse transform for all scales and components.
            Y = torch.fft.irfftn(F_x * K.unsqueeze(0), s=work, dim=fft_dims)
            Y = Y.reshape(-1, S, 1 + dims, *work)

            A = Y.norm(dim=2)                        # [B,S,*work] per-scale amp
            V = Y.sum(dim=1)                         # [B,1+dims,*work] vector sum
            num = V.norm(dim=1)                      # [B,*work] local energy E
            den = A.sum(dim=1)                       # [B,*work]

            # --- Noise threshold (Kovesi) --------------------------------
            # The bare ratio E/ΣA has no noise compensation and reads high in
            # flat, phase-incoherent regions — i.e. it would light up liver
            # parenchyma texture as if it were a lesion margin. Kovesi's fix:
            # white-noise filter responses are Rayleigh distributed, so estimate
            # the noise scale robustly from the median amplitude of the
            # highest-frequency scale and subtract a multiple of it.
            #
            #   median(A) = σ·sqrt(2 ln 2)          ⇒  σ = median / sqrt(ln 4)
            #   noise energy over scales sums as σ_s ∝ ρ_s (white noise, so each
            #   filter's response scales with its absolute bandwidth ∝ ρ_s)
            #   T = τ_total·sqrt(π/2) + k·τ_total·sqrt((4−π)/2)
            #
            # The median is detached: it is a noise-level *estimate*, and a
            # median over ~10^6 voxels would otherwise route the entire gradient
            # through a single arbitrarily-selected element. k stays learnable so
            # the network still tunes how aggressively noise is suppressed.
            finest = 0 if self.legacy else int(torch.argmax(rho_s.detach()))
            A0 = A[:, finest].detach()               # highest-frequency scale
            med = A0.flatten(1).median(dim=1).values
            tau = med / math.sqrt(math.log(4.0))     # [B]
            # Generalises Kovesi's geometric sum (1−mult^−S)/(1−mult^−1) to the
            # learned, non-geometric ρ_s by using the actual frequency ratios.
            gain = (rho_s / rho_s.max().clamp(min=1e-12)).sum()
            tau_total = tau * gain
            k = F.softplus(self.noise_k)
            T = tau_total * (math.sqrt(math.pi / 2.0)
                             + k * math.sqrt((4.0 - math.pi) / 2.0))
            T = T.view(-1, *((1,) * dims))           # [B,1,..] broadcast
            T_kovesi = T
            # Optional relative cap (off by default, see class docstring).
            if self.noise_rel_kappa > 0:
                T = torch.minimum(T, self.noise_rel_kappa * den)

            # Relative epsilon: den.mean() is homogeneous of degree 1 in x just
            # like num and den, so contrast invariance survives the guard term
            # (an absolute epsilon would break it for low-amplitude features).
            # T is degree 1 as well, so subtracting it keeps PC(k·x) == PC(x).
            sp = tuple(range(1, den.dim()))
            floor = self.eps * den.mean(dim=sp, keepdim=True)
            pc = (num - T).clamp(min=0.0) / (den + floor + 1e-12)
            pc = pc.unsqueeze(1).clamp(0.0, 1.0)     # [B,1,*work]

            if not self.training:
                with torch.no_grad():
                    dclamp = den.clamp(min=1e-12)
                    self._last_diag = {
                        'T_kovesi': float(T_kovesi.mean()),
                        'T_eff': float(T.mean()),
                        'den_mean': float(den.mean()),
                        'T_over_den_mean': float((T / dclamp).mean()),
                        'T_kovesi_over_den_mean': float((T_kovesi / dclamp).mean()),
                        'kappa': self.noise_rel_kappa,
                        'spacing': list(self.spacing),
                        'downsample_factor': int(factor),
                        'work_shape': list(work),
                        'frac_den_above_T_kovesi': float((den > T_kovesi).float().mean()),
                        'pc_mean': float(pc.mean()),
                        'pc_frac_nonzero': float((pc > 0).float().mean()),
                        'num_over_den_mean': float((num / dclamp).mean()),
                    }

            if factor > 1:
                pc = interpolate_nd(dims, pc, size=spatial)

        return pc.to(x.dtype)


class PhaseCongruencySkipFusion(nn.Module):
    """
    Phase-congruency skip modulation — the 'pc' mode of PCSF.

    Drop-in replacement for `PhaseConditionedSkipFusion` that swaps the
    amplitude-based boundary evidence (finite-difference gradient magnitude) for
    contrast-invariant phase congruency, while keeping the identical gate
    parameterisation and identity initialisation so the two are directly
    ablatable against each other.

        p     = Conv1x1(x)                    channel reduction to 1-ch proxy
        PC    = PhaseCongruency(p)            in [0, 1], contrast-invariant
        attn  = σ( (10·tanh α) · (PC − σ(β)) )
        out   = x · (1 + tanh(γ) · attn)

    γ = 0 at init → exact identity, preserving the VoCo pretrained features.

    Unlike the gradient path this needs no per-sample min-max normalisation: PC
    is already bounded and scale-free. The FFT runs on the 1-channel proxy, not
    on all C channels, so cost is independent of channel width.
    """

    def __init__(self, channels: int, dims: int = 3, n_scales: int = 3,
                 pc_max_voxels: int = 1 << 20, gate_sens_init: float = 0.5,
                 spacing=None, noise_rel_kappa: float = 0.0):
        super().__init__()
        self.dims = dims
        # Channel reduction. Initialised to the plain channel mean so the proxy
        # starts as the same statistic the gradient path used (G.mean(dim=1)),
        # then becomes free to specialise.
        self.proxy = conv_nd(dims, channels, 1, 1, bias=False)
        nn.init.constant_(self.proxy.weight, 1.0 / max(channels, 1))

        self.pc = PhaseCongruency(dims=dims, n_scales=n_scales,
                                  max_voxels=pc_max_voxels, spacing=spacing,
                                  noise_rel_kappa=noise_rel_kappa)

        # γ = 0 gives exact identity, which is what protects the VoCo weights.
        # α, by contrast, MUST start nonzero: the gate applies 10·tanh(α) to its
        # input, so α = 0 makes ∂attention/∂PC identically zero and the log-Gabor
        # parameters below receive no gradient at all — not merely a slow start.
        # Measured with α = 0: after 5 SGD steps (lr=5e-3, momentum=0.99) rho_raw,
        # sigma_raw, noise_k and proxy.weight had not moved from their init values
        # and their gradients were ~1e-16, because the only escape route is the
        # second-order product tanh(γ)·tanh(α). With the nonzero default the
        # filters train from step 2 (γ needs one step to leave zero).
        # Kept identical to the gradient variant's default so grad-vs-pc remains
        # a single-variable ablation.
        self.alpha = nn.Parameter(torch.tensor(float(gate_sens_init)))
        self.beta = nn.Parameter(torch.tensor(0.0))    # gate threshold
        self.gamma = nn.Parameter(torch.tensor(0.0))   # enhancement strength

        # Exposed for Figure-1 style visualisation / analysis, not for training.
        self._last_pc = None

    def forward(self, enc_feat: torch.Tensor) -> torch.Tensor:
        pc = self.pc(self.proxy(enc_feat))

        # Keep the live (graph-attached) tensor while training so
        # pc_boundary_ranking_loss can supervise the log-Gabor filters through a
        # path that bypasses the tanh(gamma) gate; detach in eval so inference
        # and visualisation do not retain a graph. Same pattern the rest of
        # PhaserUNet uses for _last_gates / _last_liver_prior / _last_small_logits.
        self._last_pc = pc if self.training else pc.detach()

        gate_sensitivity = torch.tanh(self.alpha) * 10.0
        gate_threshold = torch.sigmoid(self.beta)
        attention = torch.sigmoid(gate_sensitivity * (pc - gate_threshold))

        return enc_feat * (1.0 + torch.tanh(self.gamma) * attention)


class PhaseFrequencyCoherenceRouting(nn.Module):
    """
    Route Haar subbands with contrast-invariant phase evidence.

    Control and analysis switches (all default off):
      shuffle_pc  A7 capacity control. The router sees a per-sample random
                  spatial permutation of PC, so it keeps the PC value
                  distribution, parameters and compute but loses its spatial
                  content. _last_pc stays unshuffled, so the ranking loss still
                  trains the filters exactly as in the full model.
      intervene   inference-time intervention (E11), one of
                  'none' | 'pc_zero' | 'route_uniform' | 'prior_zero' | 'gate_zero'.
      legacy      pre-V2 behaviour: the prior is passed through a second
                  sigmoid and the PC noise scale is taken from scale 0. Only for
                  re-running inference with old checkpoints.
    """

    INTERVENTIONS = ('none', 'pc_zero', 'route_uniform', 'prior_zero', 'gate_zero')

    def __init__(self, channels: int, dims: int = 3, n_scales: int = 3,
                 pc_max_voxels: int = 1 << 20, use_prior: bool = False,
                 shuffle_pc: bool = False, intervene: str = 'none',
                 legacy: bool = False, spacing=None, noise_rel_kappa: float = 0.0):
        super().__init__()
        if intervene not in self.INTERVENTIONS:
            raise ValueError(f"intervene must be one of {self.INTERVENTIONS}, got {intervene!r}")
        self.dims = dims
        self.use_prior = use_prior
        self.shuffle_pc = shuffle_pc
        self.intervene = intervene
        self.legacy = legacy
        self.proxy = conv_nd(dims, channels, 1, 1, bias=False)
        nn.init.constant_(self.proxy.weight, 1.0 / max(channels, 1))
        self.pc = PhaseCongruency(dims=dims, n_scales=n_scales,
                                  max_voxels=pc_max_voxels, legacy=legacy,
                                  spacing=spacing, noise_rel_kappa=noise_rel_kappa)
        self.router = nn.Sequential(
            conv_nd(dims, 2 if use_prior else 1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            conv_nd(dims, 8, 3, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        self._last_pc = None
        self._last_route = None

    def forward(self, x: torch.Tensor, tumor_prior: torch.Tensor = None):
        original_shape = x.shape[2:]
        pad = tuple(size % 2 for size in original_shape)
        if any(pad):
            if self.dims == 3:
                x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
            else:
                x = F.pad(x, (0, pad[1], 0, pad[0]))
        pc = self.pc(self.proxy(x))
        self._last_pc = pc if self.training else pc.detach()
        pc_in = pc
        if self.shuffle_pc:
            flat = pc.flatten(2)
            perm = torch.stack([torch.randperm(flat.shape[-1], device=pc.device)
                                for _ in range(flat.shape[0])])
            pc_in = torch.gather(flat, 2, perm.unsqueeze(1)).view_as(pc)
        if self.intervene == 'pc_zero':
            pc_in = torch.zeros_like(pc_in)
        route_input = pc_in
        if self.use_prior and tumor_prior is not None:
            prior = interpolate_nd(self.dims, tumor_prior, size=pc.shape[2:])
            if self.legacy:
                prior = prior.sigmoid()
            if self.intervene == 'prior_zero':
                prior = torch.zeros_like(prior)
            route_input = torch.cat((pc_in, prior), dim=1)
        route = torch.softmax(self.router(route_input), dim=1)
        if self.intervene == 'route_uniform':
            route = torch.full_like(route, 1.0 / route.shape[1])
        self._last_route = route if self.training else route.detach()
        if self.intervene == 'gate_zero':
            if any(pad):
                slices = (slice(None), slice(None), *[slice(0, size) for size in original_shape])
                x = x[slices]
            return x, route
        bands = dwt_haar_nd(self.dims, x)
        band_route = interpolate_nd(self.dims, route, size=bands[0].shape[2:])
        if self.dims == 3:
            routed = idwt_haar_nd(3, bands[0] * band_route[:, 0:1],
                bands[1] * band_route[:, 1:2], bands[2] * band_route[:, 1:2],
                bands[3] * band_route[:, 1:2], bands[4] * band_route[:, 2:3],
                bands[5] * band_route[:, 2:3], bands[6] * band_route[:, 2:3],
                bands[7] * band_route[:, 2:3])
        else:
            routed = idwt_haar_nd(2, bands[0] * band_route[:, 0:1],
                bands[1] * band_route[:, 1:2], bands[2] * band_route[:, 1:2],
                bands[3] * band_route[:, 2:3])
        if any(pad):
            slices = (slice(None), slice(None), *[slice(0, size) for size in original_shape])
            routed, x = routed[slices], x[slices]
        return x + torch.tanh(self.residual_gate) * routed, route


# ---------------------------------------------------------------------------
# Learnable Wavelet Feature Decomposition (LWFD)
# ---------------------------------------------------------------------------

class LearnableWaveletFeatureDecomposition(nn.Module):
    """
    Frequency-band decomposition of encoder features via Haar DWT.

    Two modes:
      - 'lll'  (default):  only LLL subband reconstructed, others zeroed
                            (simpler, keeps only low-frequency enhancement)
      - 'full':            all subbands independently processed and
                            reconstructed (preserves high-frequency detail)

    3D DWT decomposes input into 8 sub-bands:
      - LLL: all-low  (liver parenchyma, homogeneous regions)
      - LLH, LHL, HLL: one high-direction (vessels, directional texture)
      - LHH, HLH, HHL, HHH: >=2 high directions (boundaries, small lesions)

    2D DWT decomposes input into 4 sub-bands:
      - LL: all-low
      - LH, HL: one high-direction
      - HH: two high directions

    Initialised as identity (residual gate = 0) regardless of mode.
    """

    def __init__(self, channels: int, mode: str = 'lll', dims: int = 3):
        super().__init__()
        self.mode = mode
        self.dims = dims
        mid = max(channels // 4, 8)

        if dims == 3:
            mid_in_ch = 3 * channels
            high_in_ch = 4 * channels
        else:
            mid_in_ch = 2 * channels
            high_in_ch = channels

        # --- Group processors (shared by both modes) ---
        self.conv_low = nn.Sequential(
            conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
            conv_nd(dims, channels, channels, 1),
            instancenorm_nd(dims, channels, affine=True),
        )
        self.conv_mid = nn.Sequential(
            conv_nd(dims, mid_in_ch, mid, 3, padding=1, groups=min(mid_in_ch, mid)),
            conv_nd(dims, mid, channels, 1),
            instancenorm_nd(dims, channels, affine=True),
        )
        self.conv_high = nn.Sequential(
            conv_nd(dims, high_in_ch, mid, 3, padding=1, groups=min(high_in_ch, mid)),
            conv_nd(dims, mid, channels, 1),
            instancenorm_nd(dims, channels, affine=True),
        )

        # --- Full-band: per-subband independent processors ---
        if mode == 'full':
            if dims == 3:
                self.conv_llh = nn.Sequential(
                    conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
                    conv_nd(dims, channels, channels, 1),
                    instancenorm_nd(dims, channels, affine=True),
                )
                self.conv_lhl = nn.Sequential(
                    conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
                    conv_nd(dims, channels, channels, 1),
                    instancenorm_nd(dims, channels, affine=True),
                )
                self.conv_hll = nn.Sequential(
                    conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
                    conv_nd(dims, channels, channels, 1),
                    instancenorm_nd(dims, channels, affine=True),
                )
                self.conv_lhh = nn.Sequential(
                    conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
                    conv_nd(dims, channels, channels, 1),
                    instancenorm_nd(dims, channels, affine=True),
                )
                self.conv_hlh = nn.Sequential(
                    conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
                    conv_nd(dims, channels, channels, 1),
                    instancenorm_nd(dims, channels, affine=True),
                )
                self.conv_hhl = nn.Sequential(
                    conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
                    conv_nd(dims, channels, channels, 1),
                    instancenorm_nd(dims, channels, affine=True),
                )
                self.conv_hhh = nn.Sequential(
                    conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
                    conv_nd(dims, channels, channels, 1),
                    instancenorm_nd(dims, channels, affine=True),
                )
            else:
                self.conv_lh = nn.Sequential(
                    conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
                    conv_nd(dims, channels, channels, 1),
                    instancenorm_nd(dims, channels, affine=True),
                )
                self.conv_hl = nn.Sequential(
                    conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
                    conv_nd(dims, channels, channels, 1),
                    instancenorm_nd(dims, channels, affine=True),
                )
                self.conv_hh = nn.Sequential(
                    conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
                    conv_nd(dims, channels, channels, 1),
                    instancenorm_nd(dims, channels, affine=True),
                )

        # Per-channel fusion weights
        self.w_low = nd_weight(channels, dims)
        self.w_mid = nd_weight(channels, dims)
        self.w_high = nd_weight(channels, dims)

        # Residual gate: 0 → identity at init
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] (or [B, C, H, W] for 2D)
        Returns:
            [B, C, D, H, W] (or [B, C, H, W] for 2D)
        """
        B, C = x.shape[:2]
        orig_shape = x.shape[2:]

        # Pad to even dims for DWT
        pad = tuple(0 if s % 2 == 0 else 1 for s in orig_shape)
        need_pad = any(p > 0 for p in pad)
        if need_pad:
            if self.dims == 3:
                x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
            else:
                x = F.pad(x, (0, pad[1], 0, pad[0]))

        if self.dims == 3:
            LLL, LLH, LHL, HLL, LHH, HLH, HHL, HHH = dwt_haar_nd(3, x)

            if self.mode == 'lll':
                # --- Original: group processing → fused → only LLL reconstructed ---
                f_low = self.conv_low(LLL)
                f_mid = self.conv_mid(torch.cat([LLH, LHL, HLL], dim=1))
                f_high = self.conv_high(torch.cat([LHH, HLH, HHL, HHH], dim=1))
                fused = self.w_low * f_low + self.w_mid * f_mid + self.w_high * f_high
                zeros = torch.zeros_like(LLL)
                out = idwt_haar_nd(3, fused, zeros, zeros, zeros, zeros, zeros, zeros, zeros)

            else:  # 'full'
                f_LLL = self.conv_low(LLL)
                f_LLH = self.conv_llh(LLH)
                f_LHL = self.conv_lhl(LHL)
                f_HLL = self.conv_hll(HLL)
                f_LHH = self.conv_lhh(LHH)
                f_HLH = self.conv_hlh(HLH)
                f_HHL = self.conv_hhl(HHL)
                f_HHH = self.conv_hhh(HHH)

                out = idwt_haar_nd(3,
                    f_LLL, f_LLH, f_LHL, f_HLL,
                    f_LHH, f_HLH, f_HHL, f_HHH,
                )

        else:  # dims == 2
            LL, LH, HL, HH = dwt_haar_nd(2, x)

            if self.mode == 'lll':
                f_low = self.conv_low(LL)
                f_mid = self.conv_mid(torch.cat([LH, HL], dim=1))
                f_high = self.conv_high(HH)
                fused = self.w_low * f_low + self.w_mid * f_mid + self.w_high * f_high
                zeros = torch.zeros_like(LL)
                out = idwt_haar_nd(2, fused, zeros, zeros, zeros)

            else:  # 'full'
                f_LL = self.conv_low(LL)
                f_LH = self.conv_lh(LH)
                f_HL = self.conv_hl(HL)
                f_HH = self.conv_hh(HH)

                out = idwt_haar_nd(2, f_LL, f_LH, f_HL, f_HH)

        # Trim padding
        if need_pad:
            if self.dims == 3:
                out = out[:, :, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
            else:
                out = out[:, :, :orig_shape[0], :orig_shape[1]]

        # Residual connection (gate=0 → identity)
        gate = torch.tanh(self.residual_gate)
        if self.dims == 3:
            x_trim = x[:, :, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
        else:
            x_trim = x[:, :, :orig_shape[0], :orig_shape[1]]
        return x_trim + gate * out


# ---------------------------------------------------------------------------
# Anatomical Context Gating (ACG) — Original GAP-based
# ---------------------------------------------------------------------------

class AnatomicalContextGating(nn.Module):
    """
    Global anatomical context extracted from the bottleneck that conditions
    each decoder stage via channel-wise feature modulation.

    bottleneck → GAP → MLP → per-stage Linear → [B, Cᵢ, 1, 1] or [B, Cᵢ, 1, 1, 1]

    Each decoder stage receives:    dec_feat' = dec_feat * (1 + mod_vec)
    where mod_vec ∈ [-1, 1] per channel (tanh), initially ~0.
    """

    def __init__(self, bottleneck_channels: int, decoder_channels: List[int], dims: int = 3):
        super().__init__()
        self.dims = dims
        hidden = max(bottleneck_channels // 4, 32)

        self.context_net = nn.Sequential(
            nn.Linear(bottleneck_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

        self.modulators = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, c), nn.Tanh())
            for c in decoder_channels
        ])

        for mod in self.modulators:
            nn.init.zeros_(mod[0].weight)
            nn.init.zeros_(mod[0].bias)

    def forward(self, bottleneck: torch.Tensor) -> List[torch.Tensor]:
        g = spatial_mean(bottleneck)                    # [B, C_bn]
        g = self.context_net(g)                         # [B, hidden]
        mod_vectors = []
        for mod in self.modulators:
            v = mod(g)                                  # [B, C_i]
            for _ in range(self.dims):
                v = v.unsqueeze(-1)
            mod_vectors.append(v)
        return mod_vectors


# ---------------------------------------------------------------------------
# Utility: create high-pass mask for spectral consistency loss
# ---------------------------------------------------------------------------

def create_high_pass_mask(shape, cutoff: float = 0.3):
    """
    Create a 3D high-pass mask in frequency domain.

    Args:
        shape: (D, H, W_last) where W_last = W//2 + 1 for rfft
        cutoff: fraction of Nyquist below which to suppress
    Returns:
        mask of shape (1, 1, D, H, W_last) with 0 at low freq, 1 at high freq
    """
    D, H, W = shape
    freqs_d = torch.fft.fftfreq(D, d=1.0).abs() / (0.5 if D > 1 else 0.5)
    freqs_h = torch.fft.fftfreq(H, d=1.0).abs() / (0.5 if H > 1 else 0.5)
    freqs_w = torch.fft.rfftfreq(W * 2 - 1).abs() / (0.5 if W > 1 else 0.5)

    grid = (freqs_d.view(-1, 1, 1) ** 2 +
            freqs_h.view(1, -1, 1) ** 2 +
            freqs_w.view(1, 1, -1) ** 2).sqrt()

    mask = (grid > cutoff).float()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]


# ---------------------------------------------------------------------------
# CFF Variant 1: Content-Frequency Fusion (Concat + Conv1x1)
# ---------------------------------------------------------------------------

class ContentFrequencyFusion(nn.Module):
    """skip_fused = Conv1x1(cat([original, processed]))"""

    def __init__(self, channels: int, ctx_channels: int = 0, dims: int = 3):
        super().__init__()
        self.dims = dims
        self.fusion = conv_nd(dims, channels * 2, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.fusion.weight)
        with torch.no_grad():
            for i in range(channels):
                if dims == 3:
                    self.fusion.weight[i, i, 0, 0, 0] = 1.0
                else:
                    self.fusion.weight[i, i, 0, 0] = 1.0

    def forward(self, original: torch.Tensor, processed: torch.Tensor, ctx=None):
        return self.fusion(torch.cat([original, processed], dim=1)), None


# ---------------------------------------------------------------------------
# CFF Variant 2: Gated Fusion (learnable per-voxel gate)
# ---------------------------------------------------------------------------

class GatedContentFrequencyFusion(nn.Module):
    """
    g = σ(Conv1x1([F_s, F_w]))
    F_fused = g ⊙ F_s + (1 - g) ⊙ F_w

    gate g ∈ [0, 1] is a per-voxel spatial map, directly interpretable.
    """

    def __init__(self, channels: int, ctx_channels: int = 0, dims: int = 3):
        super().__init__()
        self.dims = dims
        hidden = max(channels // 8, 8)
        self.gate_net = nn.Sequential(
            conv_nd(dims, channels * 2, hidden, kernel_size=3, padding=2, dilation=2, groups=4, bias=True),
            nn.ReLU(inplace=True),
            conv_nd(dims, hidden, 1, kernel_size=1, bias=True),
        )
        with torch.no_grad():
            self.gate_net[-1].bias.fill_(8.0)
            self.gate_net[-1].weight.zero_()
        if ctx_channels > 0:
            self.ctx_proj = nn.Linear(ctx_channels, 1)
            nn.init.zeros_(self.ctx_proj.weight)
            nn.init.zeros_(self.ctx_proj.bias)

    def forward(self, original: torch.Tensor, processed: torch.Tensor, ctx=None):
        x = torch.cat([original, processed], dim=1)
        logit = self.gate_net(x)
        if ctx is not None and hasattr(self, 'ctx_proj'):
            bias = self.ctx_proj(ctx)
            for _ in range(self.dims):
                bias = bias.unsqueeze(-1)
            logit = logit + bias
        gate = torch.sigmoid(logit)
        return gate * original + (1.0 - gate) * processed, gate


# ---------------------------------------------------------------------------
# CFF Variant 3: Gated Fusion + Axial Cross-Attention Refinement
# ---------------------------------------------------------------------------

class AxialCrossAttention(nn.Module):
    """Single-head axial cross-attention along D, H, W axes (3D) or H, W axes (2D)."""

    def __init__(self, channels: int, dims: int = 3):
        super().__init__()
        self.dims = dims
        self.q_proj = conv_nd(dims, channels, channels, kernel_size=1)
        self.k_proj = conv_nd(dims, channels, channels, kernel_size=1)
        self.v_proj = conv_nd(dims, channels, channels, kernel_size=1)
        self.o_proj = conv_nd(dims, channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    def _axial_attn_1d(self, q, k, v, axis=2):
        """1D cross-attention along a single spatial axis."""
        B, C = q.shape[:2]
        ndim = q.dim()
        spatial_axes = list(range(2, ndim))
        non_axis = [i for i in spatial_axes if i != axis]
        perm = [0, 1] + non_axis + [axis]
        q = q.permute(*perm).contiguous()
        k = k.permute(*perm).contiguous()
        v = v.permute(*perm).contiguous()

        *rest, L = q.shape[2:]
        q_f = q.reshape(B, C, -1, L)
        k_f = k.reshape(B, C, -1, L)
        v_f = v.reshape(B, C, -1, L)

        # Correct: contract over channel C and spatial N → [B, L, L]
        attn = torch.einsum('bcni,bcnj->bij', q_f, k_f) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bij,bcnj->bcni', attn, v_f)
        out = out.reshape(B, C, *rest, L)

        # Inverse permute
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return out.permute(*inv_perm).contiguous()

    def forward(self, fused, processed):
        q = self.q_proj(fused)
        k = self.k_proj(processed)
        v = self.v_proj(processed)
        out = 0
        ndim = fused.dim()
        axes = list(range(2, ndim))  # (2,3,4) for 3D; (2,3) for 2D
        for d in axes:
            out = out + self._axial_attn_1d(q, k, v, axis=d)
        return self.o_proj(out / len(axes))


class GatedCrossAttentionFusion(nn.Module):
    """
    g = σ(Conv1x1([F_s, F_w]))
    F_fused = g ⊙ F_s + (1 - g) ⊙ F_w
    F_out = F_fused + AxialCrossAttn(Q=F_fused, K=F_w, V=F_w)

    Axial cross-attention reduces spatial attention complexity from O(N²)
    to O(N(D+H+W)) for 3D or O(N(H+W)) for 2D by factorising along each
    spatial axis.
    """

    def __init__(self, channels: int, ctx_channels: int = 0, dims: int = 3):
        super().__init__()
        self.dims = dims
        self.gate = GatedContentFrequencyFusion(channels, ctx_channels, dims=dims)
        self.attn = AxialCrossAttention(channels, dims=dims)

    def forward(self, original: torch.Tensor, processed: torch.Tensor, ctx=None):
        fused, gate = self.gate(original, processed, ctx=ctx)
        refine = self.attn(fused, processed)
        return fused + refine, gate


# ---------------------------------------------------------------------------
# T-RAF: Tumor-aware Residual Adaptive Frequency enhancement
# ---------------------------------------------------------------------------

class TumorAwareFrequencyEnhancement(nn.Module):
    """
    Tumor-aware Residual Adaptive Frequency enhancement (T-RAF).

    Replaces LWFD with adaptive frequency enhancement controlled by a
    learned gate. Two modes:
      - 'base': full subband processing with adaptive gate (no prior)
      - 'prior': + tumor prior modulates high-frequency residual strength

    Initialized as identity (gamma=0).
    """

    def __init__(self, channels: int, mode: str = 'base', dims: int = 3):
        super().__init__()
        self.mode = mode
        self.dims = dims
        mid = max(channels // 4, 8)

        if dims == 3:
            mid_in_ch = 3 * channels
            high_in_ch = 4 * channels
        else:
            mid_in_ch = 2 * channels
            high_in_ch = channels

        self.conv_low = nn.Sequential(
            conv_nd(dims, channels, channels, 3, padding=1, groups=channels),
            conv_nd(dims, channels, channels, 1),
            instancenorm_nd(dims, channels, affine=True),
        )
        self.conv_mid = nn.Sequential(
            conv_nd(dims, mid_in_ch, mid, 3, padding=1, groups=min(mid_in_ch, mid)),
            conv_nd(dims, mid, channels, 1),
            instancenorm_nd(dims, channels, affine=True),
        )
        self.conv_high = nn.Sequential(
            conv_nd(dims, high_in_ch, mid, 3, padding=1, groups=min(high_in_ch, mid)),
            conv_nd(dims, mid, channels, 1),
            instancenorm_nd(dims, channels, affine=True),
        )

        for ll, name in [(self.conv_low, 'low'), (self.conv_mid, 'mid'), (self.conv_high, 'high')]:
            if isinstance(ll[-1], nn.InstanceNorm2d) or isinstance(ll[-1], nn.InstanceNorm3d):
                nn.init.constant_(ll[-1].weight, 1)
                nn.init.constant_(ll[-1].bias, 0)

        if dims == 3:
            self.conv_llh = conv_nd(dims, channels, channels, 3, padding=1, groups=channels, bias=False)
            self.conv_lhl = conv_nd(dims, channels, channels, 3, padding=1, groups=channels, bias=False)
            self.conv_hll = conv_nd(dims, channels, channels, 3, padding=1, groups=channels, bias=False)
            self.conv_lhh = conv_nd(dims, channels, channels, 3, padding=1, groups=channels, bias=False)
            self.conv_hlh = conv_nd(dims, channels, channels, 3, padding=1, groups=channels, bias=False)
            self.conv_hhl = conv_nd(dims, channels, channels, 3, padding=1, groups=channels, bias=False)
            self.conv_hhh = conv_nd(dims, channels, channels, 3, padding=1, groups=channels, bias=False)
        else:
            self.conv_lh = conv_nd(dims, channels, channels, 3, padding=1, groups=channels, bias=False)
            self.conv_hl = conv_nd(dims, channels, channels, 3, padding=1, groups=channels, bias=False)
            self.conv_hh = conv_nd(dims, channels, channels, 3, padding=1, groups=channels, bias=False)

        # Adaptive gate: controls which spatial locations get enhanced
        self.gate_net = nn.Sequential(
            conv_nd(dims, 3 * channels, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            conv_nd(dims, mid, 1, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

        if mode == 'prior':
            self.prior_proj = conv_nd(dims, 3 * channels + 1, 3 * channels, 1)
            nn.init.zeros_(self.prior_proj.weight)
            nn.init.zeros_(self.prior_proj.bias)

        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, tumor_prior: torch.Tensor = None):
        B, C = x.shape[:2]
        orig = x.shape[2:]

        pad = tuple(0 if s % 2 == 0 else 1 for s in orig)
        need_pad = any(p > 0 for p in pad)
        if need_pad:
            if self.dims == 3:
                x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
            else:
                x = F.pad(x, (0, pad[1], 0, pad[0]))

        if self.dims == 3:
            LLL, LLH, LHL, HLL, LHH, HLH, HHL, HHH = dwt_haar_nd(3, x)

            f_low = self.conv_low(LLL)
            f_mid = self.conv_mid(torch.cat([LLH, LHL, HLL], dim=1))
            f_high = self.conv_high(torch.cat([LHH, HLH, HHL, HHH], dim=1))

            f_llh = self.conv_llh(LLH)
            f_lhl = self.conv_lhl(LHL)
            f_hll = self.conv_hll(HLL)
            f_lhh = self.conv_lhh(LHH)
            f_hlh = self.conv_hlh(HLH)
            f_hhl = self.conv_hhl(HHL)
            f_hhh = self.conv_hhh(HHH)

            gate_in = torch.cat([f_low, f_mid, f_high], dim=1)
            if self.mode == 'prior' and tumor_prior is not None:
                prior_d = interpolate_nd(3, tumor_prior, size=gate_in.shape[2:])
                gate_in = self.prior_proj(torch.cat([gate_in, prior_d], dim=1))

            gate = torch.sigmoid(self.gate_net(gate_in))

            out = idwt_haar_nd(3,
                f_low,
                gate * f_llh, gate * f_lhl, gate * f_hll,
                gate * f_lhh, gate * f_hlh, gate * f_hhl, gate * f_hhh,
            )
        else:
            LL, LH, HL, HH = dwt_haar_nd(2, x)

            f_low = self.conv_low(LL)
            f_mid = self.conv_mid(torch.cat([LH, HL], dim=1))
            f_high = self.conv_high(HH)

            f_lh = self.conv_lh(LH)
            f_hl = self.conv_hl(HL)
            f_hh = self.conv_hh(HH)

            gate_in = torch.cat([f_low, f_mid, f_high], dim=1)
            if self.mode == 'prior' and tumor_prior is not None:
                prior_d = interpolate_nd(2, tumor_prior, size=gate_in.shape[2:])
                gate_in = self.prior_proj(torch.cat([gate_in, prior_d], dim=1))

            gate = torch.sigmoid(self.gate_net(gate_in))

            out = idwt_haar_nd(2, f_low, gate * f_lh, gate * f_hl, gate * f_hh)

        if need_pad:
            if self.dims == 3:
                out = out[:, :, :orig[0], :orig[1], :orig[2]]
            else:
                out = out[:, :, :orig[0], :orig[1]]

        g = torch.tanh(self.residual_gate)
        if self.dims == 3:
            x_trim = x[:, :, :orig[0], :orig[1], :orig[2]]
        else:
            x_trim = x[:, :, :orig[0], :orig[1]]
        return x_trim + g * out, gate


# ---------------------------------------------------------------------------
# LA-ACG: Liver-Aware Anatomical Context Gating
# ---------------------------------------------------------------------------

class LiverPriorBranch(nn.Module):
    """
    Lightweight liver prior prediction from multi-scale encoder features.

    Predicts liver probability map [B, 1, D, H, W] used as spatial gate
    for decoder stages and tumor logit modulation.

    Designed to be trained jointly with the main segmentation head.
    """

    def __init__(self, enc_channels: list, dims: int = 3):
        super().__init__()
        self.dims = dims
        self.proj_shallow = conv_nd(dims, enc_channels[0], 16, 1)
        self.proj_mid = conv_nd(dims, enc_channels[1], 16, 1)
        self.proj_bn = conv_nd(dims, enc_channels[-1], 16, 1)

        self.fuser = nn.Sequential(
            conv_nd(dims, 48, 16, 3, padding=1),
            instancenorm_nd(dims, 16, affine=True),
            nn.ReLU(inplace=True),
            conv_nd(dims, 16, 1, 1),
        )

    def forward(self, enc_skips):
        e0 = self.proj_shallow(enc_skips[0])
        e1 = self.proj_mid(enc_skips[1])
        eb = self.proj_bn(enc_skips[-1])

        target_size = e0.shape[2:]
        e1 = interpolate_nd(self.dims, e1, size=target_size)
        eb = interpolate_nd(self.dims, eb, size=target_size)

        fused = torch.cat([e0, e1, eb], dim=1)
        return self.fuser(fused)


class LiverAwareContextGating(nn.Module):
    """
    Liver-Aware Anatomical Context Gating (LA-ACG).

    Uses a liver prior probability map to spatially modulate decoder features.
    Replaces the channel-only ACG with a spatial liver prior.

    Each decoder stage receives:
        x_s = x_s * (1 + alpha_s * prior_s)
    where prior_s is the liver prior interpolated to the decoder stage's size.

    Modulation is applied directly in PhaserUNet._decoder_forward.
    This class only stores the learnable per-stage alpha parameters.
    """

    def __init__(self, decoder_channels: list):
        super().__init__()
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in decoder_channels
        ])


# ---------------------------------------------------------------------------
# Small-Tumor Detail Head
# ---------------------------------------------------------------------------

class SmallTumorDetailHead(nn.Module):
    """
    High-resolution auxiliary head for small tumor detection (binary).

    Uses shallow encoder features (e0, e1) to produce a tumor-vs-background
    binary logit map at high resolution, preserving small lesion details
    that may be lost in the deep decoder path.

    Only used during training; output is [B, 2, D, H, W] (BG, tumor).
    """

    def __init__(self, enc_channels: list, num_classes: int = 2, dims: int = 3):
        super().__init__()
        self.dims = dims
        self.proj_e0 = conv_nd(dims, enc_channels[0], 32, 3, padding=1)
        self.proj_e1 = conv_nd(dims, enc_channels[1], 32, 3, padding=1)

        total = 64
        self.head = nn.Sequential(
            conv_nd(dims, total, 32, 3, padding=1),
            instancenorm_nd(dims, 32, affine=True),
            nn.ReLU(inplace=True),
            conv_nd(dims, 32, 2, 1),
        )

    def forward(self, enc_skips):
        e0 = self.proj_e0(enc_skips[0])
        e1 = self.proj_e1(enc_skips[1])
        e1 = interpolate_nd(self.dims, e1, size=e0.shape[2:])
        return self.head(torch.cat([e0, e1], dim=1))
