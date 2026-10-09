"""
PhaserUNet: Wraps a PlainConvUNet with PhASER modules injected between
encoder and decoder.

Design principle:
  PlainConvUNet.forward(x):
      skips = self.encoder(x)     # [e0, e1, ..., e_N_bottleneck]
      return self.decoder(skips)

  PhaserUNet intercepts the skips list to apply LWFD and PCSF, and
  optionally injects ACG modulation into the decoder.

All modules are initialised as identity (output≈input) so that training
starts from the VoCo pretrained baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .modules import (
    PhaseConditionedSkipFusion,
    PhaseCongruencySkipFusion,
    PhaseFrequencyCoherenceRouting,
    LearnableWaveletFeatureDecomposition,
    AnatomicalContextGating,
    ContentFrequencyFusion,
    GatedContentFrequencyFusion,
    GatedCrossAttentionFusion,
    TumorAwareFrequencyEnhancement,
    LiverPriorBranch,
    LiverAwareContextGating,
    SmallTumorDetailHead,
)
from .dim_utils import spatial_mean, interpolate_nd, interp_mode


class PhaserUNet(nn.Module):
    """
    VoCo-pretrained PlainConvUNet augmented with PhASER frequency modules.

    Module insertion points:
      - LWFD: after each encoder stage output (excl. bottleneck)
      - PCSF: on each skip connection to the decoder
      - ACG:  bottleneck → per-decoder-stage channel modulation

    Args:
        base_model: nnUNet PlainConvUNet instance (already built + weights loaded)
        feature_channels: list of encoder output channel counts
                          e.g. [32, 64, 128, 256, 320, 320] for 6-stage UNet
        enable_pcsf: insert skip boundary-gating modules
        pcsf_mode:   'grad' = gradient magnitude (PhaseConditionedSkipFusion),
                     'pc'   = monogenic phase congruency (PhaseCongruencySkipFusion)
        pc_scales:   number of learnable log-Gabor scales when pcsf_mode='pc'
        pc_max_voxels: FFT cost cap per skip when pcsf_mode='pc'; above it the
                     1-channel proxy is downsampled by powers of two and the PC
                     map upsampled back
        pcsf_gate_sens: initial gate sensitivity (alpha) shared by BOTH pcsf
                     modes so the grad-vs-pc ablation is matched. Must be
                     nonzero or the 'pc' filters get zero gradient.
        enable_lwfd: insert LearnableWaveletFeatureDecomposition modules
        enable_acg:  insert AnatomicalContextGating
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_channels: List[int],
        spatial_dims: int = 3,
        enable_pcsf: bool = False,
        pcsf_mode: str = 'grad',
        pc_scales: int = 3,
        pc_max_voxels: int = 1 << 20,
        pcsf_gate_sens: float = 0.5,
        enable_pfcr: bool = False,
        pfcr_use_prior: bool = False,
        enable_lwfd: bool = False,
        enable_acg: bool = False,
        fusion_mode: str = 'gate',
        lwfd_mode: str = 'full',
        t_raf_mode: str = 'none',
        enable_la_acg: bool = False,
        enable_small_head: bool = False,
        num_classes: int = 3,
        pfcr_shuffle_pc: bool = False,
        pfcr_intervene: str = 'none',
        pfcr_intervene_levels=None,
        legacy: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.dims = spatial_dims
        # legacy=True reproduces pre-V2 inference (double-sigmoid prior, PC
        # noise scale from scale 0) for old checkpoints such as Copy2.
        self.legacy = legacy

        # Encoder and decoder references
        self.encoder = base_model.encoder
        self._decoder = base_model.decoder

        # Intermediate feature storage
        self._last_enc_skips = None
        self._last_dec_features = None
        self._last_gates = None
        self._last_liver_prior = None
        self._last_tumor_prior = None
        self._last_small_logits = None
        self._last_pc_maps = []
        self._last_pfcr_routes = []
        # Number of deepest decoder scales whose features keep gradient for the
        # spectral consistency loss. Must match max_scales in
        # spectral_consistency_loss (default 3). Deepest scales are smallest in
        # spatial size, so retaining their graph is cheap on memory.
        self._spectral_grad_scales = 3

        n_skips = len(feature_channels) - 1

        # --- LWFD (original, mutually exclusive with T-RAF) ---
        if t_raf_mode != 'none' and enable_lwfd:
            raise ValueError(
                "T-RAF and LWFD are mutually exclusive. "
                f"Got t_raf_mode={t_raf_mode}, enable_lwfd={enable_lwfd}"
            )
        self.enable_lwfd = enable_lwfd
        if enable_lwfd:
            self.lwfd_modules = nn.ModuleList([
                LearnableWaveletFeatureDecomposition(feature_channels[i], mode=lwfd_mode, dims=self.dims)
                for i in range(n_skips)
            ])

        # --- T-RAF: tumor-aware frequency enhancement ---
        self.t_raf_mode = t_raf_mode
        if t_raf_mode != 'none':
            self.t_raf_modules = nn.ModuleList([
                TumorAwareFrequencyEnhancement(feature_channels[i], mode=t_raf_mode, dims=self.dims)
                for i in range(n_skips)
            ])

        # --- PCSF: modulate skip features ---
        # Two boundary-evidence modes, directly ablatable against each other:
        #   'grad' -> PhaseConditionedSkipFusion, finite-difference |∇f|
        #             (amplitude-based; the original module, untouched)
        #   'pc'   -> PhaseCongruencySkipFusion, monogenic phase congruency
        #             (contrast-invariant; targets isoattenuating lesions)
        self.enable_pcsf = enable_pcsf
        self.pcsf_mode = pcsf_mode
        if enable_pcsf:
            if pcsf_mode == 'grad':
                self.pcsf_modules = nn.ModuleList([
                    PhaseConditionedSkipFusion(
                        feature_channels[i], dims=self.dims,
                        gate_sens_init=pcsf_gate_sens,
                    )
                    for i in range(n_skips)
                ])
            elif pcsf_mode == 'pc':
                self.pcsf_modules = nn.ModuleList([
                    PhaseCongruencySkipFusion(
                        feature_channels[i], dims=self.dims,
                        n_scales=pc_scales, pc_max_voxels=pc_max_voxels,
                        gate_sens_init=pcsf_gate_sens,
                    )
                    for i in range(n_skips)
                ])
            else:
                raise ValueError(
                    f"Unknown pcsf_mode: {pcsf_mode!r}. Expected 'grad' or 'pc'."
                )

        # --- PFCR: phase-conditioned frequency routing ---
        self.enable_pfcr = enable_pfcr
        self.pfcr_use_prior = pfcr_use_prior
        if enable_pfcr:
            # Inference-time interventions (E11) apply to the listed skip levels
            # (0 = shallowest); None means all levels.
            levels = set(range(n_skips)) if pfcr_intervene_levels is None \
                else set(pfcr_intervene_levels)
            self.pfcr_modules = nn.ModuleList([
                PhaseFrequencyCoherenceRouting(
                    feature_channels[i], dims=self.dims, n_scales=pc_scales,
                    pc_max_voxels=pc_max_voxels, use_prior=pfcr_use_prior,
                    shuffle_pc=pfcr_shuffle_pc,
                    intervene=pfcr_intervene if i in levels else 'none',
                    legacy=legacy,
                )
                for i in range(n_skips)
            ])

        # --- CFF: fuse original encoder features with enhanced ---
        self.fusion_mode = fusion_mode
        has_enhance = (enable_lwfd or enable_pcsf or t_raf_mode != 'none')
        if has_enhance and fusion_mode != 'none':
            if fusion_mode == 'concat':
                cff_cls = ContentFrequencyFusion
            elif fusion_mode == 'gate':
                cff_cls = GatedContentFrequencyFusion
            elif fusion_mode == 'attn':
                cff_cls = GatedCrossAttentionFusion
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
            self.cff_modules = nn.ModuleList([
                cff_cls(feature_channels[i], ctx_channels=feature_channels[-1], dims=self.dims)
                for i in range(n_skips)
            ])
        else:
            self.fusion_mode = 'none'

        # --- ACG (original GAP-based) ---
        self.enable_acg = enable_acg
        if enable_acg:
            decoder_chs = [feature_channels[-(i + 2)] for i in range(n_skips)]
            self.acg = AnatomicalContextGating(feature_channels[-1], decoder_chs, dims=self.dims)

        # --- LA-ACG (new: liver-aware spatial gating) ---
        self.enable_la_acg = enable_la_acg
        if enable_la_acg:
            self.liver_prior_branch = LiverPriorBranch(feature_channels, dims=self.dims)
            self.la_acg = LiverAwareContextGating(
                [feature_channels[-(i + 2)] for i in range(n_skips)]
            )

        # --- Small-Tumor Detail Head ---
        self.enable_small_head = enable_small_head
        if enable_small_head:
            self.small_head = SmallTumorDetailHead(feature_channels, num_classes, dims=self.dims)

    # ------------------------------------------------------------------
    # Property forwarding for nnUNet compatibility
    # ------------------------------------------------------------------
    @property
    def decoder(self):
        """Expose decoder for nnUNet deep_supervision toggling etc."""
        return self._decoder

    @property
    def do_ds(self):
        """Whether deep supervision is active."""
        return self._decoder.deep_supervision

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        # Clear per-forward auxiliary caches before computing new features.  The
        # trainer consumes these tensors immediately after forward() to compute
        # PFCR budget and PC-boundary losses, so they must remain available until
        # the next forward pass.
        self._last_pc_maps = []
        self._last_pfcr_routes = []

        # 1. Encoder forward → skips list (native 4D/5D from PlainConvUNet)
        skips = list(self.encoder(x))

        # Store raw encoder skip features (before processing) for spectral loss
        self._last_enc_skips = [s.detach() for s in skips[:-1]]

        # Save original encoder features for CFF
        if self.fusion_mode != 'none':
            orig_skips = [s.clone() for s in skips[:-1]]

        # 2. Generate tumor prior from shallow features (before T-RAF).
        small_logits = None
        tumor_prior = None
        _prior_mode = (
            self.t_raf_mode in ('prior',)
            or (self.enable_pfcr and self.pfcr_use_prior)
        )
        compute_small = self.enable_small_head and (self.training or _prior_mode)
        if compute_small:
            small_logits = self.small_head(skips)
            if self.legacy:
                tumor_prior = torch.sigmoid(small_logits[:, 1:2])
            else:
                # The head is trained with 2-class CE, so the tumor probability
                # is the softmax over its two logits.
                tumor_prior = torch.softmax(small_logits.float(), dim=1)[:, 1:2].to(small_logits.dtype)
        elif _prior_mode:
            small_logits = None

        self._last_tumor_prior = tumor_prior
        self._last_small_logits = small_logits if self.training else None

        # 3. T-RAF: tumor-aware frequency enhancement
        if self.t_raf_mode != 'none':
            for i in range(len(skips) - 1):
                result = self.t_raf_modules[i](skips[i], tumor_prior)
                if isinstance(result, tuple):
                    skips[i], _ = result
                else:
                    skips[i] = result

        # 4. LWFD: wavelet enhancement
        if self.enable_lwfd:
            for i in range(len(skips) - 1):
                skips[i] = self.lwfd_modules[i](skips[i])

        # 5. PFCR: phase-conditioned frequency routing
        self._last_pfcr_routes = []
        if self.enable_pfcr:
            for i in range(len(skips) - 1):
                skips[i], route = self.pfcr_modules[i](skips[i], tumor_prior)
                self._last_pfcr_routes.append(route)
            # PFCR owns the phase evidence used for its frequency routing. Expose
            # the underlying PC maps through the common auxiliary-loss interface.
            self._last_pc_maps = [m._last_pc for m in self.pfcr_modules]

        # 6. PCSF: optional gradient/phase skip gating baseline.
        if self.enable_pcsf:
            for i in range(len(skips) - 1):
                skips[i] = self.pcsf_modules[i](skips[i])
            if self.pcsf_mode == 'pc':
                self._last_pc_maps = [m._last_pc for m in self.pcsf_modules]

        # 6. LA-ACG: liver prior prediction
        if self.enable_la_acg:
            self._last_liver_prior = self.liver_prior_branch(skips)

        # 7. CFF: fuse original and enhanced features
        gates = []
        if self.fusion_mode != 'none':
            ctx = spatial_mean(skips[-1])
            for i in range(len(skips) - 1):
                skips[i], gate_i = self.cff_modules[i](orig_skips[i], skips[i], ctx=ctx)
                gates.append(gate_i)
        self._last_gates = gates

        # 8. Decoder forward
        if self.enable_acg and not self.enable_la_acg:
            acg_vectors = self.acg(skips[-1])
            return self._decoder_forward_with_acg(skips, acg_vectors)
        else:
            return self._decoder_forward(skips)

    # ------------------------------------------------------------------
    # Decoder forward (with optional LA-ACG modulation)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Decoder forward (with optional LA-ACG modulation)
    # ------------------------------------------------------------------
    def _decoder_forward(self, skips: list):
        dec = self._decoder
        lres_input = skips[-1]
        seg_outputs = []
        dec_features = []

        for s in range(len(dec.stages)):
            x = dec.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = dec.stages[s](x)

            # LA-ACG: liver-aware spatial modulation at each decoder stage
            if self.enable_la_acg:
                prior = interpolate_nd(self.dims,
                    torch.sigmoid(self._last_liver_prior), size=x.shape[2:])
                alpha = torch.tanh(self.la_acg.alphas[s])
                x = x * (1.0 + alpha * prior)

            if self.training and s < self._spectral_grad_scales:
                dec_features.append(x)
            else:
                dec_features.append(x.detach())

            if dec.deep_supervision:
                seg_outputs.append(dec.seg_layers[s](x))
            elif s == (len(dec.stages) - 1):
                seg_outputs.append(dec.seg_layers[-1](x))

            lres_input = x

        self._last_dec_features = dec_features[::-1]
        seg_outputs = seg_outputs[::-1]

        if not dec.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

    # ------------------------------------------------------------------
    # Decoder forward with ACG modulation injected
    # ------------------------------------------------------------------
    def _decoder_forward_with_acg(
        self, skips: list, acg_maps: List[torch.Tensor]
    ):
        dec = self._decoder
        lres_input = skips[-1]
        seg_outputs = []
        dec_features = []

        for s in range(len(dec.stages)):
            x = dec.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = dec.stages[s](x)

            # Spatial ACG: upsample modulation map to current resolution
            mod = acg_maps[s]
            if mod.shape[2:] != x.shape[2:]:
                mod = interpolate_nd(self.dims, mod, size=x.shape[2:])
            x = x * (1.0 + mod)

            if self.training and s < self._spectral_grad_scales:
                dec_features.append(x)
            else:
                dec_features.append(x.detach())

            if dec.deep_supervision:
                seg_outputs.append(dec.seg_layers[s](x))
            elif s == (len(dec.stages) - 1):
                seg_outputs.append(dec.seg_layers[-1](x))

            lres_input = x

        self._last_dec_features = dec_features[::-1]
        seg_outputs = seg_outputs[::-1]

        if not dec.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------
    def get_phaser_params(self):
        """Collect all PhASER-specific parameters for differential LR."""
        params = []
        if self.enable_lwfd:
            params.extend(self.lwfd_modules.parameters())
        if self.t_raf_mode != 'none':
            for m in self.t_raf_modules:
                params.extend(m.parameters())
        if self.enable_pcsf:
            params.extend(self.pcsf_modules.parameters())
        if self.enable_pfcr:
            params.extend(self.pfcr_modules.parameters())
        if self.enable_acg:
            params.extend(self.acg.parameters())
        if self.enable_la_acg:
            params.extend(self.liver_prior_branch.parameters())
            params.extend(self.la_acg.parameters())
        if self.fusion_mode != 'none':
            params.extend(self.cff_modules.parameters())
        if self.enable_small_head:
            params.extend(self.small_head.parameters())
        return params


def load_pretrained_weights(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    Load VoComni_nnunet.pt weights into a PlainConvUNet with
    shape-matching selective loading.

    Handles:
      - state_dict / network_weights top-level keys
      - module. prefix (DataParallel)
      - Shape mismatch (e.g. different num_classes): mismatched keys
        keep their random initialisation
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'network_weights' in ckpt:
        state_dict = ckpt['network_weights']
    else:
        state_dict = ckpt

    # Strip 'module.' prefix (DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        for key in list(state_dict.keys()):
            state_dict[key[7:]] = state_dict.pop(key)

    current = model.state_dict()

    # Selective load: only transfer matching keys with matching shapes
    new_state = {}
    matched = 0
    total = 0
    for k, v in current.items():
        total += 1
        if k in state_dict and state_dict[k].shape == v.shape:
            new_state[k] = state_dict[k]
            matched += 1
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=True)

    # Log
    try:
        import logging
        logging.getLogger('nnunet').info(
            f'Pretrained weights: {matched}/{total} keys loaded')
    except Exception:
        print(f'Pretrained weights: {matched}/{total} keys loaded')

    return model
