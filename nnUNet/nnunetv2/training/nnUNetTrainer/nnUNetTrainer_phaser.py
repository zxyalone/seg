"""
PhASER Trainer for nnUNet.

Supports ablation studies via environment variables:
  PHASER_PCSF=1        → skip boundary gating
  PHASER_PCSF_MODE=grad|pc
                       → boundary evidence: 'grad' = finite-difference |∇f|
                         (amplitude-based, original), 'pc' = monogenic phase
                         congruency (contrast-invariant, for isoattenuating
                         lesions). Single-variable ablation.
  PHASER_PC_SCALES=3   → learnable log-Gabor scales when PCSF_MODE=pc
  PHASER_PC_MAX_VOXELS → FFT cost cap per skip (default 2^20).
                         NOTE: 2^20 is BELOW 128^3, so the shallowest skip --
                         the only one that resolves small lesions -- is computed
                         at 64^3 and interpolated back. This is NOT purely a
                         cost device: the 2x average acts as a denoiser and was
                         measured to IMPROVE boundary localisation (synthetic
                         slab AUC at CNR 2/4: 1.00/1.00 capped vs 0.50/0.52
                         uncapped). On a real patch the uncapped version scored
                         better on boundary/peri ratio (0.62 vs 0.37), so the
                         evidence is mixed -- the default is kept as-is and the
                         knob is exposed as a one-variable ablation.
  PHASER_PC_SPACING    → mm per tensor axis for the log-Gabor bank. Default
                         EMPTY = voxel-isotropic grid (d = 1), i.e. unchanged
                         behaviour. 'plan' uses the plan spacing; or give
                         explicit values. Enabling it is physically more correct
                         but rescales the bank by up to 1.28x and showed no
                         clear win on one real patch, so it is an ablation
                         variable, not a default.
  PHASER_PC_NOISE_KAPPA → optionally cap the Kovesi threshold at this
                         fraction of the local multi-scale amplitude sum.
                         DEFAULT 0 = pure Kovesi (absolute) threshold.
                         Values > 0 were measured to LOCALISE WORSE (see the
                         PhaseCongruency docstring and out/s45.log); they are
                         kept only so the experiment can be reproduced.
  PHASER_PCSF_GATE_SENS → initial gate sensitivity alpha, shared by both PCSF
                         modes (default 0.5). MUST be nonzero for PCSF_MODE=pc:
                         alpha=0 zeroes d(attention)/d(PC) and the log-Gabor
                         filters never train. 0.0 = pre-2026-09 behaviour.
  PHASER_L_PC_BOUNDARY=1 → PC boundary ranking loss (requires PCSF_MODE=pc).
                         Active from epoch 0, NOT deferred to Phase B, because
                         it exists to un-throttle the log-Gabor filters early.
                         PHASER_PC_BOUNDARY_WEIGHT (default 0.2)
                         PHASER_PC_BOUNDARY_MARGIN (default 0.2)
                         PHASER_PC_BOUNDARY_WARMUP (default 5 epochs)
  PHASER_LWFD=1        → Learnable Wavelet Feature Decomposition
  PHASER_ACG=1         → Anatomical Context Gating
  PHASER_L_SPECTRAL=1  → Spectral consistency loss
  PHASER_L_BOUNDARY=1  → Boundary-aware Dice loss

  PHASER_BOUNDARY_WEIGHT=0.1   (default)
  PHASER_SPECTRAL_WEIGHT=0.05  (default)
  PHASER_FT_WEIGHT=0.3         Focal Tversky weight (default)

Usage:
  # Baseline
  nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr nnUNetTrainer_phaser

  # Full PhASER
  PHASER_PCSF=1 PHASER_LWFD=1 PHASER_ACG=1 PHASER_L_SPECTRAL=1 \
      nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr nnUNetTrainer_phaser
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.focal_tversky import MultiClassFocalTverskyLoss
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.helpers import dummy_context

from torch.optim.lr_scheduler import _LRScheduler

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

from .phaser.unet import PhaserUNet, load_pretrained_weights
from .phaser.losses import boundary_dice_loss, spectral_consistency_loss, \
    compute_lesion_importance_map, lesion_weighted_loss, \
    gate_supervision_loss, inside_liver_penalty_loss, boundary_weighted_dice_loss, \
    pc_boundary_ranking_loss, phase_frequency_budget_loss


class GroupPolyLRScheduler(_LRScheduler):
    """
    Polynomial LR decay that PRESERVES per-group base LR ratios.

    The stock PolyLRScheduler overwrites every param_group's lr with a single
    value derived from one initial_lr, which silently destroys the differential
    LR configured in configure_optimizers (encoder 0.3x / decoder 0.5x /
    phaser 1.0x). This scheduler snapshots each group's initial lr and scales
    them all by the same poly factor, keeping the ratios intact.
    """

    def __init__(self, optimizer, max_steps: int, exponent: float = 0.9):
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        # Snapshot each group's base lr (the 0.3x / 0.5x / 1.0x absolute values)
        self.base_lrs_per_group = [g['lr'] for g in optimizer.param_groups]
        super().__init__(optimizer, -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        factor = (1 - current_step / self.max_steps) ** self.exponent
        for g, base in zip(self.optimizer.param_groups, self.base_lrs_per_group):
            g['lr'] = base * factor


class CombinedDiceCEFocalTverskyLoss(nn.Module):
    """DiceCE + Focal Tversky combined loss."""
    def __init__(self, base_loss: nn.Module, ft_loss: nn.Module, ft_weight: float = 0.3):
        super().__init__()
        self.base_loss = base_loss
        self.ft_loss = ft_loss
        self.ft_weight = ft_weight

    def forward(self, net_output, target):
        return self.base_loss(net_output, target) + self.ft_weight * self.ft_loss(net_output, target)


def resolve_pc_spacing(configuration_manager, dims: int):
    """Physical voxel spacing (mm per tensor axis) for the phase-congruency bank.

    PHASER_PC_SPACING values:
      ''            -> None, i.e. the log-Gabor grid stays in cycles/voxel
                       (d = 1). This is the DEFAULT and preserves the behaviour
                       every existing checkpoint was trained under.
      'plan'/'auto' -> use configuration_manager.spacing (nnU-Net array-axis
                       order; LiTS is [1.0, 0.779, 0.779] over 486x512x512).
      'a,b,c'       -> explicit mm per axis; a single value is broadcast.

    Why it is not on by default: switching to physical spacing reinterprets the
    learnable rho (cycles/mm rather than cycles/voxel) and rescales the effective
    filter bank by up to 1.28x on LiTS, so it changes operator behaviour rather
    than purely fixing a bug. Measured on one real validation patch, enabling it
    moved the boundary/peri-liver PC ratio from 0.616 to 0.405 -- i.e. no clear
    win -- so it is exposed as a one-variable ablation instead of a silent
    default change. See step2_mechanism/out/s45.log.
    """
    raw = os.environ.get('PHASER_PC_SPACING', '').strip()
    if raw.lower() in ('plan', 'auto'):
        cm_spacing = getattr(configuration_manager, 'spacing', None)
        if not cm_spacing:
            return None
        vals = [float(v) for v in cm_spacing]
    elif raw:
        vals = [float(v) for v in raw.replace(';', ',').split(',') if v.strip()]
    else:
        return None
    if len(vals) == 1:
        vals = vals * dims
    if len(vals) != dims:
        raise ValueError(
            f"PHASER_PC_SPACING has {len(vals)} entries but the configuration is "
            f"{dims}D; supply one value or one per axis.")
    if any(v <= 0 for v in vals):
        raise ValueError(f"PHASER_PC_SPACING must be positive, got {vals}")
    return tuple(vals)


class nnUNetTrainer_phaser(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Disable AMP for diagnosis/workstations where rare FP16 overflow creates
        # non-finite losses. Default remains enabled for normal CUDA training.
        self.use_amp = os.environ.get('PHASER_AMP', '1') == '1'

        # --- Module switches ---
        self.enable_pcsf = os.environ.get('PHASER_PCSF', '0') == '1'
        # PCSF boundary evidence: 'grad' = finite-difference |∇f| (original),
        # 'pc' = monogenic phase congruency (contrast-invariant). The two are
        # matched in gate parameterisation and initialisation so PHASER_PCSF_MODE
        # is a clean single-variable ablation.
        self.pcsf_mode = os.environ.get('PHASER_PCSF_MODE', 'grad')
        if self.pcsf_mode not in ('grad', 'pc'):
            raise ValueError(
                f"PHASER_PCSF_MODE must be 'grad' or 'pc', got {self.pcsf_mode!r}"
            )
        self.pc_scales = int(os.environ.get('PHASER_PC_SCALES', '3'))
        # FFT cost cap per skip. Was 2^20, which is BELOW 128^3 = 2.10M, so the
        # shallowest skip -- the only scale that resolves small lesions -- ran
        # silently at 64^3 and was interpolated back up. Measured against no
        # cap: 49-57% of voxels changed by >1e-3, rank correlation 0.45-0.51.
        # 2^24 covers 256^3 and never triggers for nnU-Net patches.
        self.pc_max_voxels = int(os.environ.get('PHASER_PC_MAX_VOXELS', str(1 << 20)))
        # Physical spacing for the log-Gabor bank. Defaults to the plan spacing
        # so the bank is isotropic in mm instead of in voxels.
        self.pc_spacing = resolve_pc_spacing(self.configuration_manager, len(
            self.configuration_manager.pool_op_kernel_sizes[0]))
        # Relative cap on the Kovesi noise threshold. 0 restores the
        # pre-2026-09 absolute-threshold behaviour.
        self.pc_noise_rel_kappa = float(os.environ.get('PHASER_PC_NOISE_KAPPA', '0.0'))
        # Shared by both PCSF modes to keep the ablation matched. Must be nonzero
        # (see PhaseCongruencySkipFusion); 0.0 restores pre-2026-09 behaviour.
        self.pcsf_gate_sens = float(os.environ.get('PHASER_PCSF_GATE_SENS', '0.5'))
        # PC boundary ranking supervision. Only meaningful with PCSF_MODE=pc.
        self.enable_pc_boundary_loss = os.environ.get('PHASER_L_PC_BOUNDARY', '0') == '1'
        # Default 0.2, not the 0.05 used by the other aux losses. Measured on a
        # synthetic low-contrast lesion with distractor edges: at init the
        # boundary-minus-surround PC gap is NEGATIVE (-0.206), i.e. PC responds
        # more to lesion interior than to its margin. w=0.05 only lifts it to
        # -0.027 (still inverted) after 300 steps; w=0.2 flips it to +0.024 and
        # retunes the centre frequencies from [.333,.159,.076] to [.350,.204,.023].
        self.pc_boundary_weight = float(os.environ.get('PHASER_PC_BOUNDARY_WEIGHT', '0.2'))
        self.pc_boundary_margin = float(os.environ.get('PHASER_PC_BOUNDARY_MARGIN', '0.2'))
        # Unlike every other auxiliary loss this one is NOT deferred to Phase B.
        # Its whole purpose is to un-throttle the log-Gabor filters, which are
        # scaled by tanh(gamma)~0 early in training; starting it at epoch 60
        # would leave them frozen for the first 60 epochs. Short 5-epoch ramp
        # instead of the global 50-epoch warmup, so it is active from epoch 0.
        self.pc_boundary_warmup = int(os.environ.get('PHASER_PC_BOUNDARY_WARMUP', '5'))
        self.enable_pfcr = os.environ.get('PHASER_PFCR', '0') == '1'
        self.pfcr_use_prior = os.environ.get('PHASER_PFCR_USE_PRIOR', '0') == '1'
        self.enable_pfcr_budget = os.environ.get('PHASER_L_PFCR_BUDGET', '0') == '1'
        self.pfcr_budget_weight = float(os.environ.get('PHASER_PFCR_BUDGET_WEIGHT', '0.05'))
        self.enable_lwfd = os.environ.get('PHASER_LWFD', '0') == '1'
        self.enable_acg = os.environ.get('PHASER_ACG', '0') == '1'
        self.t_raf_mode = os.environ.get('PHASER_T_RAF', 'none')
        self.enable_la_acg = os.environ.get('PHASER_LA_ACG', '0') == '1'
        self.enable_small_head = os.environ.get('PHASER_SMALL_HEAD', '0') == '1'
        if self.pfcr_use_prior:
            self.enable_small_head = True
        # Tier-1 frequency-paper ablation modules (new, independent switches)
        # T-RAF prior mode requires small head (raises ValueError in build_network if disabled)
        if self.t_raf_mode in ('prior',) and not self.enable_small_head:
            self.enable_small_head = True
        self.enable_gate_loss = os.environ.get('PHASER_L_GATE', '0') == '1'
        self.enable_inside_penalty = os.environ.get('PHASER_L_INSIDE', '0') == '1'
        self.enable_boundary_w_loss = os.environ.get('PHASER_L_BOUNDARY_W', '0') == '1'
        self.enable_spectral_loss = os.environ.get('PHASER_L_SPECTRAL', '0') == '1'
        self.enable_boundary_loss = os.environ.get('PHASER_L_BOUNDARY', '0') == '1'
        self.enable_lesion_weight = os.environ.get('PHASER_ENABLE_LESION_WEIGHT', '0') == '1'
        # CFF fusion mode: 'none' | 'concat' | 'gate' | 'attn'
        self.fusion_mode = os.environ.get('PHASER_FUSION', 'none')
        valid_fusion = ('none', 'concat', 'gate', 'attn')
        if self.fusion_mode not in valid_fusion:
            self.fusion_mode = 'none'

        # --- Loss weights ---
        self.boundary_weight = float(os.environ.get('PHASER_BOUNDARY_WEIGHT', '0.05'))
        self.spectral_weight = float(os.environ.get('PHASER_SPECTRAL_WEIGHT', '0.05'))
        self.lesion_weight = float(os.environ.get('PHASER_LESION_WEIGHT', '0.2'))
        self.ft_weight = float(os.environ.get('PHASER_FT_WEIGHT', '0.15'))
        self.gate_loss_weight = float(os.environ.get('PHASER_GATE_LOSS_WEIGHT', '0.03'))
        self.inside_penalty_weight = float(os.environ.get('PHASER_INSIDE_WEIGHT', '0.02'))
        self.boundary_w_weight = float(os.environ.get('PHASER_BOUNDARY_W_WEIGHT', '0.05'))

        # --- Small-tumour head supervision schedule -----------------------
        # The head is a *conditioning input* to PFCR, so it must be trained from
        # the start of training, not from Phase B. Before the fix the head loss
        # only ran when epoch >= phase2_start (60), which left the tumour prior
        # at its random initialisation for the whole of Phase A -- and Copy2
        # (epoch 48) is exactly such a checkpoint. Default now switches on at
        # epoch 0 with its own short warmup.
        self.small_head_start = int(os.environ.get('PHASER_SMALL_HEAD_START', '0'))
        self.small_head_warmup = int(os.environ.get('PHASER_SMALL_HEAD_WARMUP', '10'))
        self.small_head_weight = float(
            os.environ.get('PHASER_SMALL_HEAD_WEIGHT', '0.1'))

        # --- Focal Tversky asymmetry --------------------------------------
        # alpha/beta are exposed so the symmetric setting (0.5/0.5) that removes
        # the FN-vs-FP bias is a single-variable change. Defaults reproduce the
        # historical 0.3/0.7.
        self.ft_alpha = float(os.environ.get('PHASER_FT_ALPHA', '0.3'))
        self.ft_beta = float(os.environ.get('PHASER_FT_BETA', '0.7'))

        # --- Reproducibility ----------------------------------------------
        # -1 disables seeding (the historical behaviour: nnU-Net never called
        # torch.manual_seed, so no run in this repo is reproducible).
        self.phaser_seed = int(os.environ.get('PHASER_SEED', '-1'))

        # --- Warmup for auxiliary losses ---
        self.warmup_epochs = int(os.environ.get('PHASER_WARMUP_EPOCHS', '50'))

        # --- Three-phase training ---
        self.phase2_start = int(os.environ.get('PHASER_PHASE2_START', '60'))
        self.phase3_start = int(os.environ.get('PHASER_PHASE3_START', '80'))

        # --- Training config ---
        self.enable_deep_supervision = True
        self.num_epochs = int(os.environ.get('PHASER_NUM_EPOCHS', '200'))
        self.initial_lr = 5e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = float(os.environ.get('PHASER_OVERSAMPLE', '0.66'))
        self.tumor_oversample_percent = float(os.environ.get('PHASER_TUMOR_OVERSAMPLE', '0.0'))

        # Best raw validation Dice for tumor label 2. This is tracked
        # independently from nnU-Net's EMA mean-foreground Dice checkpoint.
        self._best_tumor_dice = None

        self._set_batch_size_and_oversample()

        # Print configuration
        self.print_to_log_file(
            f"\n{'='*60}\n"
            f"PhASER-GPT Trainer Configuration:\n"
            f"  PCSF: {self.enable_pcsf} (mode={self.pcsf_mode}, gate_sens={self.pcsf_gate_sens}"
            + (f", pc_scales={self.pc_scales}, pc_max_voxels={self.pc_max_voxels}"
               if self.pcsf_mode == 'pc' else "") + ")\n"
            f"  PFCR: {self.enable_pfcr} (prior={self.pfcr_use_prior}, budget={self.enable_pfcr_budget}, w={self.pfcr_budget_weight})\n"
            f"  Phase congruency: pc_scales={self.pc_scales}, pc_max_voxels={self.pc_max_voxels}, "
            f"spacing={self.pc_spacing}, noise_rel_kappa={self.pc_noise_rel_kappa}\n"
            f"  PC boundary loss: {self.enable_pc_boundary_loss} "
            f"(w={self.pc_boundary_weight}, margin={self.pc_boundary_margin}, "
            f"warmup={self.pc_boundary_warmup}ep, active from epoch 0)\n"
            f"  LWFD: {self.enable_lwfd}\n"
            f"  LWFD Mode: {'full'}\n"
            f"  T-RAF: {self.t_raf_mode}\n"
            f"  ACG:  {self.enable_acg}\n"
            f"  LA-ACG: {self.enable_la_acg}\n"
            f"  Small-Tumor Head: {self.enable_small_head}\n"
            f"  Small-Head Loss: start={self.small_head_start}, warmup={self.small_head_warmup}, w={self.small_head_weight}\n"
            f"  CFF Fusion: {self.fusion_mode}\n"
            f"  Spectral loss: {self.enable_spectral_loss} (w={self.spectral_weight})\n"
            f"  Boundary loss: {self.enable_boundary_loss} (w={self.boundary_weight})\n"
            f"  Gate supervision: {self.enable_gate_loss} (w={self.gate_loss_weight})\n"
            f"  Inside-liver penalty: {self.enable_inside_penalty} (w={self.inside_penalty_weight})\n"
            f"  BoundaryW Dice: {self.enable_boundary_w_loss} (w={self.boundary_w_weight})\n"
            f"  Lesion weight: {self.enable_lesion_weight} (w={self.lesion_weight})\n"
            f"  Focal Tversky: w={self.ft_weight} (alpha={self.ft_alpha}, beta={self.ft_beta})\n"
            f"  PHASER_SEED: {self.phaser_seed}"
            + ("" if self.phaser_seed >= 0 else " (unset -> not reproducible)") + "\n"
            f"  Aux loss warmup: {self.warmup_epochs} epochs\n"
            f"  Three-phase: 0-{self.phase2_start-1} (A), "
            f"{self.phase2_start}-{self.phase3_start-1} (B), "
            f"{self.phase3_start}-{self.num_epochs} (C)\n"
            f"  Total epochs: {self.num_epochs}\n"
            f"  Initial LR: {self.initial_lr}\n"
            f"  Oversample: {self.oversample_foreground_percent}\n"
            f"  Tumor oversample (within FG): {self.tumor_oversample_percent}\n"
            f"{'='*60}\n"
        )

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)
        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        # Build standard nnUNet PlainConvUNet
        features_per_stage = [
            min(
                configuration_manager.UNet_base_num_features * 2 ** i,
                configuration_manager.unet_max_num_features,
            )
            for i in range(num_stages)
        ]

        base_model = PlainConvUNet(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
            num_classes=label_manager.num_segmentation_heads,
            n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=enable_deep_supervision,
        )

        # Load VoCo pretrained weights (skip if checkpoint path is empty)
        checkpoint_path = os.environ.get(
            'VOCOMNI_NNUNET_CKPT',
            '/root/autodl-tmp/Downstream/VoComni_nnunet.pt',
        )
        if checkpoint_path and os.path.isfile(checkpoint_path):
            base_model = load_pretrained_weights(base_model, checkpoint_path)

        # Read module switches from environment (in static context)
        enable_pcsf = os.environ.get('PHASER_PCSF', '0') == '1'
        enable_pfcr = os.environ.get('PHASER_PFCR', '0') == '1'
        pfcr_use_prior = os.environ.get('PHASER_PFCR_USE_PRIOR', '0') == '1'
        pcsf_mode = os.environ.get('PHASER_PCSF_MODE', 'grad')
        pc_scales = int(os.environ.get('PHASER_PC_SCALES', '3'))
        pc_max_voxels = int(os.environ.get('PHASER_PC_MAX_VOXELS', str(1 << 20)))
        pc_spacing = resolve_pc_spacing(configuration_manager, dim)
        pc_noise_rel_kappa = float(os.environ.get('PHASER_PC_NOISE_KAPPA', '0.0'))
        pcsf_gate_sens = float(os.environ.get('PHASER_PCSF_GATE_SENS', '0.5'))
        enable_lwfd = os.environ.get('PHASER_LWFD', '0') == '1'
        enable_acg = os.environ.get('PHASER_ACG', '0') == '1'
        fusion_mode = os.environ.get('PHASER_FUSION', 'none')
        lwfd_mode = 'full'
        t_raf_mode = os.environ.get('PHASER_T_RAF', 'none')
        enable_la_acg = os.environ.get('PHASER_LA_ACG', '0') == '1'
        enable_small_head = os.environ.get('PHASER_SMALL_HEAD', '0') == '1'
        # Tier-1 ablation module switches
        # T-RAF prior mode requires small head for tumor prior generation
        if (t_raf_mode in ('prior',) or (pfcr_use_prior and enable_pfcr)) and not enable_small_head:
            enable_small_head = True
        # ACG (channel) and LA-ACG (spatial liver-prior) both modulate decoder
        # stages but only one path runs in PhaserUNet.forward (LA-ACG wins).
        # Enabling both would build an ACG module that never executes — its
        # parameters become dead weights in the optimizer. Reject the combo.
        if enable_acg and enable_la_acg:
            raise ValueError(
                "PHASER_ACG and PHASER_LA_ACG are mutually exclusive "
                "(LA-ACG supersedes the channel-wise ACG in the decoder path). "
                "Enable only one of --acg / --la_acg."
            )

        model = PhaserUNet(
            base_model=base_model,
            feature_channels=features_per_stage,
            spatial_dims=dim,          # 2 or 3: propagated to all sub-modules
            enable_pcsf=enable_pcsf,
            enable_pfcr=enable_pfcr,
            pfcr_use_prior=pfcr_use_prior,
            pcsf_mode=pcsf_mode,
            pc_scales=pc_scales,
            pc_max_voxels=pc_max_voxels,
            pc_spacing=pc_spacing,
            pc_noise_rel_kappa=pc_noise_rel_kappa,
            pcsf_gate_sens=pcsf_gate_sens,
            enable_lwfd=enable_lwfd,
            enable_acg=enable_acg,
            fusion_mode=fusion_mode,
            lwfd_mode=lwfd_mode,
            t_raf_mode=t_raf_mode,
            enable_la_acg=enable_la_acg,
            enable_small_head=enable_small_head,
            num_classes=label_manager.num_segmentation_heads,
        )

        return model

    # ------------------------------------------------------------------
    # Loss construction
    # ------------------------------------------------------------------
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': True,
                    'smooth': 1e-5,
                    'ddp': self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            dice_ce = DC_and_CE_loss(
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'smooth': 1e-5,
                    'do_bg': False,
                    'ddp': self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            focal_tversky = MultiClassFocalTverskyLoss(
                alpha=self.ft_alpha,
                beta=self.ft_beta,
                gamma=4.0 / 3.0,
                include_background=False,
                ignore_label=self.label_manager.ignore_label,
            )
            loss = CombinedDiceCEFocalTverskyLoss(dice_ce, focal_tversky, self.ft_weight)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=float)
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    # ------------------------------------------------------------------
    # Optimizer with differential LR
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        phaser_params = []
        encoder_params = []
        decoder_params = []
        other_params = []

        # Check if network is PhaserUNet
        if isinstance(self.network, PhaserUNet):
            phaser_param_ids = set(id(p) for p in self.network.get_phaser_params())
        else:
            phaser_param_ids = set()

        for name, param in self.network.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in phaser_param_ids:
                phaser_params.append(param)
            elif 'encoder' in name:
                encoder_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': self.initial_lr * 0.3})
        if decoder_params:
            param_groups.append({'params': decoder_params, 'lr': self.initial_lr * 0.5})
        if phaser_params:
            param_groups.append({'params': phaser_params, 'lr': self.initial_lr})
        if other_params:
            param_groups.append({'params': other_params, 'lr': self.initial_lr})

        if not param_groups:
            param_groups = [{'params': self.network.parameters(), 'lr': self.initial_lr}]

        optimizer = torch.optim.SGD(
            param_groups,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        # GroupPolyLRScheduler preserves the per-group ratios (encoder 0.3x /
        # decoder 0.5x / phaser 1.0x). The stock PolyLRScheduler would overwrite
        # every group with a single lr and destroy the differential schedule.
        lr_scheduler = GroupPolyLRScheduler(optimizer, self.num_epochs)
        return optimizer, lr_scheduler

    # ------------------------------------------------------------------
    # Auxiliary-loss warmup ramp
    # ------------------------------------------------------------------
    def _aux_ramp(self, start_epoch: int) -> float:
        """
        Linear ramp in [0, 1] for an auxiliary loss that switches on at
        `start_epoch`. The weight is 0 at start_epoch and reaches 1 after
        `self.warmup_epochs` epochs, so every loss warms up relative to its own
        enable point instead of sharing one global progress value.
        """
        if self.current_epoch < start_epoch:
            return 0.0
        return min((self.current_epoch - start_epoch) / max(self.warmup_epochs, 1), 1.0)

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    def on_train_start(self):
        """Seed every RNG when PHASER_SEED is set.

        nnU-Net seeds nothing, so no run in this repository is reproducible and
        no A-vs-B difference can be separated from run-to-run variance. With
        PHASER_SEED >= 0 the three RNGs and any dataloader generators are
        pinned. Note cuDNN determinism is deliberately NOT forced: it costs a
        lot of throughput and the plan is to compare seeds anyway.
        """
        super().on_train_start()
        if self.phaser_seed < 0:
            self.print_to_log_file(
                "Reproducibility: PHASER_SEED unset -> RNGs NOT seeded "
                "(historical behaviour; reruns will differ).")
            return
        import random
        seed = int(self.phaser_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        for name in ('dataloader_train', 'dataloader_val'):
            gen = getattr(getattr(self, name, None), 'generator', None)
            if gen is not None:
                gen.manual_seed(seed)
        self.print_to_log_file(
            f"Reproducibility: seeded torch / numpy / random with PHASER_SEED={seed} "
            f"(cuDNN deterministic NOT forced).")

    # ------------------------------------------------------------------
    # Training step (with auxiliary losses)
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=self.use_amp) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            # Small-tumor head logits (training only) are exposed as an
            # attribute so forward keeps a constant return signature. The
            # network output is always just the segmentation result.
            net = self.network.module if self.is_ddp else self.network
            small_logits = getattr(net, '_last_small_logits', None)

            # Main segmentation loss
            seg_loss = self.loss(output, target)

            # Determine main output for auxiliary losses
            if self.enable_deep_supervision:
                main_output = output[0]
                main_target = target[0]
            else:
                main_output = output
                main_target = target

            # Phase tracking (three-phase)
            in_phase2 = self.current_epoch >= self.phase2_start
            in_phase3 = self.current_epoch >= self.phase3_start
            # Each auxiliary loss ramps up smoothly relative to its OWN enable
            # epoch (not a single global progress), so Phase-C losses start at
            # weight 0 when they switch on rather than jumping to a partial
            # value of a Phase-B ramp.
            ramp_b = self._aux_ramp(self.phase2_start)   # Phase B losses
            ramp_c = self._aux_ramp(self.phase3_start)   # Phase C losses
            aux_loss = main_output.new_tensor(0.0)

            # Boundary loss (tumor-specific, label 2) — Phase B/C
            if self.enable_boundary_loss and self.boundary_weight > 0 and in_phase2:
                b_loss = boundary_dice_loss(
                    main_output, main_target,
                    include_background=False,
                    ignore_label=self.label_manager.ignore_label,
                    tumor_label=2,
                )
                aux_loss = aux_loss + self.boundary_weight * ramp_b * b_loss

            # Boundary-weighted Dice loss (GPU-only) — Phase C
            if self.enable_boundary_w_loss and self.boundary_w_weight > 0 and in_phase3:
                bw_l = boundary_weighted_dice_loss(
                    main_output, main_target, label=2,
                )
                aux_loss = aux_loss + self.boundary_w_weight * ramp_c * bw_l

            # Spectral consistency loss — Phase B/C
            if self.enable_spectral_loss and self.spectral_weight > 0 and in_phase2:
                s_loss = self._compute_spectral_loss(main_output)
                aux_loss = aux_loss + self.spectral_weight * ramp_b * s_loss

            # Gate supervision loss — Phase C
            if self.enable_gate_loss and self.gate_loss_weight > 0 and in_phase3:
                gates = getattr(self.network, '_last_gates', None)
                if gates and any(g is not None for g in gates):
                    g_loss = gate_supervision_loss(gates, main_target)
                    aux_loss = aux_loss + self.gate_loss_weight * ramp_c * g_loss

            # Inside-liver penalty — Phase C
            if self.enable_inside_penalty and self.inside_penalty_weight > 0 and in_phase3:
                liver_prior = getattr(self.network, '_last_liver_prior', None)
                if liver_prior is not None:
                    il_loss = inside_liver_penalty_loss(main_output, liver_prior, tumor_label=2)
                    aux_loss = aux_loss + self.inside_penalty_weight * ramp_c * il_loss

            # Liver prior Dice supervision — Phase B/C (prevents degeneration to all-1)
            if self.enable_la_acg and in_phase2:
                liver_prior = getattr(self.network, '_last_liver_prior', None)
                if liver_prior is not None:
                    t4d = main_target.squeeze(1).long() if main_target.dim() == 5 else main_target.long()
                    liver_gt = ((t4d == 1) | (t4d == 2)).float().unsqueeze(1)
                    prior_l = F.binary_cross_entropy_with_logits(liver_prior, liver_gt, reduction='mean')
                    aux_loss = aux_loss + 0.02 * ramp_b * prior_l

            # PFCR frequency-budget constraint.
            if self.enable_pfcr_budget and self.pfcr_budget_weight > 0:
                net_pfcr = self.network.module if self.is_ddp else self.network
                routes = getattr(net_pfcr, '_last_pfcr_routes', None)
                if routes:
                    budget_loss = phase_frequency_budget_loss(routes, main_target)
                    aux_loss = aux_loss + self.pfcr_budget_weight * ramp_b * budget_loss

            # PC boundary ranking loss — active from epoch 0 (see __init__ note).
            # Supervises the PFCR/PCSF phase-congruency maps so the log-Gabor filters get a
            # gradient path that bypasses the residual enhancement gate.
            if (self.enable_pc_boundary_loss and self.pc_boundary_weight > 0
                    and (self.enable_pfcr or self.pcsf_mode == 'pc')):
                net_pc = self.network.module if self.is_ddp else self.network
                pc_maps = getattr(net_pc, '_last_pc_maps', None)
                if pc_maps:
                    ramp_pc = min((self.current_epoch + 1)
                                  / max(self.pc_boundary_warmup, 1), 1.0)
                    pc_l = pc_boundary_ranking_loss(
                        pc_maps, main_target, tumor_label=2,
                        margin=self.pc_boundary_margin,
                    )
                    aux_loss = aux_loss + self.pc_boundary_weight * ramp_pc * pc_l

            # Lesion-aware importance weighting — Phase B/C
            if self.enable_lesion_weight and self.lesion_weight > 0 and in_phase2:
                lesion_w = compute_lesion_importance_map(main_target)
                l_loss = lesion_weighted_loss(
                    main_output, main_target, lesion_w,
                    ignore_label=self.label_manager.ignore_label,
                )
                aux_loss = aux_loss + self.lesion_weight * ramp_b * l_loss

            # Small tumor head loss — active from PHASER_SMALL_HEAD_START
            # (default 0), with its own short warmup. This head produces PFCR's
            # conditioning input, so gating it behind Phase B (epoch >= 60) left
            # the tumour prior at random initialisation for the whole of Phase A
            # -- which is exactly the regime Copy2 lives in.
            if (small_logits is not None
                    and (self.current_epoch + 1) > self.small_head_start):
                t4d = main_target.squeeze(1).long() if main_target.dim() == 5 else main_target.long()
                tumor_target = (t4d == 2).long()
                tumor_weight = (t4d == 2).float() * 3.0 + 1.0
                sh_loss = F.cross_entropy(small_logits, tumor_target, reduction='none')
                sh_loss = (sh_loss * tumor_weight).mean()
                ramp_sh = min(
                    (self.current_epoch + 1 - self.small_head_start)
                    / max(self.small_head_warmup, 1), 1.0)
                aux_loss = aux_loss + self.small_head_weight * ramp_sh * sh_loss

            total_loss = seg_loss + aux_loss

        # Guard mixed-precision and auxiliary losses against rare NaN/Inf patches.
        # Skip the optimizer update for a non-finite batch; GradScaler alone does
        # not catch NaNs created before scaling.
        if not torch.isfinite(total_loss).all():
            self.print_to_log_file(f"WARNING: non-finite loss at epoch {self.current_epoch}; skipping update")
            self.optimizer.zero_grad(set_to_none=True)
            return {'loss': np.asarray(0.0, dtype=np.float32)}

        # Backward
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            if not torch.isfinite(grad_norm):
                self.print_to_log_file(f"WARNING: non-finite gradient at epoch {self.current_epoch}; skipping update")
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.update()
                return {'loss': total_loss.detach().cpu().numpy()}
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    # Deep supervision toggle (required for nnUNet validation/inference)
    # ------------------------------------------------------------------
    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            self.network.module.decoder.deep_supervision = enabled
        else:
            self.network.decoder.deep_supervision = enabled

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """
        Load checkpoints even when new PhASER-GPT modules were added after a
        run started. Missing newly introduced keys keep their initialization.
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        else:
            checkpoint = filename_or_checkpoint

        current_state = self.network.state_dict()
        compatible_state = {}
        skipped = []
        for k, value in checkpoint['network_weights'].items():
            key = k[7:] if k not in current_state and k.startswith('module.') else k
            if key in current_state and current_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                skipped.append(key)

        missing, unexpected = self.network.load_state_dict(compatible_state, strict=False)
        if missing:
            self.print_to_log_file(
                f"PhASER checkpoint compatibility: initialized {len(missing)} missing new keys."
            )
        if skipped or unexpected:
            self.print_to_log_file(
                f"PhASER checkpoint compatibility: skipped {len(skipped)} incompatible and "
                f"{len(unexpected)} unexpected keys."
            )

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self._best_tumor_dice = checkpoint.get('_best_tumor_dice')
        if self._best_tumor_dice is None:
            dice_history = self.logger.my_fantastic_logging.get('dice_per_class_or_region', [])
            historical_tumor_dice = []
            if self.label_manager.has_regions:
                regions = self.label_manager.all_regions
                tumor_index = next(
                    (i for i, region in enumerate(regions)
                     if region == 2 or (isinstance(region, (tuple, list)) and 2 in region)),
                    None,
                )
            else:
                foreground_labels = list(self.label_manager.foreground_labels)
                tumor_index = foreground_labels.index(2) if 2 in foreground_labels else None
            if tumor_index is not None:
                historical_tumor_dice = [
                    float(values[tumor_index])
                    for values in dice_history
                    if tumor_index < len(values) and np.isfinite(values[tumor_index])
                ]
            if historical_tumor_dice:
                self._best_tumor_dice = max(historical_tumor_dice)
        self.inference_allowed_mirroring_axes = checkpoint.get(
            'inference_allowed_mirroring_axes', self.inference_allowed_mirroring_axes
        )

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        except Exception as e:
            self.print_to_log_file(
                f"PhASER checkpoint compatibility: optimizer state not loaded ({e}). "
                "Continuing with a freshly initialized optimizer."
            )
        if self.grad_scaler is not None and checkpoint.get('grad_scaler_state') is not None:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    # ------------------------------------------------------------------
    # Spectral consistency loss (self-supervised)
    # ------------------------------------------------------------------
    def _compute_spectral_loss(self, _output) -> torch.Tensor:
        """
        Compute spectral consistency between encoder skips and decoder
        features.  Accesses intermediate features stored by PhaserUNet
        during forward pass.
        """
        if not isinstance(self.network, PhaserUNet):
            return _output.new_tensor(0.0)

        enc_skips = getattr(self.network, '_last_enc_skips', None)
        dec_feats = getattr(self.network, '_last_dec_features', None)

        if enc_skips is None or dec_feats is None:
            return _output.new_tensor(0.0)

        return spectral_consistency_loss(enc_skips, dec_feats, cutoff=0.3)

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=self.use_amp) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # forward returns the segmentation output only (small-head logits,
            # if any, are stored on the network and not produced in eval mode).
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }

    def _get_latest_tumor_dice(self):
        """Return the latest aggregated validation Dice for tumor label 2."""
        dice_history = self.logger.my_fantastic_logging.get('dice_per_class_or_region', [])
        if not dice_history:
            return None

        dice_values = dice_history[-1]
        if self.label_manager.has_regions:
            regions = self.label_manager.all_regions
            tumor_index = next(
                (i for i, region in enumerate(regions)
                 if region == 2 or (isinstance(region, (tuple, list)) and 2 in region)),
                None,
            )
        else:
            foreground_labels = list(self.label_manager.foreground_labels)
            tumor_index = foreground_labels.index(2) if 2 in foreground_labels else None

        if tumor_index is None or tumor_index >= len(dice_values):
            return None

        tumor_dice = float(dice_values[tumor_index])
        return tumor_dice if np.isfinite(tumor_dice) else None

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        """Aggregate validation metrics and save the best tumor-Dice model."""
        super().on_validation_epoch_end(val_outputs)

        tumor_dice = self._get_latest_tumor_dice()
        if tumor_dice is None:
            return

        if self._best_tumor_dice is None or tumor_dice > self._best_tumor_dice:
            self._best_tumor_dice = tumor_dice
            self.print_to_log_file(
                f"Yayy! New best tumor Dice: {np.round(self._best_tumor_dice, decimals=4)}"
            )
            self.save_checkpoint(
                os.path.join(self.output_folder, 'checkpoint_best_tumor_dice.pth')
            )
