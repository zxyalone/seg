"""
PhASER Trainer for nnUNet.

V2 configuration (used for all experiments in the manuscript; see
experiments/configs/*.env for the exact per-group settings):
  PHASER_PFCR=1 PHASER_PFCR_USE_PRIOR=1 PHASER_L_PFCR_BUDGET=1
  PHASER_L_PC_BOUNDARY=1, everything else at its default (off).
  Loss = DiceCE (deep supervision) + psi(e)*(0.2 L_rank + 0.05 L_budget + 0.1 L_prior),
  psi(e) = min(1, (e+1)/PHASER_AUX_WARMUP), all three active from epoch 0.

V2 switches added for the ablation / analysis protocol:
  PHASER_SEED=<int>        seeds python/numpy/torch and the augmentation workers
  PHASER_AUX_WARMUP=5      ramp length (epochs) shared by L_rank, L_budget, L_prior
  PHASER_PFCR_SHUFFLE_PC=1 A7 control: router sees spatially permuted PC
  PHASER_INTERVENE=none|pc_zero|route_uniform|prior_zero|gate_zero
                           inference-time intervention (E11)
  PHASER_INTERVENE_LEVELS=0,1,2  skip levels the intervention applies to (default all)
  PHASER_LEGACY=1          pre-V2 inference (double-sigmoid prior, PC noise scale
                           from scale 0); only for old checkpoints such as Copy2
  PHASER_FT_WEIGHT         Focal Tversky weight, default 0.0 (= DiceCE only);
                           pre-V2 runs used 0.15

Supports ablation studies via environment variables:
  PHASER_PCSF=1        → skip boundary gating
  PHASER_PCSF_MODE=grad|pc
                       → boundary evidence: 'grad' = finite-difference |∇f|
                         (amplitude-based, original), 'pc' = monogenic phase
                         congruency (contrast-invariant, for isoattenuating
                         lesions). Single-variable ablation.
  PHASER_PC_SCALES=3   → learnable log-Gabor scales when PCSF_MODE=pc
  PHASER_PC_MAX_VOXELS → FFT cost cap per skip (default 2^20). 2^20 is below
                         128^3, so the shallowest skip is computed at 64^3 and
                         interpolated back. Kept after measurement: the 2x average
                         denoises (synthetic AUC better capped, one real patch
                         better uncapped; step2_mechanism/p0_fix_report.md).
                         4194304 (= no downsampling) is an ablation.
  PHASER_PC_SPACING    → '' (default, cycles/voxel), 'plan', or 'a,b,c' mm per
                         axis for the log-Gabor bank. Ablation only.
  PHASER_PC_NOISE_KAPPA → cap the Kovesi threshold at kappa * sum_s A_s.
                         Default 0 (off); measured to localise worse.
  PHASER_FT_ALPHA / PHASER_FT_BETA → Focal Tversky asymmetry (0.3 / 0.7),
                         only used when PHASER_FT_WEIGHT > 0
  PHASER_PCSF_GATE_SENS → initial gate sensitivity alpha, shared by both PCSF
                         modes (default 0.5). MUST be nonzero for PCSF_MODE=pc:
                         alpha=0 zeroes d(attention)/d(PC) and the log-Gabor
                         filters never train. 0.0 = pre-2026-09 behaviour.
  PHASER_L_PC_BOUNDARY=1 → PC boundary ranking loss (requires PCSF_MODE=pc).
                         Active from epoch 0, NOT deferred to Phase B, because
                         it exists to un-throttle the log-Gabor filters early.
                         PHASER_PC_BOUNDARY_WEIGHT (default 0.2)
                         PHASER_PC_BOUNDARY_MARGIN (default 0.2)
  PHASER_LWFD=1        → Learnable Wavelet Feature Decomposition
  PHASER_ACG=1         → Anatomical Context Gating
  PHASER_L_SPECTRAL=1  → Spectral consistency loss
  PHASER_L_BOUNDARY=1  → Boundary-aware Dice loss

  PHASER_BOUNDARY_WEIGHT=0.1   (default)
  PHASER_SPECTRAL_WEIGHT=0.05  (default)
  PHASER_FT_WEIGHT=0.0         Focal Tversky weight (default; 0 disables it)

Usage:
  # Baseline
  nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr nnUNetTrainer_phaser

  # Full PhASER
  PHASER_PCSF=1 PHASER_LWFD=1 PHASER_ACG=1 PHASER_L_SPECTRAL=1 \
      nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr nnUNetTrainer_phaser
"""

import hashlib
import json
import os
import random
import socket
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
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
from .phaser.segresnet import SegResNetWithFeatures
from .phaser.lowcnr import lesion_contrast_augment, component_dicece_loss, contrast_consistency_loss, \
    relative_mse_consistency_loss
from .phaser.losses import boundary_dice_loss, spectral_consistency_loss, \
    compute_lesion_importance_map, lesion_weighted_loss, \
    gate_supervision_loss, inside_liver_penalty_loss, boundary_weighted_dice_loss, \
    pc_boundary_ranking_loss, phase_frequency_budget_loss


# FFT voxel cap. 2^20 < 128^3, so the shallowest skip runs at 64^3. Removing the
# cap changes that PC map substantially (design review I2), but measurements on
# the fix/p0-mechanism-defects branch did not show it to be better (the 2x
# average denoises), so the historical value is kept and 2^22 is an ablation.
PC_MAX_VOXELS_DEFAULT = 1 << 20


def resolve_pc_spacing(configuration_manager, dims: int):
    """mm per tensor axis for the PC log-Gabor bank, from PHASER_PC_SPACING:
    '' -> None (cycles/voxel, default); 'plan'/'auto' -> the plan spacing;
    'a,b,c' -> explicit values (a single value is broadcast)."""
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
    if len(vals) != dims or any(v <= 0 for v in vals):
        raise ValueError(f"PHASER_PC_SPACING must give 1 or {dims} positive values, got {vals}")
    return tuple(vals)

# Scalar / reparameterised PhASER parameters. Weight decay would pull them
# towards 0, i.e. towards rho=0.25, sigma=0.5, k=ln 2 and closed gates, which
# are arbitrary targets, so they go into a no-decay group.
NO_DECAY_PARAM_NAMES = ('rho_raw', 'sigma_raw', 'noise_k', 'residual_gate',
                        'alpha', 'beta', 'gamma', 'alphas')


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        super().__init__(optimizer, -1)

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
        # Seed before super().__init__ / initialize() so that network
        # initialisation (re-initialised heads and all PhASER modules) is
        # reproducible as well.
        seed_env = os.environ.get('PHASER_SEED', '')
        self.seed = int(seed_env) if seed_env != '' else None
        if self.seed is not None:
            _seed_everything(self.seed)
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Disable AMP for diagnosis/workstations where rare FP16 overflow creates
        # non-finite losses. Default remains enabled for normal CUDA training.
        self.use_amp = os.environ.get('PHASER_AMP', '1') == '1'
        # Autocast dtype. bf16 has the fp32 exponent range, so activations above
        # the fp16 maximum (65504) stay finite; it needs no loss scaling. Used for
        # the SegResNet + CFC runs, whose unnormalised residual decoder outgrew
        # fp16 (plan §11, 2026-10-13). Default fp16 = unchanged behaviour.
        self.amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}[os.environ.get('PHASER_AMP_DTYPE', 'fp16')]
        if self.amp_dtype == torch.bfloat16:
            self.grad_scaler = None

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
        self.pc_max_voxels = int(os.environ.get('PHASER_PC_MAX_VOXELS', str(PC_MAX_VOXELS_DEFAULT)))
        self.pc_spacing = resolve_pc_spacing(
            self.configuration_manager, len(self.configuration_manager.pool_op_kernel_sizes[0]))
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
        # Like the other V2 auxiliary losses this one is active from epoch 0 with
        # the shared PHASER_AUX_WARMUP ramp: it exists to un-throttle the
        # log-Gabor filters, which are scaled by tanh(gamma)~0 early in training.
        self.enable_pfcr = os.environ.get('PHASER_PFCR', '0') == '1'
        self.pfcr_use_prior = os.environ.get('PHASER_PFCR_USE_PRIOR', '0') == '1'
        self.enable_pfcr_budget = os.environ.get('PHASER_L_PFCR_BUDGET', '0') == '1'
        self.pfcr_budget_weight = float(os.environ.get('PHASER_PFCR_BUDGET_WEIGHT', '0.05'))
        self.pfcr_shuffle_pc = os.environ.get('PHASER_PFCR_SHUFFLE_PC', '0') == '1'
        self.legacy = os.environ.get('PHASER_LEGACY', '0') == '1'
        self.prior_weight = float(os.environ.get('PHASER_PRIOR_WEIGHT', '0.1'))
        # One ramp for the three V2 auxiliary losses (rank, budget, prior).
        self.aux_warmup = int(os.environ.get('PHASER_AUX_WARMUP', '5'))
        self.enable_lwfd = os.environ.get('PHASER_LWFD', '0') == '1'
        self.enable_acg = os.environ.get('PHASER_ACG', '0') == '1'
        self.t_raf_mode = os.environ.get('PHASER_T_RAF', 'none')
        self.enable_la_acg = os.environ.get('PHASER_LA_ACG', '0') == '1'
        self.enable_small_head = os.environ.get('PHASER_SMALL_HEAD', '0') == '1'
        # LMC (v0 of the detectability redesign): liver-referenced multi-scale
        # contrast skip gating. Its liver head is trained with BCE on liver labels
        # (label > 0), weight PHASER_LMC_LIVER_WEIGHT, shared V2 ramp, epoch 0 on.
        self.enable_lmc = os.environ.get('PHASER_LMC', '0') == '1'
        self.lmc_shuffle = os.environ.get('PHASER_LMC_SHUFFLE', '0') == '1'
        self.lmc_liver_weight = float(os.environ.get('PHASER_LMC_LIVER_WEIGHT', '0.1'))
        # Low-CNR training tools (plan §5b): lesion contrast augmentation (A11) and
        # component-wise DiceCE (A12). Both are training-only.
        self.enable_lca = os.environ.get('PHASER_LCA', '0') == '1'
        self.lca_p = float(os.environ.get('PHASER_LCA_P', '0.5'))
        self.lca_alpha_min = float(os.environ.get('PHASER_LCA_ALPHA_MIN', '0.3'))
        # A13a: per-lesion target CNR from the training low-CNR tier ('cnr');
        # A13b: + decoder-feature contrast consistency (weight, finest stages).
        self.lca_target = os.environ.get('PHASER_LCA_TARGET', 'uniform')
        self.lca_cons_weight = float(os.environ.get('PHASER_LCA_CONS_WEIGHT', '0.0'))
        self.lca_cons_stages = int(os.environ.get('PHASER_LCA_CONS_STAGES', '2'))
        # B13 (docs/训练侧频域计划.md): low-pass of the DCA shift field, sigma in mm (0 = plain DCA).
        self.lca_band_sigma_mm = float(os.environ.get('PHASER_LCA_BAND_SIGMA_MM', '0'))
        # E13: consistency teacher / loss. 'cosine' = A13b (same network, no grad, 1 - cos);
        # 'ema_rmse' = EMA teacher (momentum PHASER_LCA_EMA_M) with relative MSE.
        self.lca_cons_mode = os.environ.get('PHASER_LCA_CONS_MODE', 'cosine')
        self.lca_ema_m = float(os.environ.get('PHASER_LCA_EMA_M', '0.999'))
        self._ema_teacher = None
        if self.lca_cons_mode not in ('cosine', 'ema_rmse'):
            raise ValueError(f"PHASER_LCA_CONS_MODE must be 'cosine' or 'ema_rmse', got {self.lca_cons_mode!r}")
        if self.lca_band_sigma_mm > 0 and not self.enable_lca:
            raise ValueError('PHASER_LCA_BAND_SIGMA_MM > 0 requires PHASER_LCA=1')
        if self.lca_cons_mode == 'ema_rmse' and self.lca_cons_weight <= 0:
            raise ValueError("PHASER_LCA_CONS_MODE=ema_rmse requires PHASER_LCA_CONS_WEIGHT > 0")
        if self.lca_target not in ('uniform', 'cnr'):
            raise ValueError(f"PHASER_LCA_TARGET must be 'uniform' or 'cnr', got {self.lca_target!r}")
        if self.lca_cons_weight > 0 and not self.enable_lca:
            raise ValueError('PHASER_LCA_CONS_WEIGHT > 0 requires PHASER_LCA=1')
        self.enable_cc_loss = os.environ.get('PHASER_CC_LOSS', '0') == '1'
        self.cc_weight = float(os.environ.get('PHASER_CC_WEIGHT', '1.0'))
        # A14 (plan §11, 2026-10-14): Pixel2Cancer synthetic tumours, generated
        # offline by experiments/external/p2c_synthesize.py; a training case is
        # replaced by one of its variants with probability PHASER_P2C_P.
        self.p2c_dir = os.environ.get('PHASER_P2C_DIR', '')
        self.p2c_p = float(os.environ.get('PHASER_P2C_P', '0.9'))
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
        self.ft_weight = float(os.environ.get('PHASER_FT_WEIGHT', '0.0'))
        self.ft_alpha = float(os.environ.get('PHASER_FT_ALPHA', '0.3'))
        self.ft_beta = float(os.environ.get('PHASER_FT_BETA', '0.7'))
        self.gate_loss_weight = float(os.environ.get('PHASER_GATE_LOSS_WEIGHT', '0.03'))
        self.inside_penalty_weight = float(os.environ.get('PHASER_INSIDE_WEIGHT', '0.02'))
        self.boundary_w_weight = float(os.environ.get('PHASER_BOUNDARY_W_WEIGHT', '0.05'))

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
        # TriALS-setting experiment (plan §11, 2026-10-14): 'nnunet' = stock nnU-Net
        # optimiser (SGD, lr 1e-2 for every parameter, poly decay) as in the TriALS
        # nnUNetTrainer_100epochs; 'finetune' (default) = the 0.3x / 0.5x / 1x groups.
        self.lr_recipe = os.environ.get('PHASER_LR_RECIPE', 'finetune')
        if self.lr_recipe not in ('finetune', 'nnunet'):
            raise ValueError(f"PHASER_LR_RECIPE must be 'finetune' or 'nnunet', got {self.lr_recipe!r}")
        if self.lr_recipe == 'nnunet':
            self.initial_lr = 1e-2
        # Architecture-transfer experiment (plan §11): MONAI SegResNet with the
        # TriALS optimiser settings (Adam, lr 1e-4, wd 1e-5, no deep supervision).
        self.arch = os.environ.get('PHASER_ARCH', 'plainunet')
        if self.arch not in ('plainunet', 'segresnet'):
            raise ValueError(f"PHASER_ARCH must be 'plainunet' or 'segresnet', got {self.arch!r}")
        if self.arch == 'segresnet':
            self.enable_deep_supervision = False
            self.initial_lr = 1e-4
            self.weight_decay = 1e-5
        self.oversample_foreground_percent = float(os.environ.get('PHASER_OVERSAMPLE', '0.66'))

        # Best raw validation Dice for tumor label 2. This is tracked
        # independently from nnU-Net's EMA mean-foreground Dice checkpoint.
        self._best_tumor_dice = None

        self._set_batch_size_and_oversample()

        # Print configuration
        self.print_to_log_file(
            f"\n{'='*60}\n"
            f"PhASER-GPT Trainer Configuration:\n"
            f"  Seed: {self.seed}  Legacy: {self.legacy}  PFCR shuffle PC (A7): {self.pfcr_shuffle_pc}\n"
            f"  Aux warmup (rank/budget/prior): {self.aux_warmup} epochs, prior w={self.prior_weight}\n"
            f"  PCSF: {self.enable_pcsf} (mode={self.pcsf_mode}, gate_sens={self.pcsf_gate_sens}"
            + (f", pc_scales={self.pc_scales}, pc_max_voxels={self.pc_max_voxels}"
               if self.pcsf_mode == 'pc' else "") + ")\n"
            f"  Phase congruency: pc_max_voxels={self.pc_max_voxels}, spacing={self.pc_spacing}, "
            f"noise_rel_kappa={self.pc_noise_rel_kappa}\n"
            f"  PFCR: {self.enable_pfcr} (prior={self.pfcr_use_prior}, budget={self.enable_pfcr_budget}, w={self.pfcr_budget_weight})\n"
            f"  PC boundary loss: {self.enable_pc_boundary_loss} "
            f"(w={self.pc_boundary_weight}, margin={self.pc_boundary_margin}, "
            f"ramp=PHASER_AUX_WARMUP, active from epoch 0)\n"
            f"  LWFD: {self.enable_lwfd}\n"
            f"  LWFD Mode: {'full'}\n"
            f"  T-RAF: {self.t_raf_mode}\n"
            f"  ACG:  {self.enable_acg}\n"
            f"  LA-ACG: {self.enable_la_acg}\n"
            f"  Small-Tumor Head: {self.enable_small_head}\n"
            f"  LMC contrast gate: {self.enable_lmc} (shuffle={self.lmc_shuffle}, "
            f"liver head w={self.lmc_liver_weight}, spacing={list(self.configuration_manager.spacing)})\n"
            f"  Architecture: {self.arch} (deep supervision={self.enable_deep_supervision}, lr={self.initial_lr}, "
            f"autocast={'off' if not self.use_amp else str(self.amp_dtype).replace('torch.', '')})\n"
            f"  Lesion contrast aug: {self.enable_lca} (p={self.lca_p}, alpha_min={self.lca_alpha_min}, "
            f"target={self.lca_target}, consistency w={self.lca_cons_weight}, stages={self.lca_cons_stages}, "
            f"band sigma={self.lca_band_sigma_mm} mm, consistency mode={self.lca_cons_mode}, "
            f"EMA m={self.lca_ema_m})\n"
            f"  Component DiceCE: {self.enable_cc_loss} (w={self.cc_weight})\n"
            f"  Pixel2Cancer variants: {self.p2c_dir or 'off'} (p={self.p2c_p})\n"
            f"  CFF Fusion: {self.fusion_mode}\n"
            f"  Spectral loss: {self.enable_spectral_loss} (w={self.spectral_weight})\n"
            f"  Boundary loss: {self.enable_boundary_loss} (w={self.boundary_weight})\n"
            f"  Gate supervision: {self.enable_gate_loss} (w={self.gate_loss_weight})\n"
            f"  Inside-liver penalty: {self.enable_inside_penalty} (w={self.inside_penalty_weight})\n"
            f"  BoundaryW Dice: {self.enable_boundary_w_loss} (w={self.boundary_w_weight})\n"
            f"  Lesion weight: {self.enable_lesion_weight} (w={self.lesion_weight})\n"
            f"  Focal Tversky: w={self.ft_weight} (alpha={self.ft_alpha}, beta={self.ft_beta})\n"
            f"  Aux loss warmup: {self.warmup_epochs} epochs\n"
            f"  Three-phase: 0-{self.phase2_start-1} (A), "
            f"{self.phase2_start}-{self.phase3_start-1} (B), "
            f"{self.phase3_start}-{self.num_epochs} (C)\n"
            f"  Total epochs: {self.num_epochs}\n"
            f"  Initial LR: {self.initial_lr} (recipe {self.lr_recipe})\n"
            f"  Oversample: {self.oversample_foreground_percent}\n"
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

        if os.environ.get('PHASER_ARCH', 'plainunet') == 'segresnet':
            # Trained from scratch (VoComni weights exist only for PlainConvUNet).
            return SegResNetWithFeatures(num_input_channels, label_manager.num_segmentation_heads, dim)

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
            print(f'Pretrained weights: {checkpoint_path}')
        else:
            print('Pretrained weights: none (trained from scratch)')

        # Read module switches from environment (in static context)
        enable_pcsf = os.environ.get('PHASER_PCSF', '0') == '1'
        enable_pfcr = os.environ.get('PHASER_PFCR', '0') == '1'
        pfcr_use_prior = os.environ.get('PHASER_PFCR_USE_PRIOR', '0') == '1'
        pcsf_mode = os.environ.get('PHASER_PCSF_MODE', 'grad')
        pc_scales = int(os.environ.get('PHASER_PC_SCALES', '3'))
        pc_max_voxels = int(os.environ.get('PHASER_PC_MAX_VOXELS', str(PC_MAX_VOXELS_DEFAULT)))
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
        pfcr_shuffle_pc = os.environ.get('PHASER_PFCR_SHUFFLE_PC', '0') == '1'
        pfcr_intervene = os.environ.get('PHASER_INTERVENE', 'none')
        levels_env = os.environ.get('PHASER_INTERVENE_LEVELS', '').strip()
        pfcr_intervene_levels = [int(v) for v in levels_env.split(',')] if levels_env else None
        legacy = os.environ.get('PHASER_LEGACY', '0') == '1'
        enable_lmc = os.environ.get('PHASER_LMC', '0') == '1'
        lmc_shuffle = os.environ.get('PHASER_LMC_SHUFFLE', '0') == '1'
        lmc_intervene = os.environ.get('PHASER_LMC_INTERVENE', 'none')
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
            pcsf_gate_sens=pcsf_gate_sens,
            enable_lwfd=enable_lwfd,
            enable_acg=enable_acg,
            fusion_mode=fusion_mode,
            lwfd_mode=lwfd_mode,
            t_raf_mode=t_raf_mode,
            enable_la_acg=enable_la_acg,
            enable_small_head=enable_small_head,
            num_classes=label_manager.num_segmentation_heads,
            pfcr_shuffle_pc=pfcr_shuffle_pc,
            pfcr_intervene=pfcr_intervene,
            pfcr_intervene_levels=pfcr_intervene_levels,
            legacy=legacy,
            pc_spacing=pc_spacing,
            pc_noise_rel_kappa=pc_noise_rel_kappa,
            enable_lmc=enable_lmc,
            lmc_spacing=[float(v) for v in configuration_manager.spacing],
            lmc_shuffle=lmc_shuffle,
            lmc_intervene=lmc_intervene,
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
            if self.ft_weight > 0:
                focal_tversky = MultiClassFocalTverskyLoss(
                    alpha=self.ft_alpha,
                    beta=self.ft_beta,
                    gamma=4.0 / 3.0,
                    include_background=False,
                    ignore_label=self.label_manager.ignore_label,
                )
                loss = CombinedDiceCEFocalTverskyLoss(dice_ce, focal_tversky, self.ft_weight)
            else:
                loss = dice_ce

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
        if self.arch == 'segresnet':
            from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
            optimizer = torch.optim.Adam(self.network.parameters(), lr=self.initial_lr,
                                         weight_decay=self.weight_decay, eps=1e-5)
            return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.9)
        phaser_params = []
        phaser_no_decay = []
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
                if name.split('.')[-1] in NO_DECAY_PARAM_NAMES:
                    phaser_no_decay.append(param)
                else:
                    phaser_params.append(param)
            elif 'encoder' in name:
                encoder_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if self.lr_recipe == 'nnunet':
            encoder_params, decoder_params = [], []
            other_params = [p for p in self.network.parameters() if p.requires_grad]
            phaser_params, phaser_no_decay = [], []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': self.initial_lr * 0.3})
        if decoder_params:
            param_groups.append({'params': decoder_params, 'lr': self.initial_lr * 0.5})
        if phaser_params:
            param_groups.append({'params': phaser_params, 'lr': self.initial_lr})
        if phaser_no_decay:
            param_groups.append({'params': phaser_no_decay, 'lr': self.initial_lr,
                                 'weight_decay': 0.0})
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
        self.print_to_log_file(
            "Optimizer groups: " + ", ".join(
                f"{len(g['params'])} params lr={g['lr']:.2e} wd={g['weight_decay']:g}"
                for g in optimizer.param_groups))
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

    def _v2_ramp(self) -> float:
        """psi(e) = min(1, (e+1)/W): shared by L_rank, L_budget and L_prior,
        non-zero from the first epoch (manuscript Eq. 28)."""
        return min((self.current_epoch + 1) / max(self.aux_warmup, 1), 1.0)

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

        cons_region, cons_teacher = None, None
        if self.enable_lca:
            full_target = target[0] if isinstance(target, list) else target
            clean = data
            if self.lca_cons_weight > 0:
                cons_region = torch.zeros(full_target.shape, device=data.device, dtype=torch.float32)
            band = None
            if self.lca_band_sigma_mm > 0:
                band = tuple(self.lca_band_sigma_mm / float(sp) for sp in self.configuration_manager.spacing)
            data, _ = lesion_contrast_augment(data, full_target, p=self.lca_p,
                                              alpha_min=self.lca_alpha_min,
                                              target_cnr=self.lca_target == 'cnr',
                                              region_out=cons_region, band_sigma=band)
            # A13b teacher: same network on the original patch, no gradient. Run
            # before the student pass so the cached features are the student's.
            if self.lca_cons_mode == 'ema_rmse' and self._ema_teacher is None:
                self._init_ema_teacher()
            if cons_region is not None and bool(cons_region.any()):
                net_c = self.network.module if self.is_ddp else self.network
                net_c._spectral_grad_scales = 10 ** 6   # keep every decoder stage in the graph
                teacher = self._ema_teacher if self.lca_cons_mode == 'ema_rmse' else self.network
                teacher_c = teacher.module if (teacher is self.network and self.is_ddp) else teacher
                with torch.no_grad():
                    with autocast(self.device.type, enabled=self.use_amp, dtype=self.amp_dtype) if self.device.type == 'cuda' else dummy_context():
                        teacher(clean)
                cons_teacher = [f.detach() for f in teacher_c._last_dec_features[:self.lca_cons_stages]]

        with autocast(self.device.type, enabled=self.use_amp, dtype=self.amp_dtype) if self.device.type == 'cuda' else dummy_context():
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
                    aux_loss = aux_loss + self.pfcr_budget_weight * self._v2_ramp() * budget_loss

            # PC boundary ranking loss — active from epoch 0 (see __init__ note).
            # Supervises the PFCR/PCSF phase-congruency maps so the log-Gabor filters get a
            # gradient path that bypasses the residual enhancement gate.
            if (self.enable_pc_boundary_loss and self.pc_boundary_weight > 0
                    and (self.enable_pfcr or self.pcsf_mode == 'pc')):
                net_pc = self.network.module if self.is_ddp else self.network
                pc_maps = getattr(net_pc, '_last_pc_maps', None)
                if pc_maps:
                    ramp_pc = self._v2_ramp()
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

            # Prior (small tumor) head loss (binary: BG vs tumor). V2: active
            # from epoch 0, because PFCR conditions on the prior from epoch 0.
            if small_logits is not None and self.prior_weight > 0:
                t4d = main_target.squeeze(1).long() if main_target.dim() == 5 else main_target.long()
                tumor_target = (t4d == 2).long()
                tumor_weight = (t4d == 2).float() * 3.0 + 1.0
                sh_loss = F.cross_entropy(small_logits.float(), tumor_target, reduction='none')
                sh_loss = (sh_loss * tumor_weight).mean()
                aux_loss = aux_loss + self.prior_weight * self._v2_ramp() * sh_loss

            # Component-wise DiceCE (A12), full resolution output only.
            if self.enable_cc_loss and self.cc_weight > 0:
                aux_loss = aux_loss + self.cc_weight * component_dicece_loss(main_output, main_target)

            # Contrast consistency (A13b): attenuated vs original decoder features.
            if cons_teacher is not None:
                net_c = self.network.module if self.is_ddp else self.network
                student = net_c._last_dec_features[:self.lca_cons_stages]
                if not all(f.requires_grad for f in student):
                    raise RuntimeError('A13b: student decoder features carry no gradient')
                cons_fn = relative_mse_consistency_loss if self.lca_cons_mode == 'ema_rmse' \
                    else contrast_consistency_loss
                aux_loss = aux_loss + self.lca_cons_weight * self._v2_ramp() * \
                    cons_fn(student, cons_teacher, cons_region, self.lca_cons_stages)

            # LMC liver head: BCE against liver (label > 0), from epoch 0.
            if self.enable_lmc and self.lmc_liver_weight > 0:
                net_lmc = self.network.module if self.is_ddp else self.network
                liver_logits = getattr(net_lmc, '_last_lmc_liver_logits', None)
                if liver_logits is not None:
                    t4d = main_target.squeeze(1).long() if main_target.dim() == 5 else main_target.long()
                    liver_gt = (t4d > 0).float().unsqueeze(1)
                    if liver_logits.shape[2:] != liver_gt.shape[2:]:
                        liver_gt = F.interpolate(liver_gt, size=liver_logits.shape[2:], mode='nearest')
                    lmc_l = F.binary_cross_entropy_with_logits(liver_logits.float(), liver_gt)
                    aux_loss = aux_loss + self.lmc_liver_weight * self._v2_ramp() * lmc_l

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
            self._ema_update()
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            if not torch.isfinite(grad_norm):
                self.print_to_log_file(f"WARNING: non-finite gradient at epoch {self.current_epoch}; skipping update")
                self.optimizer.zero_grad(set_to_none=True)
                return {'loss': total_loss.detach().cpu().numpy()}
            self.optimizer.step()
            self._ema_update()

        return {'loss': total_loss.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    # E13: EMA teacher for the contrast-consistency loss
    # ------------------------------------------------------------------
    def _student_module(self):
        net = self.network.module if self.is_ddp else self.network
        return getattr(net, '_orig_mod', net)      # unwrap torch.compile if enabled

    def _init_ema_teacher(self):
        """Teacher = copy of the student at the first training step (after the
        pretrained weights are loaded); never receives gradients."""
        import copy
        self._ema_teacher = copy.deepcopy(self._student_module())
        for p in self._ema_teacher.parameters():
            p.requires_grad_(False)
        self.print_to_log_file(f"E13: EMA teacher initialised from the student (m={self.lca_ema_m})")

    @torch.no_grad()
    def _ema_update(self):
        if self._ema_teacher is None:
            return
        m = self.lca_ema_m
        student = self._student_module()
        for pt, ps in zip(self._ema_teacher.parameters(), student.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bt, bs in zip(self._ema_teacher.buffers(), student.buffers()):
            bt.copy_(bs)

    # ------------------------------------------------------------------
    # Reproducibility: worker seeds and run manifest
    # ------------------------------------------------------------------
    def get_tr_and_val_datasets(self):
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()
        if self.p2c_dir:
            dataset_tr = P2CDataset(dataset_tr, self.p2c_dir, self.p2c_p)
            assert not set(dataset_tr.variants) & set(dataset_val.keys()), 'synthetic variants of validation cases'
            self.print_to_log_file(f"Pixel2Cancer: {sum(map(len, dataset_tr.variants.values()))} variants for "
                                   f"{len(dataset_tr.variants)} of {len(dataset_tr.keys())} training cases, p={self.p2c_p}")
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        mt_gen_train, mt_gen_val = super().get_dataloaders()
        if self.seed is not None:
            # The multi-process augmenters start lazily, so their per-worker
            # seeds can still be set here. Batch order across workers stays
            # non-deterministic (NonDetMultiThreadedAugmenter), so runs with the
            # same seed are close but not bit-identical.
            for offset, gen in ((0, mt_gen_train), (10_000, mt_gen_val)):
                n = getattr(gen, 'num_processes', None)
                if n and hasattr(gen, 'seeds'):
                    gen.seeds = [self.seed * 100_000 + offset + i for i in range(n)]
        return mt_gen_train, mt_gen_val

    def _manifest_path(self) -> str:
        return os.path.join(self.output_folder, 'run_manifest.json')

    def _write_manifest(self, **updates) -> None:
        if self.local_rank != 0:
            return
        path = self._manifest_path()
        manifest = {}
        if os.path.isfile(path):
            with open(path) as f:
                manifest = json.load(f)
        manifest.update(updates)
        os.makedirs(self.output_folder, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    @staticmethod
    def _git_commit() -> str:
        repo = os.path.dirname(os.path.abspath(__file__))
        try:
            commit = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'],
                                             stderr=subprocess.DEVNULL, text=True).strip()
            dirty = subprocess.check_output(['git', '-C', repo, 'status', '--porcelain'],
                                            stderr=subprocess.DEVNULL, text=True).strip()
            return commit + ('-dirty' if dirty else '')
        except Exception:
            return 'unknown'

    def on_train_start(self):
        super().on_train_start()
        gpu = torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else self.device.type
        self._write_manifest(
            git_commit=self._git_commit(),
            env={k: v for k, v in sorted(os.environ.items())
                 if k.startswith(('PHASER_', 'nnUNet_', 'VOCOMNI_'))},
            seed=self.seed,
            num_epochs=self.num_epochs,
            num_iterations_per_epoch=self.num_iterations_per_epoch,
            fold=self.fold,
            gpu=gpu,
            host=socket.gethostname(),
            torch_version=torch.__version__,
            start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            start_epoch=self.current_epoch,
        )

    def on_train_end(self):
        super().on_train_end()
        final = os.path.join(self.output_folder, 'checkpoint_final.pth')
        sha = None
        if os.path.isfile(final):
            h = hashlib.sha256()
            with open(final, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
            sha = h.hexdigest()
        self._write_manifest(
            end_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            epochs_completed=self.current_epoch,
            checkpoint_final_sha256=sha,
        )

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Final validation without mirroring TTA (protocol: no TTA anywhere).

        nnU-Net's predictor here always mirrors over inference_allowed_mirroring_axes;
        clearing the axes for the duration of the call disables it. The value
        stored in the checkpoint is unaffected (checkpoint_final is written
        before validation). PHASER_VAL_TTA=1 restores nnU-Net's behaviour.
        """
        if os.environ.get('PHASER_VAL_TTA', '0') == '1':
            return super().perform_actual_validation(save_probabilities)
        axes = self.inference_allowed_mirroring_axes
        self.inference_allowed_mirroring_axes = None
        try:
            return super().perform_actual_validation(save_probabilities)
        finally:
            self.inference_allowed_mirroring_axes = axes

    # ------------------------------------------------------------------
    # Deep supervision toggle (required for nnUNet validation/inference)
    # ------------------------------------------------------------------
    def set_deep_supervision_enabled(self, enabled: bool):
        net = self.network.module if self.is_ddp else self.network
        if not hasattr(net, 'decoder'):      # SegResNet: no deep supervision
            return
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

        with autocast(self.device.type, enabled=self.use_amp, dtype=self.amp_dtype) if self.device.type == 'cuda' else dummy_context():
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


class _PatchedArray:
    """Read-only view of a (C, Z, Y, X) case array with a box replaced, for the
    slicing done by nnUNetDataLoader3D (data[slices]); returns copies."""

    def __init__(self, base, lo, patch):
        self.base, self.lo, self.patch = base, np.asarray(lo), patch.astype(base.dtype)
        self.shape, self.dtype = base.shape, base.dtype

    def __getitem__(self, sl):
        out = np.array(self.base[sl])
        starts = np.array([s.start or 0 for s in sl[1:]])
        stops = np.array([n if s.stop is None else s.stop for s, n in zip(sl[1:], self.shape[1:])])
        a = np.maximum(starts, self.lo)
        b = np.minimum(stops, self.lo + np.array(self.patch.shape[1:]))
        if np.all(b > a):
            out[(slice(None),) + tuple(slice(x - s, y - s) for x, y, s in zip(a, b, starts))] = \
                self.patch[(sl[0],) + tuple(slice(x - p, y - p) for x, y, p in zip(a, b, self.lo))]
        return out


class P2CDataset(nnUNetDataset):
    """Training dataset for A14: with probability p a case is replaced by one of
    its offline Pixel2Cancer variants (changed box pasted over the case), and the
    synthetic tumour voxels are added to the label-2 sampling locations."""

    def __init__(self, base: nnUNetDataset, synth_dir: str, p: float):
        self.__dict__.update(base.__dict__)
        index = json.load(open(os.path.join(synth_dir, 'index.json')))
        self.variants = {c: [os.path.join(synth_dir, f) for f in fs]
                         for c, fs in index['variants'].items() if c in self.dataset}
        self.p = p

    def load_case(self, key):
        data, seg, properties = super().load_case(key)
        files = self.variants.get(key)
        if not files or np.random.uniform() >= self.p:
            return data, seg, properties
        v = np.load(files[np.random.randint(len(files))])
        lo = v['lo']
        data = _PatchedArray(data, lo, v['image'][None])
        seg = _PatchedArray(seg, lo, v['seg'][None])
        properties = dict(properties)
        locs = dict(properties['class_locations'])
        # cases without real tumour store an empty (float) array; keep integer coordinates
        old = np.asarray(locs.get(2, np.zeros((0, 4), np.int64))).reshape(-1, 4).astype(np.int64)
        locs[2] = np.concatenate([old, np.asarray(v['tumour_locs'], np.int64).reshape(-1, 4)])
        properties['class_locations'] = locs
        return data, seg, properties

