#!/usr/bin/env python3
"""
E0 gradient check: every trainable PhASER parameter must receive a non-zero
gradient and move during a short training run, for every ablation group.

This would have caught the Copy2 problem (log-Gabor filters and prior head never
trained). It runs the real nnUNetTrainer_phaser.train_step and
configure_optimizers on synthetic CT-like volumes, so no dataset is needed and
it also runs on CPU.

    PYTHONPATH=nnUNet python experiments/check_grads.py                 # all groups, CPU
    PYTHONPATH=nnUNet python experiments/check_grads.py --groups A6 A7 --steps 50 --patch 128 --device cuda

Criteria per PhASER tensor:
  FAIL  never receives a non-zero gradient (dead parameter, the Copy2 failure)
  WARN  max |grad| < 1e-6. Expected early in training for parameters whose only
        path runs through tanh(gamma) ~ 0 (log-Gabor filters in A3-A5, or in A6
        at levels where the ranking hinge is already satisfied) and for the
        router's first conv before its zero-initialised output conv moves.
Exit code 1 if any group has a FAIL.
Groups are read from experiments/configs/<group>.env, so the check covers the
exact switches the training queue uses.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "experiments" / "configs"
sys.path.insert(0, str(ROOT / "nnUNet"))

def read_env(group: str) -> dict:
    env = {}
    for line in (CONFIGS / f"{group}.env").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def synthetic_batch(n: int, patch: int, rng: np.random.Generator, deep_scales):
    """CT-like volume in nnU-Net z-score units: liver sphere + 1-3 tumors.

    Lesion contrast 0.05-0.25 against noise sigma 0.15 gives CNR 0.3-1.7, close
    to the LiTS training lesions (median CNR 1.6). With easier lesions the PC
    ranking hinge is already satisfied at init and the check says little.
    """
    zz, yy, xx = np.meshgrid(*[np.arange(patch)] * 3, indexing="ij")
    imgs, segs = [], []
    for _ in range(n):
        seg = np.zeros((patch,) * 3, np.int64)
        c = rng.uniform(0.4, 0.6, 3) * patch
        r = patch * rng.uniform(0.3, 0.38)
        liver = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 < r ** 2
        seg[liver] = 1
        for _ in range(rng.integers(1, 4)):
            tc = c + rng.normal(0, r / 3, 3)
            tr = rng.uniform(2.5, patch / 8)
            tumor = ((zz - tc[0]) ** 2 + (yy - tc[1]) ** 2 + (xx - tc[2]) ** 2 < tr ** 2) & liver
            seg[tumor] = 2
        img = np.where(seg == 1, 0.8, -1.0) + np.where(seg == 2, rng.uniform(-0.25, -0.05), 0.0)
        img = img + rng.normal(0, 0.15, img.shape)
        imgs.append(img.astype(np.float32))
        segs.append(seg)
    data = torch.from_numpy(np.stack(imgs))[:, None]
    seg = torch.from_numpy(np.stack(segs))[:, None].float()
    targets = []
    for s in deep_scales:
        size = tuple(int(round(patch * f)) for f in s)
        targets.append(torch.nn.functional.interpolate(seg, size=size, mode="nearest"))
    return {"data": data, "target": targets}


def make_trainer(patch: int, device: torch.device, n_stages: int):
    """Instantiate nnUNetTrainer_phaser without plans/dataset files."""
    from nnunetv2.training.nnUNetTrainer import nnUNetTrainer_phaser as mod
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
    cm = SimpleNamespace(
        conv_kernel_sizes=[[3, 3, 3]] * n_stages,
        pool_op_kernel_sizes=strides,
        UNet_base_num_features=32,
        unet_max_num_features=320,
        n_conv_per_stage_encoder=[2] * n_stages,
        n_conv_per_stage_decoder=[2] * (n_stages - 1),
        batch_dice=False,
        patch_size=[patch] * 3,
        spacing=[1.0, 0.7791720032691956, 0.7791720032691956],
    )
    lm = SimpleNamespace(has_regions=False, ignore_label=None, has_ignore_label=False,
                         foreground_labels=[1, 2], num_segmentation_heads=3)
    pm = SimpleNamespace(get_label_manager=lambda _dj: lm)

    def fake_base_init(self, *a, **k):
        self.device = device
        self.is_ddp = False
        self.local_rank = 0
        self.label_manager = lm
        self.configuration_manager = cm
        self.plans_manager = pm
        self.current_epoch = 0
        self.enable_deep_supervision = True
        self.output_folder = None

    orig_init = nnUNetTrainer.__init__
    orig_set_bs = mod.nnUNetTrainer_phaser._set_batch_size_and_oversample
    nnUNetTrainer.__init__ = fake_base_init
    mod.nnUNetTrainer_phaser._set_batch_size_and_oversample = lambda self: None
    mod.nnUNetTrainer_phaser.print_to_log_file = lambda self, *a, **k: None
    try:
        tr = mod.nnUNetTrainer_phaser({}, "3d_fullres", 0, {}, device=device)
    finally:
        nnUNetTrainer.__init__ = orig_init
        mod.nnUNetTrainer_phaser._set_batch_size_and_oversample = orig_set_bs

    tr.network = mod.nnUNetTrainer_phaser.build_network_architecture(
        pm, {}, cm, 1, enable_deep_supervision=True).to(device)
    tr.loss = tr._build_loss()
    tr.optimizer, tr.lr_scheduler = tr.configure_optimizers()
    tr.grad_scaler = None
    tr.use_amp = False
    return tr


def check_group(group: str, args) -> bool:
    from nnunetv2.training.nnUNetTrainer import nnUNetTrainer_phaser as trainer_mod
    original_aug = trainer_mod.lesion_contrast_augment
    original_cc = trainer_mod.component_dicece_loss
    original_cons = trainer_mod.contrast_consistency_loss
    paths = dict(augmentation_calls=0, augmented_samples=0, changed_batches=0,
                 component_loss_calls=0, positive_component_losses=0,
                 consistency_calls=0, positive_consistency_losses=0)

    def checked_aug(data, *a, **kw):
        result, applied = original_aug(data, *a, **kw)
        paths['augmentation_calls'] += 1
        paths['augmented_samples'] += len(applied)
        paths['changed_batches'] += int(not torch.equal(result, data))
        return result, applied

    def checked_cc(*a, **kw):
        loss = original_cc(*a, **kw)
        if not torch.isfinite(loss) or not loss.requires_grad:
            raise RuntimeError('Component loss must be finite and retain its gradient path')
        paths['component_loss_calls'] += 1
        paths['positive_component_losses'] += int(loss.detach() > 0)
        return loss

    def checked_cons(*a, **kw):
        loss = original_cons(*a, **kw)
        if not torch.isfinite(loss) or not loss.requires_grad:
            raise RuntimeError('Consistency loss must be finite and retain its gradient path')
        paths['consistency_calls'] += 1
        paths['positive_consistency_losses'] += int(loss.detach() > 0)
        return loss

    trainer_mod.lesion_contrast_augment = checked_aug
    trainer_mod.component_dicece_loss = checked_cc
    trainer_mod.contrast_consistency_loss = checked_cons
    env = read_env(group)
    saved = {k: os.environ.get(k) for k in env}
    for k in list(os.environ):
        if k.startswith("PHASER_"):
            saved.setdefault(k, os.environ[k])
            del os.environ[k]
    os.environ.update(env)
    os.environ["VOCOMNI_NNUNET_CKPT"] = ""  # pretrained weights are irrelevant here
    try:
        torch.manual_seed(0)
        device = torch.device(args.device)
        tr = make_trainer(args.patch, device, args.stages)
        net = tr.network
        phaser_ids = {id(p) for p in net.get_phaser_params()}
        named = [(n, p) for n, p in net.named_parameters() if id(p) in phaser_ids]
        init = {n: p.detach().clone() for n, p in named}
        got_grad = {n: False for n, _ in named}
        max_grad = {n: 0.0 for n, _ in named}
        base_name, base_param = next(net.encoder.named_parameters())
        base_initial = base_param.detach().clone()
        base_grad = 0.0

        rng = np.random.default_rng(0)
        scales = tr._get_deep_supervision_scales()

        losses = []
        for step in range(args.steps):
            tr.current_epoch = step * args.epochs_per_step
            batch = synthetic_batch(2, args.patch, rng, scales)
            # Record gradients right before the optimizer step.
            orig_step = tr.optimizer.step

            def recording_step(*a, **k):
                nonlocal base_grad
                if base_param.grad is not None:
                    if not torch.isfinite(base_param.grad).all():
                        raise RuntimeError(f'Non-finite encoder gradient: {base_name}')
                    base_grad = max(base_grad, float(base_param.grad.abs().max()))
                for n, p in named:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise RuntimeError(f'Non-finite gradient: {n}')
                    if p.grad is not None and torch.count_nonzero(p.grad) > 0:
                        got_grad[n] = True
                        max_grad[n] = max(max_grad[n], float(p.grad.abs().max()))
                return orig_step(*a, **k)

            tr.optimizer.step = recording_step
            out = tr.train_step(batch)
            tr.optimizer.step = orig_step
            losses.append(float(out["loss"]))
            if not np.isfinite(losses[-1]):
                raise RuntimeError(f'Non-finite training loss at step {step}')

        moved = {n: not torch.equal(init[n], p.detach()) for n, p in named}
        bad = [n for n, _ in named if not got_grad[n]]
        if base_grad == 0 or torch.equal(base_initial, base_param.detach()):
            bad.append('encoder: no non-zero gradient or parameter update')
        if env.get('PHASER_LCA') == '1' and not (
                paths['augmentation_calls'] == args.steps and paths['changed_batches'] > 0):
            bad.append('A11: augmentation path not exercised')
        if env.get('PHASER_CC_LOSS') == '1' and not (
                paths['component_loss_calls'] == args.steps and paths['positive_component_losses'] > 0):
            bad.append('A12: component loss path not exercised')
        if float(env.get('PHASER_LCA_CONS_WEIGHT', '0')) > 0 and not (
                paths['consistency_calls'] > 0 and paths['positive_consistency_losses'] > 0):
            bad.append('A13b: consistency loss path not exercised')
        weak = [n for n, _ in named if got_grad[n] and max_grad[n] < 1e-6]
        groups = [(len(g["params"]), g["lr"], g["weight_decay"]) for g in tr.optimizer.param_groups]
        print(f"\n[{group}] {len(named)} PhASER tensors, loss {losses[0]:.3f} -> {losses[-1]:.3f}")
        print(f"   optimizer groups (n, lr, wd): {groups}")
        print(f"   encoder max|grad|={base_grad:.2e}; code paths: {paths}")
        if hasattr(net, "pfcr_modules"):
            for i, m in enumerate(net.pfcr_modules):
                pc = m._last_pc
                print(f"   PFCR{i}: gamma={float(m.residual_gate.detach()):+.4f} "
                      f"k={float(torch.nn.functional.softplus(m.pc.noise_k.detach())):.3f} "
                      f"PC>0 frac={float((pc > 0).float().mean()):.3f}")
        if args.verbose:
            for n, p in named:
                if p.numel() <= 8:
                    print(f"   {n:45s} max|grad|={max_grad[n]:.2e} max|delta|="
                          f"{float((p.detach() - init[n]).abs().max()):.2e}")
        for n in bad:
            print(f"   FAIL {n}: no gradient in {args.steps} steps")
        for n in weak:
            print(f"   WARN {n}: max|grad|={max_grad[n]:.1e} moved={'yes' if moved[n] else 'no'}")
        print(f"   {'PASS' if not bad else 'FAIL'} ({len(named) - len(bad) - len(weak)} ok, "
              f"{len(weak)} weak, {len(bad)} dead)")
        return not bad
    finally:
        trainer_mod.lesion_contrast_augment = original_aug
        trainer_mod.component_dicece_loss = original_cc
        trainer_mod.contrast_consistency_loss = original_cons
        for k in list(os.environ):
            if k.startswith("PHASER_"):
                del os.environ[k]
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+",
                    default=sorted(p.stem for p in CONFIGS.glob("A*.env")))
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--stages", type=int, default=6)
    ap.add_argument("--epochs-per-step", type=int, default=1,
                    help="advance current_epoch per step so the aux-loss ramp is exercised")
    ap.add_argument("--verbose", action="store_true", help="print grad/update size of small tensors")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ok = []
    for group in args.groups:
        ok.append(check_group(group, args))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nSUMMARY:", ", ".join(f"{g}={'PASS' if o else 'FAIL'}" for g, o in zip(args.groups, ok)))
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
