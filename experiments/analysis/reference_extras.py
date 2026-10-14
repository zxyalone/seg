#!/usr/bin/env python3
"""Further use of the TriALS reference models (plan §11, 2026-10-14,
registered before computing; descriptive, single seed, training conditions
differ from this study).

The TriALS test predictions come from train_new_split/09_test_inference.sh:
checkpoint_best (EMA pseudo Dice on the validation fold), --disable_tta.

params    trainable/stored parameters from each checkpoint's network weights
          (shared tensors counted once);
          SAMed checkpoints store only the LoRA and decoder weights, so the
          frozen SAM image encoder is added from the original SAM checkpoint.
tta       existing mirror-TTA test predictions (checkpoint_best) of nnU-Net
          3D, SegResNet and U-Mamba scored with the frozen protocol pipeline.
boundary  boundary_metrics.py definitions on the test predictions of all
          reference models (no TTA).

    python reference_extras.py params --out <dir>
    python reference_extras.py tta --out <dir> --jobs 3
    python reference_extras.py boundary --out <dir> --jobs 24
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import reference_baselines_eval as RB
from boundary_metrics import region_metrics
from missed_lesion_probability import E, _drop_small

OURS = Path('/root/autodl-tmp/nnUNet_data/nnUNet_results_v2')
SAM = Path('/root/autodl-tmp/TriALS-main/checkpoints')
TTA = ('nnUNet3D', 'SegResNet', 'UMambaBot')


def numel(sd):
    """Parameters in a saved state dict. nnU-Net-style decoders keep a reference to
    the encoder, so the same tensor can appear under several keys (encoder.* and
    decoder.encoder.*); each storage is counted once. Integer buffers (e.g. Swin
    relative-position indices) and BatchNorm statistics are not parameters.
    Corrected 2026-10-14 (plan §11): the first version summed every key."""
    seen = {}
    for k, v in sd.items():
        if 'running_' in k or 'num_batches' in k or not v.is_floating_point():
            continue
        seen.setdefault((v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape)), v.numel())
    return sum(seen.values())


def cmd_params(a):
    import torch
    rows = []
    for name, folder in RB.BASELINES.items():
        sd = torch.load(RB.TRIALS / folder / 'fold_0/checkpoint_best.pth', map_location='cpu',
                        weights_only=False)['network_weights']
        stored = numel(sd)
        if name.startswith('SAMed'):
            frozen = numel(torch.load(SAM / ('sam_vit_b_01ec64.pth' if name == 'SAMed-B' else 'sam_vit_h_4b8939.pth'),
                                      map_location='cpu'))
            lora = sum(v.numel() for k, v in sd.items() if k.startswith('w_'))
            rows.append(dict(model=name, params_m=(frozen + lora) / 1e6, trainable_m=stored / 1e6,
                             note='SAM total + LoRA; trainable = stored LoRA + decoder weights'))
            continue
        rows.append(dict(model=name, params_m=stored / 1e6, trainable_m=stored / 1e6, note=''))
    for g in ('A0', 'A1', 'A13b'):
        sd = torch.load(OURS / f'{g}_s0' / RB.TRAINER / 'checkpoint_final.pth', map_location='cpu',
                        weights_only=False)['network_weights']
        rows.append(dict(model=g, params_m=numel(sd) / 1e6, trainable_m=numel(sd) / 1e6, note=f'exact count {numel(sd)}'))
    P = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / 'params.csv', index=False)
    print(P.round(3).to_string(index=False))


def _tta_eval(name, out):
    root = Path(out) / 'stage'
    run = f'{name}+TTA'
    fold = root / run / RB.TRAINER
    fold.mkdir(parents=True, exist_ok=True)
    link = fold / 'test_pred'
    if not link.exists():
        link.symlink_to(RB.TRIALS / RB.BASELINES[name] / 'fold_0/test_predictions_tta')
    return RB.evaluate(run, root)


def cmd_tta(a):
    with ProcessPoolExecutor(a.jobs) as ex:
        parts = list(ex.map(_tta_eval, TTA, [a.out] * len(TTA)))
    C = pd.concat([c for c, _ in parts]); L = pd.concat([l for _, l in parts])
    out = Path(a.out)
    C.to_csv(out / 'tta_cases_pp4mm.csv', index=False); L.to_csv(out / 'tta_lesions_pp4mm.csv', index=False)
    T = RB.summarise(L, C)
    T.to_csv(out / 'tta_table.csv', index=False)
    print(T.round(2).to_string(index=False))


def _boundary(args):
    name, case = args
    gimg, gt = E.read(RB.GT / f'{case}.nii.gz')
    _, pred = E.read(RB.TRIALS / RB.BASELINES[name] / 'fold_0/test_predictions' / f'{case}.nii.gz')
    assert pred.shape == gt.shape, (name, case)
    sp = tuple(float(s) for s in gimg.GetSpacing()[::-1])
    diag = float(np.linalg.norm(np.array(gt.shape) * np.array(sp)))
    rows = []
    if (gt == 2).any():
        pt = _drop_small(pred == 2, float(np.prod(sp)), 4.0)
        rows.append(dict(run=name, case=case, region='tumour', **region_metrics(pt, gt == 2, sp, diag)))
    rows.append(dict(run=name, case=case, region='liver', **region_metrics(pred > 0, gt > 0, sp, diag)))
    return rows


def cmd_boundary(a):
    cases = RB.CASES.read_text().split()
    with ProcessPoolExecutor(a.jobs) as ex:
        R = pd.DataFrame([r for rows in ex.map(_boundary, [(n, c) for n in RB.BASELINES for c in cases]) for r in rows])
    out = Path(a.out)
    R.to_csv(out / 'boundary_cases.csv', index=False)
    S = R.groupby(['region', 'run']).agg(n_cases=('case', 'nunique'), pred_empty_cases=('pred_empty', 'sum'),
                                         **{m: (m, 'mean') for m in ('dice', 'voe', 'rvd', 'assd', 'hd95', 'mssd')})
    S.reset_index().to_csv(out / 'boundary_summary.csv', index=False)
    print(S.round(2).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('params', 'tta', 'boundary'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dict(params=cmd_params, tta=cmd_tta, boundary=cmd_boundary)[a.cmd](a)


if __name__ == '__main__':
    main()
