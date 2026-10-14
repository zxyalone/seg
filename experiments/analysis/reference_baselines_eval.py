#!/usr/bin/env python3
"""Unified test-set evaluation of the TriALS reference architectures (plan §11,
2026-10-13; descriptive, not part of the confirmatory tests).

Their saved test predictions (no TTA, checkpoint_best as written by TriALS
train_new_split/09_test_inference.sh; corrected 2026-10-14, plan §11; Dataset101, same
83/21/27 split, 100 epochs, no pretraining, one seed) are scored with exactly
the pipeline of the frozen protocol: evaluate.py against the same labels,
26-connectivity, IoU > 0.1 Hungarian matching, frozen CNR tertiles, then the
fixed 4-mm filter (lowcnr_pilot_metrics.metrics). No inference is run.

    python reference_baselines_eval.py --out <dir> --ours <results/test_protocol/tables> --jobs 8
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from lowcnr_pilot_metrics import metrics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = Path('/root/autodl-tmp/nnUNet_data')
TRIALS = DATA / 'nnUNet_results/Dataset101_LiTS'
TRAINER = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
BASELINES = {
    'nnUNet3D': 'nnUNetTrainer_100epochs__nnUNetPlansV221__3d_fullres',
    'SegResNet': 'nnUNetTrainerSegResNet_100epochs__nnUNetPlansV221__3d_fullres',
    'UMambaBot': 'nnUNetTrainerUMambaBot_100epochs__nnUNetPlansV221__3d_fullres',
    'MedNeXt-L': 'nnUNetTrainerV2_MedNeXt_L_kernel5_100epochs__nnUNetPlansV221__3d_fullres',
    'SwinUNETR': 'nnUNetTrainerSwinUNETR_100epochs__nnUNetPlansV221__3d_fullres',
    'nnUNet2D': 'nnUNetTrainer_100epochs__nnUNetPlansV221__2d',
    'SAMed-B': 'nnUNetTrainerV2_SAMed_b_r_4_100epochs__nnUNetPlansV221__2d',
    'SAMed-H': 'nnUNetTrainerV2_SAMed_h_r_4_100epochs__nnUNetPlansV221__2d',
}
GT = DATA / 'nnUNet_raw/Dataset101_LiTS/labelsTr'
IMAGES = DATA / 'nnUNet_raw/Dataset102_LiTS/imagesTs'
CASES = REPO / 'experiments/eval/test_cases.txt'


def stage(name, root):
    fold = root / name / TRAINER
    fold.mkdir(parents=True, exist_ok=True)
    link = fold / 'test_pred'
    if not link.exists():
        link.symlink_to(TRIALS / BASELINES[name] / 'fold_0/test_predictions')
    return fold


def evaluate(name, root):
    fold = stage(name, root)
    if not (fold / 'eval_test/summary.json').is_file():
        subprocess.run([sys.executable, str(REPO / 'experiments/eval/evaluate.py'), '--pred-dir', str(fold / 'test_pred'),
                        '--gt-dir', str(GT), '--image-dir', str(IMAGES), '--cases-file', str(CASES),
                        '--out-dir', str(fold / 'eval_test'), '--workers', '2'], check=True)
    ns = argparse.Namespace(results_root=str(root), gt_dir=str(GT), pred_subdir='test_pred', eval_subdir='eval_test')
    _, c, l = metrics(name, ns, CASES.read_text().split())
    return c, l


def summarise(L, C):
    rows = []
    for run in C.run.unique():
        l, c = L[L.run == run], C[C.run == run]
        tp, fp = int(l.detected.sum()), int(c.fp.sum())
        rows.append(dict(run=run, recall_low=100 * l[l.cnr_stratum == 'low'].detected.mean(), recall_all=100 * l.detected.mean(),
                         fp_per_scan=fp / len(c), tumor_dice=100 * c[c.has_tumor].tumor_dice.mean(),
                         f1=100 * 2 * tp / max(len(l) + tp + fp, 1)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ours', required=True, help='tables/ of the frozen test protocol (lesions_pp4mm.csv, cases_pp4mm.csv)')
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); root = out / 'stage'; root.mkdir(parents=True, exist_ok=True)
    names = list(BASELINES)
    with ProcessPoolExecutor(a.jobs) as ex:
        parts = list(ex.map(evaluate, names, [root] * len(names)))
    C = pd.concat([c for c, _ in parts]); L = pd.concat([l for _, l in parts])
    C.to_csv(out / 'cases_pp4mm.csv', index=False); L.to_csv(out / 'lesions_pp4mm.csv', index=False)
    ref = summarise(L, C).assign(group=lambda d: d.run, seeds=1, training='TriALS: 100 epochs, from scratch')
    Lo = pd.read_csv(Path(a.ours) / 'lesions_pp4mm.csv'); Co = pd.read_csv(Path(a.ours) / 'cases_pp4mm.csv')
    ours = summarise(Lo, Co)
    ours['group'] = ours.run.str.rsplit('_s', n=1).str[0]
    ours = ours.groupby('group', sort=False).mean(numeric_only=True).reset_index()
    ours['seeds'] = [3 if g != 'A12' else 1 for g in ours.group]
    ours['training'] = 'this study: 200 epochs, VoComni init'
    T = pd.concat([ref.drop(columns='run'), ours], ignore_index=True)[
        ['group', 'training', 'seeds', 'recall_low', 'recall_all', 'fp_per_scan', 'tumor_dice', 'f1']]
    T.to_csv(out / 'reference_table.csv', index=False)
    strata = (L.groupby(['run', 'cnr_stratum', 'size_stratum']).detected.agg(['mean', 'size']).reset_index())
    strata.to_csv(out / 'reference_strata.csv', index=False)
    print(T.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
