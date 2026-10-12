#!/usr/bin/env python3
"""Exact fixed-4-mm pilot metrics from saved validation masks; no inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from missed_lesion_probability import E, _drop_small

TRAINER = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'


def metrics(run, args, cases):
    """Fixed 4-mm metrics from saved masks. args.pred_subdir / args.eval_subdir
    select validation (default: validation/, eval_val/) or test (test_pred/,
    eval_test/) outputs of the same run."""
    folder = Path(args.results_root) / run / TRAINER
    pred_subdir = getattr(args, 'pred_subdir', 'validation')
    eval_subdir = getattr(args, 'eval_subdir', 'eval_val')
    ref = pd.read_csv(folder / eval_subdir / 'lesions.csv').set_index(['case', 'lesion'])
    refc = pd.read_csv(folder / eval_subdir / 'cases.csv').set_index('case')
    assert set(refc.index) == set(cases)
    rows, lesions = [], []
    for case in cases:
        im, gt = E.read(Path(args.gt_dir) / f'{case}.nii.gz')
        _, pred = E.read(folder / pred_subdir / f'{case}.nii.gz')
        assert pred.shape == gt.shape
        gt_mask = gt == 2
        gl, ng = E.components(gt_mask, 26)
        pl, npred = E.components(pred == 2, 26)
        raw, _ = E.match_lesions(gl, ng, pl, npred, .1)
        assert npred - sum(v is not None for v in raw.values()) == refc.loc[case, 'les_fp@0.1']
        pp = _drop_small(pred == 2, float(np.prod(im.GetSpacing())), 4.0)
        pl, npred = E.components(pp, 26)
        match, _ = E.match_lesions(gl, ng, pl, npred, .1)
        for i in range(1, ng + 1):
            r = ref.loc[(case, i)]
            assert (raw[i] is not None) == bool(r['detected@0.1'])
            lesions.append(dict(run=run, case=case, lesion=i, cnr_stratum=r.cnr_stratum,
                                size_stratum=r.size_stratum, diameter_mm=r.diameter_mm, cnr=r.cnr,
                                detected=match[i] is not None))
        tp = int((pp & gt_mask).sum())
        fpvox = int(pp.sum()) - tp
        fn = int(gt_mask.sum()) - tp
        rows.append(dict(run=run, case=case, fp=npred - sum(v is not None for v in match.values()),
                         tp_vox=tp, fp_vox=fpvox, fn_vox=fn, has_tumor=bool(gt_mask.any()),
                         tumor_dice=2*tp/max(2*tp+fpvox+fn, 1)))
        print(f'{run} {case}: fixed 4-mm metrics complete', flush=True)
    L, C = pd.DataFrame(lesions), pd.DataFrame(rows)
    low = L[L.cnr_stratum == 'low']
    ntp = int(L.detected.sum()); nfp = int(C.fp.sum())
    summary = dict(n_cases=len(C), n_lesions=len(L), n_low_cnr=len(low),
                   recall_low_pct=100*float(low.detected.mean()),
                   recall_pct=100*float(L.detected.mean()), fp_per_scan=nfp/len(C),
                   tumor_dice_pct=100*float(C[C.has_tumor].tumor_dice.mean()),
                   f1_pct=100*2*ntp/max(len(L)+ntp+nfp, 1), raw_consistency_passed=True)
    return summary, C, L


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', default='/root/autodl-tmp/nnUNet_data/nnUNet_results_v2')
    p.add_argument('--gt-dir', default='/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/labelsTr')
    p.add_argument('--cases-file', default=str(Path(__file__).resolve().parents[1]/'eval/val_cases.txt'))
    p.add_argument('--runs', nargs='+', default=['A1_s0', 'A9_s0', 'A11_s0', 'A12_s0'])
    p.add_argument('--out', required=True)
    args = p.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases = Path(args.cases_file).read_text().split()
    assert len(cases) == len(set(cases)) == 21
    summaries, cs, ls = {}, [], []
    for run in args.runs:
        s, c, l = metrics(run, args, cases); summaries[run] = s; cs.append(c); ls.append(l)
    decisions = {}
    if 'A1_s0' in summaries:
        baseline = summaries['A1_s0']
        for run, s in summaries.items():
            if run == 'A1_s0':
                continue
            gain = s['recall_low_pct'] - baseline['recall_low_pct']
            fpdelta = s['fp_per_scan'] - baseline['fp_per_scan']
            decisions[run] = dict(low_cnr_gain_pp=gain, fp_per_scan_delta=fpdelta,
                                  passed=bool(gain >= 5 and fpdelta <= .3))
    pd.concat(cs).to_csv(out/'cases_pp4mm.csv', index=False)
    pd.concat(ls).to_csv(out/'lesions_pp4mm.csv', index=False)
    (out/'summary_pp4mm.json').write_text(json.dumps(summaries, indent=2)+'\n')
    (out/'decisions_pp4mm.json').write_text(json.dumps(decisions, indent=2)+'\n')
    lines = ['# Fixed 4-mm pilot metrics', '', 'All 21 validation cases; 26-connectivity; IoU > 0.1 Hungarian matching. No inference or threshold tuning.', '',
             '| Run | Low-CNR recall (%) | FP/scan | Tumor Dice (%) | F1 (%) |', '|---|---:|---:|---:|---:|']
    for run, s in summaries.items():
        lines.append(f"| {run} | {s['recall_low_pct']:.3f} | {s['fp_per_scan']:.3f} | {s['tumor_dice_pct']:.3f} | {s['f1_pct']:.3f} |")
    lines += ['', 'Pre-registered gate: low-CNR recall >= A1_s0 + 5 pp AND FP/scan <= A1_s0 + 0.3.', '',
              'Every pilot is reported. These results are gates, not manuscript evidence. A0 + threshold adjustment and recall at matched FP/scan remain required comparisons.', '']
    for run, d in decisions.items():
        lines.append(f"- {run}: {'PASS' if d['passed'] else 'STOP'}; gain {d['low_cnr_gain_pp']:+.3f} pp; FP/scan delta {d['fp_per_scan_delta']:+.3f}")
    (out/'PILOT_DECISION.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
