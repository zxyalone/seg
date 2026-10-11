#!/usr/bin/env python3
"""Export E11 detection metrics and integer case counts without new inference."""
import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
import stats as S


def export(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    runs = [f'A6_s{s}' for s in range(3)]
    conditions = sorted(p.name for p in (source / 'eval' / runs[0]).iterdir() if p.is_dir())
    frames = []
    per_seed, pooled = [], []
    for condition in conditions:
        dirs = [source / 'eval' / run / condition for run in runs]
        for run, folder in zip(runs, dirs):
            raw = pd.read_csv(folder / 'cases.csv')
            assert len(raw) == 21 and raw.case.nunique() == 21
            frames.append(raw.assign(run=run, condition=condition))
            for cohort in ('Tumor', 'All'):
                data = raw[raw.has_tumor] if cohort == 'Tumor' else raw
                for threshold in ('0.1', '0.5'):
                    row = dict(run=run, condition=condition, cohort=cohort,
                               iou_threshold=float(threshold), n_cases=len(data),
                               tumor_dice=data.tumor_dice.mean(), **S.pooled(data, threshold))
                    row.update({c: data[f'{c}@{threshold}'].sum() for c in S.COUNT_COLS})
                    per_seed.append(row)
        for cohort in ('Tumor', 'All'):
            for threshold in ('0.1', '0.5'):
                data = S.load_method([str(p) for p in dirs], cohort, threshold)
                row = dict(condition=condition, cohort=cohort, iou_threshold=float(threshold),
                           n_seeds=3, n_cases=len(data), tumor_dice=data.tumor_dice.mean(),
                           **S.pooled(data, threshold))
                row.update({c + '_seed_mean': data[f'{c}@{threshold}'].sum() for c in S.COUNT_COLS})
                pooled.append(row)
    pd.concat(frames, ignore_index=True).to_csv(destination / 'e11_case_metrics.csv', index=False)
    pd.DataFrame(per_seed).to_csv(destination / 'e11_detection_per_seed.csv', index=False)
    result = pd.DataFrame(pooled)
    result.to_csv(destination / 'e11_detection_seed_pooled.csv', index=False)
    lines = ['# E11 detection metrics', '',
             'Three A6 seeds; 21 validation cases (18 tumor, 3 negative). Counts are averaged over',
             'seeds per case, then pooled before computing precision, recall and F1. These are',
             'ratios of pooled counts, not averages of seed F1 values or a prediction ensemble.', '',
             '`e11_case_metrics.csv` retains the original integer TP/FP/FN counts for every',
             'case, seed, condition and both IoU thresholds. `e11_detection_per_seed.csv`',
             'gives each seed separately. All CSV metrics are fractions; tables below use percent.', '',
             'Delta = intervention minus none. Paired confidence intervals for the original',
             'Tumor / IoU > 0.1 comparisons are in `e11_paired_vs_none.csv`; blank p-values',
             'for pooled lesion metrics mean no Wilcoxon test was performed, not p=0.', '',
             'The joint PC/prior-zero follow-up is separate and is not included in the original',
             'nine-comparison Holm family. Similar point estimates alone do not establish equivalence.', '']
    for cohort in ('Tumor', 'All'):
        selected = result[(result.cohort == cohort) & (result.iou_threshold == .1)]
        base = selected[selected.condition == 'none'].iloc[0]
        lines += [f'## {cohort}: IoU > 0.1', '',
                  '| Condition | FP/scan | Δ FP/scan | Precision (%) | Δ precision (pp) | F1 (%) | Δ F1 (pp) | Recall (%) |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in selected.itertuples(index=False):
            lines.append(f'| {row.condition} | {row.fp_per_scan:.3f} | {row.fp_per_scan-base.fp_per_scan:+.3f} | '
                         f'{100*row.lesion_precision:.2f} | {100*(row.lesion_precision-base.lesion_precision):+.2f} | '
                         f'{100*row.lesion_f1:.2f} | {100*(row.lesion_f1-base.lesion_f1):+.2f} | '
                         f'{100*row.lesion_recall:.2f} |')
        lines.append('')
    (destination / 'DETECTION_METRICS.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.out)
