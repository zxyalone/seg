#!/usr/bin/env python3
"""Fixed 4-mm fold-0 seed replication; implements frozen plan section 6c."""
import argparse
import json
from pathlib import Path

import pandas as pd
from lowcnr_pilot_metrics import metrics

GROUPS = ('A13b', 'A13a', 'A11')  # Registered priority, never sorted by score.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', default='/root/autodl-tmp/nnUNet_data/nnUNet_results_v2')
    p.add_argument('--gt-dir', default='/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/labelsTr')
    p.add_argument('--cases-file', default=str(Path(__file__).resolve().parents[1]/'eval/val_cases.txt'))
    p.add_argument('--out', required=True)
    args = p.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases = Path(args.cases_file).read_text().split()
    assert len(cases) == len(set(cases)) == 21
    runs = [f'{g}_s{s}' for g in ('A1', 'A11', 'A13a', 'A13b') for s in range(3)]
    summaries, cs, ls = {}, [], []
    for run in runs:
        summary, c, l = metrics(run, args, cases)
        assert summary['n_lesions'] == 92 and summary['n_low_cnr'] == 39
        summaries[run] = summary; cs.append(c); ls.append(l)
        # Save progress after each completed run without issuing a partial gate.
        (out/'summary_pp4mm.json').write_text(json.dumps(summaries, indent=2)+'\n')
    pd.concat(cs).to_csv(out/'cases_pp4mm.csv', index=False)
    pd.concat(ls).to_csv(out/'lesions_pp4mm.csv', index=False)
    frame = pd.DataFrame.from_dict(summaries, orient='index').rename_axis('run').reset_index()
    frame['group'] = frame.run.str.split('_s').str[0]
    frame['seed'] = frame.run.str.split('_s').str[1].astype(int)
    frame.to_csv(out/'per_seed_pp4mm.csv', index=False)
    base = frame[frame.group == 'A1'][['recall_low_pct', 'fp_per_scan']].mean()
    decisions = {}
    for group in GROUPS:
        part = frame[frame.group == group]
        assert set(part.seed) == {0, 1, 2}
        gain = float(part.recall_low_pct.mean()-base.recall_low_pct)
        fpdelta = float(part.fp_per_scan.mean()-base.fp_per_scan)
        single = (part.recall_low_pct >= base.recall_low_pct+5) & (part.fp_per_scan <= base.fp_per_scan+.3)
        decisions[group] = dict(mean_low_cnr_recall_pct=float(part.recall_low_pct.mean()),
                                mean_fp_per_scan=float(part.fp_per_scan.mean()),
                                gain_low_cnr_pp=gain, fp_per_scan_delta=fpdelta,
                                seeds_passing_against_A1_mean=part[single].seed.tolist(),
                                passed=bool(gain >= 5 and fpdelta <= .3 and single.sum() >= 2))
    chosen = next((g for g in GROUPS if decisions[g]['passed']), None)
    result = dict(baseline_A1_mean=base.to_dict(), priority=list(GROUPS), groups=decisions,
                  first_eligible_group=chosen, test_set_evaluated=False,
                  scope='Validation replication only; test protocol must be frozen separately by the user.')
    (out/'replication_decisions.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Fold-0 low-CNR seed replication', '',
             'All 21 validation cases, seeds 0/1/2; fixed 4-mm removal; IoU >0.1 Hungarian matching, 26-connectivity. Unfiltered detections and FP counts reproduce eval_val.', '',
             '| Run | Low-CNR recall (%) | FP/scan | Tumor Dice (%) |', '|---|---:|---:|---:|']
    for run, s in summaries.items():
        lines.append(f"| {run} | {s['recall_low_pct']:.3f} | {s['fp_per_scan']:.3f} | {s['tumor_dice_pct']:.3f} |")
    lines += ['', 'Frozen section 6c: mean recall gain >=5 pp; mean FP/scan increase <=0.3; at least two individual seeds pass both conditions against the A1 three-seed mean.', '',
              '| Group | Mean recall gain (pp) | Mean FP/scan change | Passing seeds | Eligible |', '|---|---:|---:|---|---|']
    for g, d in decisions.items():
        lines.append(f"| {g} | {d['gain_low_cnr_pp']:+.3f} | {d['fp_per_scan_delta']:+.3f} | {d['seeds_passing_against_A1_mean']} | {'PASS' if d['passed'] else 'STOP'} |")
    lines += ['', f"First eligible group in registered order A13b -> A13a -> A11: **{chosen or 'none'}**.", '',
              'All groups are reported. Validation was used for development; these results do not establish independent efficacy. No test-set inference or protocol-freezing action is performed by this script.', '']
    (out/'REPLICATION_DECISION.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
