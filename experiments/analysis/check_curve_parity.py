#!/usr/bin/env python3
"""Parity check between re-inferred validation curves and the saved validation
masks (plan §11, 2026-10-12). Bitwise parity of the unfiltered output is not
attainable: training-time validation runs with cudnn.benchmark = True, so
near-tie liver/tumor voxels can flip between runs and create or remove a
sub-4-mm component. The curves are read after the fixed 4-mm filter, so the
check is made at that level:

  1  unfiltered lesion detections (argmax) == eval_val, every lesion;
  2  after the 4-mm filter, per-case FP count and per-lesion detection of the
     re-inferred argmax == those of the saved validation masks, exactly;
  3  unfiltered FP count differs from eval_val by at most 1 per case
     (reported, never hidden).

No saved mask replaces a re-inferred result.

    python experiments/analysis/check_curve_parity.py --curves <out>/cases \\
        --runs A0_s0 A0_s1 ... --out <dir>
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from lowcnr_pilot_metrics import metrics


def run_rows(run, args, cases):
    _, saved_c, saved_l = metrics(run, args, cases)       # saved masks, fixed 4-mm filter
    saved_c = saved_c.set_index('case'); saved_l = saved_l.set_index(['case', 'lesion'])
    ev_c = pd.read_csv(Path(args.eval_root) / run / 'eval_val/cases.csv').set_index('case')
    ev_l = pd.read_csv(Path(args.eval_root) / run / 'eval_val/lesions.csv').set_index(['case', 'lesion'])
    rows = []
    for case in cases:
        d = json.loads((Path(args.curves) / run / f'{case}.json').read_text())
        les = {l['lesion']: l for l in d['lesions']}
        raw_det = all(les[i]['det@argmax'] == bool(ev_l.loc[(case, i), 'detected@0.1']) for i in les)
        pp_det = all(les[i]['det_pp@argmax'] == bool(saved_l.loc[(case, i), 'detected']) for i in les)
        raw_fp_diff = d['rules']['argmax']['fp'] - int(ev_c.loc[case, 'les_fp@0.1'])
        pp_fp_diff = d['rules']['argmax_pp']['fp'] - int(saved_c.loc[case, 'fp'])
        rows.append(dict(run=run, case=case, n_lesions=len(les), raw_detections_agree=raw_det,
                         pp_detections_agree=pp_det, raw_fp_diff=raw_fp_diff, pp_fp_diff=pp_fp_diff,
                         export_logits_dtype=d.get('export_logits_dtype', 'unrecorded')))
    print(f'{run}: parity rows written', flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--curves', required=True, help='cases/ folder written by missed_lesion_probability.py run')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--results-root', default='/root/autodl-tmp/nnUNet_data/nnUNet_results_v2')
    p.add_argument('--gt-dir', default='/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/labelsTr')
    p.add_argument('--cases-file', default=str(Path(__file__).resolve().parents[1] / 'eval/val_cases.txt'))
    p.add_argument('--eval-root', default=str(Path(__file__).resolve().parents[2] / 'results'))
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=1, help='runs checked in parallel processes')
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases = Path(args.cases_file).read_text().split()
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            parts = list(ex.map(run_rows, args.runs, [args] * len(args.runs), [cases] * len(args.runs)))
    else:
        parts = [run_rows(run, args, cases) for run in args.runs]
    rows = [r for part in parts for r in part]
    T = pd.DataFrame(rows)
    T['passed'] = T.raw_detections_agree & T.pp_detections_agree & (T.pp_fp_diff == 0) & (T.raw_fp_diff.abs() <= 1)
    T.to_csv(out / 'curve_parity_cases.csv', index=False)
    per_run = T.groupby('run').agg(cases=('case', 'size'), raw_det=('raw_detections_agree', 'all'),
                                   pp_det=('pp_detections_agree', 'all'),
                                   pp_fp_abs_diff=('pp_fp_diff', lambda s: int(s.abs().sum())),
                                   raw_fp_abs_diff=('raw_fp_diff', lambda s: int(s.abs().sum())),
                                   raw_fp_max_case=('raw_fp_diff', lambda s: int(s.abs().max())),
                                   dtypes=('export_logits_dtype', lambda s: ','.join(sorted(set(s)))),
                                   passed=('passed', 'all')).reset_index()
    per_run.to_csv(out / 'curve_parity_runs.csv', index=False)
    ok = bool(per_run.passed.all() and len(per_run) == len(args.runs) and (per_run.cases == len(cases)).all())
    (out / 'curve_parity.json').write_text(json.dumps(dict(
        passed=ok, runs=args.runs, n_cases=len(cases),
        criteria=['unfiltered detections == eval_val', 'after 4 mm: FP per case and detections == saved masks',
                  '|unfiltered FP - eval_val| <= 1 per case'],
        raw_fp_differences=T[T.raw_fp_diff != 0][['run', 'case', 'raw_fp_diff']].to_dict('records')), indent=2) + '\n')
    print(per_run.to_string(index=False))
    print('PARITY', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
