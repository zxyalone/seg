#!/usr/bin/env python3
"""External validation on MSD Task08 (Memorial Sloan Kettering; independent
of LiTS, overlap check passed) with the frozen test protocol's metrics and
statistics (plan §11, 2026-10-13, registered before inference).

Reuses test_protocol.py unchanged except for the scope: groups A0, A1, A13b
(3 seeds each; user decision), which keeps both confirmatory comparisons
(A13b vs A1 primary, A13b vs A0 secondary, Holm). Split 'external_msd08':
predictions in fold_0/external_msd08_pred/, evaluate.py output in
fold_0/eval_external_msd08/, reference labels from prepare_msd08.py.

    python external_msd08_protocol.py metrics --split external_msd08 --jobs 12 --out <dir>
    python external_msd08_protocol.py stats --tables <dir> --out <dir>
"""
from pathlib import Path

import test_protocol as TP

DATASET = Path('/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08')
TP.GROUPS = ('A0', 'A1', 'A13b')
TP.DESCRIPTIVE = ()
TP.SPLITS['external_msd08'] = dict(pred_subdir='external_msd08_pred', eval_subdir='eval_external_msd08',
                                   gt_dir=DATASET / 'labelsTs', cases_file=DATASET / 'cases.txt')

if __name__ == '__main__':
    TP.main()
