#!/usr/bin/env python3
"""External validation on 3D-IRCADb with the frozen test protocol's metrics and
statistics (plan §11, registered before inference).

Reuses test_protocol.py unchanged and only adds the 'external' split:
predictions in fold_0/external_pred/, evaluate.py output in fold_0/eval_external/,
labels and case list from the converted dataset.

    python external_protocol.py metrics --split external --jobs 16 --out <dir>
    python external_protocol.py stats --tables <dir> --out <dir>
"""
import sys
from pathlib import Path

import test_protocol as TP

DATASET = Path('/root/autodl-tmp/ext_data/Dataset103_3DIRCADb')
TP.SPLITS['external'] = dict(pred_subdir='external_pred', eval_subdir='eval_external',
                             gt_dir=DATASET / 'labelsTs', cases_file=DATASET / 'cases.txt')

if __name__ == '__main__':
    TP.main()
