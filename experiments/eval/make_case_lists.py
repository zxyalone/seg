#!/usr/bin/env python3
"""Write val_cases.txt / test_cases.txt from the nnU-Net split files and check
that train, validation and test are disjoint (validation_plan.txt §3.2).

    python experiments/eval/make_case_lists.py \
        --preprocessed /root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset102_LiTS
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed", required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    pre = Path(args.preprocessed)
    split = json.loads((pre / "splits_final.json").read_text())[args.fold]
    test = json.loads((pre / "test_set.json").read_text())["test"]
    tr, va, te = set(split["train"]), set(split["val"]), set(test)
    overlaps = {"train&val": tr & va, "train&test": tr & te, "val&test": va & te}
    for k, v in overlaps.items():
        if v:
            sys.exit(f"overlap {k}: {sorted(v)}")
    (HERE / "val_cases.txt").write_text("\n".join(sorted(va)) + "\n")
    (HERE / "test_cases.txt").write_text("\n".join(sorted(te)) + "\n")
    print(f"train {len(tr)} / val {len(va)} / test {len(te)}; disjoint. "
          f"Wrote {HERE / 'val_cases.txt'} and {HERE / 'test_cases.txt'}")


if __name__ == "__main__":
    main()
