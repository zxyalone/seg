#!/usr/bin/env python3
"""
E16 / Table 7: parameters, forward latency and peak memory per 128^3 patch for
each ablation group, using the same network construction as training.

    PYTHONPATH=nnUNet python experiments/cost.py --groups A0 A1 A6 --device cuda
Whole-case inference time comes from nnUNetv2_predict (time the 27-case run of
predict_test.sh and divide by 27).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_grads as cg  # noqa: E402  (reuses its plan-free trainer construction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["A0", "A1", "A6"])
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    base = None
    print(f"{'group':6} {'params':>12} {'PhASER':>9} {'overhead':>9} {'fwd ms':>8} {'peak GB':>8}")
    for g in args.groups:
        for k in list(os.environ):
            if k.startswith("PHASER_"):
                del os.environ[k]
        os.environ.update(cg.read_env(g))
        os.environ["VOCOMNI_NNUNET_CKPT"] = ""
        tr = cg.make_trainer(args.patch, dev, 6)
        net = tr.network.eval()
        net.decoder.deep_supervision = False
        n = sum(p.numel() for p in net.parameters())
        ph = sum(p.numel() for p in net.get_phaser_params())
        base = base or (n - ph)
        x = torch.randn(1, 1, *(args.patch,) * 3, device=dev)
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            for _ in range(3):
                net(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                net(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / args.reps * 1000
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
        print(f"{g:6} {n:12,d} {ph:9,d} {100 * ph / base:8.3f}% {ms:8.1f} {peak:8.2f}")


if __name__ == "__main__":
    main()
