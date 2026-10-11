#!/usr/bin/env python3
"""Check the optional s46 GPU filters against CPU before a full GPU run.

Uses one synthetic volume and the first fold-0 training case only. The bounds
were recorded in the detectability plan before this comparison was run.
"""
import argparse
import time

import numpy as np

import s46_detectability_probe as probe
from common import PLAN_SPACING_SITK, load_raw_resampled, save_json, splits


def compare(case, img, seg, spacing, device):
    records, timings = {}, {}
    for backend in ("cpu", device):
        probe.configure_device(backend)
        start = time.perf_counter()
        records[backend], _, _ = probe.process_volume(
            case, img, seg, spacing, np.random.default_rng(0))
        timings[backend] = time.perf_counter() - start
    cpu, gpu = records["cpu"], records[device]
    assert len(cpu) == len(gpu) > 0
    max_auc_diff = 0.0
    hits_match = properties_match = True
    for a, b in zip(cpu, gpu):
        for key in a:
            if key.startswith(("auc_bg_", "auc_local_")):
                av, bv = a[key], b[key]
                assert np.isfinite(av) == np.isfinite(bv), (case, key, av, bv)
                if np.isfinite(av):
                    max_auc_diff = max(max_auc_diff, abs(av - bv))
            elif key.startswith("hit_"):
                hits_match &= a[key] == b[key]
            elif not key.startswith("med_"):
                properties_match &= a[key] == b[key]
    result = dict(case=case, n_lesions=len(cpu), timings_seconds=timings,
                  max_auc_abs_diff=max_auc_diff, hits_match=bool(hits_match),
                  properties_match=bool(properties_match),
                  passed=bool(max_auc_diff <= 0.001 and hits_match and properties_match))
    print(result, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    if args.device == "cpu":
        ap.error("select a CUDA device to compare against CPU")
    results = []
    img, seg, sp, _ = probe.synth(np.random.default_rng(0), hetero=3.0)
    results.append(compare("synthetic_hetero3HU", img, seg, sp, args.device))
    del img, seg
    tr, va, te = splits()
    assert not (set(tr) & set(va) or set(tr) & set(te))
    case = tr[0]
    img, seg = load_raw_resampled(case)
    results.append(compare(case, img, seg, tuple(reversed(PLAN_SPACING_SITK)), args.device))
    report = dict(device=args.device, cupy_version=probe._GPU.__version__,
                  auc_tolerance=0.001, results=results,
                  passed=all(r["passed"] for r in results))
    save_json(report, "s46_backend_check.json")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
