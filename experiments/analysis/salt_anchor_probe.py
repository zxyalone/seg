#!/usr/bin/env python3
"""
SALT G0 (docs/SALT模型设计.md §3.2): do the spectral scale-space anchors reach the lesions the baseline misses?
Inference only, with the trained A0 seed 0 (VoComni nnU-Net, no SALT module); no training.

Per case, in nnU-Net's preprocessed space: A0 sliding-window softmax (no TTA); lesions = 26-connected
components of label 2 with >= 20 voxels; A0 detection = some predicted tumour component (26-connected,
>= 4 mm equivalent diameter) with IoU > 0.1; 'partially lit' = missed with max tumour probability inside the
lesion in [0.1, 0.5), 'unlit' = below 0.1. Anchors exactly as the network computes them
(phaser/lesion_query.spectral_anchors: 16 semantic + 16 image anchors) in every 128^3 sliding window
(step 0.5, nnU-Net's window positions). A lesion is covered when some anchor centre lies inside the lesion
dilated by 2 voxels and the anchor radius is within [r/3, 3r] of its equivalent radius r. Control: the same
number of anchors per window at random foreground positions with random bank radii (fixed seed).

  CUDA_VISIBLE_DEVICES=0 python experiments/analysis/salt_anchor_probe.py run --split msd08 --shard 0/3
  python experiments/analysis/salt_anchor_probe.py summarize
Outputs results/salt/g0/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "nnUNet"))
OUT = REPO / "results" / "salt" / "g0"
S26 = ndi.generate_binary_structure(3, 3)
PATCH = (128, 128, 128)
N_SEM, N_IMG = 16, 16
MSD08 = Path("/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08")
SPLITS = {
    "msd08": dict(images=MSD08 / "imagesTs", labels=MSD08 / "labelsTs", cases=MSD08 / "cases.txt"),
    "val": dict(images=Path("/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/imagesTr"),
                labels=Path("/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/labelsTr"),
                cases=REPO / "experiments/eval/val_cases.txt"),
}


def window_starts(shape, patch, step=0.5):
    from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
    return compute_steps_for_sliding_window(shape, patch, step)


def lesion_table(data, seg, spacing, prob, pred_lab, n_pred, min_voxels=20):
    """Per GT lesion: voxels, equivalent radius (mm), CNR (s42 ring rule), A0 status."""
    tumour, liver = seg == 2, seg == 1
    lab, n = ndi.label(tumour, structure=S26)
    vox = float(np.prod(spacing))
    pred_sz = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)
    rows = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m_full = lab == i
        nv = int(m_full.sum())
        if nv < min_voxels:
            continue
        lo = [max(s.start - 10, 0) for s in sl]
        hi = [min(s.stop + 10, d) for s, d in zip(sl, lab.shape)]
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        m, liv, oth = m_full[box], liver[box], (tumour & ~m_full)[box]
        d_out = ndi.distance_transform_edt(~m)
        ring = (~m) & liv & (d_out >= 2) & (d_out <= 6) & ~oth
        if ring.sum() < 20:
            ring = (~m) & liv & (d_out >= 1) & (d_out <= 6) & ~oth
        im = data[box]
        cnr = abs(float(im[m].mean()) - float(im[ring].mean())) / (float(im[ring].std()) + 1e-8) \
            if ring.sum() >= 10 else float("nan")
        # A0 detection: best IoU with a predicted component
        ids, inter = np.unique(pred_lab[m_full], return_counts=True)
        iou = 0.0
        for j, it in zip(ids, inter):
            if j:
                iou = max(iou, it / (nv + pred_sz[j] - it))
        pmax = float(prob[m_full].max())
        status = "detected" if iou > 0.1 else ("partial" if pmax >= 0.1 else "unlit")
        r_eq = (3 * nv * vox / (4 * np.pi)) ** (1 / 3)
        rows.append(dict(lesion=i, voxels=nv, radius_mm=r_eq, diameter_mm=2 * r_eq, cnr=cnr, iou=iou,
                         pmax=pmax, status=status))
    return lab, rows


def collect_anchors(xs, pts, fgs, spacing, device, rng):
    """Anchors of every 128^3 sliding window (volume coordinates) and the random control.
    Returns {name: array [M, 4] (z, y, x, radius_mm)}."""
    from nnunetv2.training.nnUNetTrainer.phaser.lesion_query import DEFAULT_RADII_MM, spectral_anchors
    shape = xs.shape
    anchors = {"ssa": [], "sem": [], "img": [], "rand": []}
    steps = window_starts(shape, PATCH)
    for z0 in steps[0]:
        for y0 in steps[1]:
            for x0 in steps[2]:
                w = (slice(z0, z0 + PATCH[0]), slice(y0, y0 + PATCH[1]), slice(x0, x0 + PATCH[2]))
                sem = torch.from_numpy(np.ascontiguousarray(pts[w]))[None].to(device)
                img = torch.from_numpy(np.ascontiguousarray(xs[w]))[None].to(device)
                fg = torch.from_numpy(np.ascontiguousarray(fgs[w]))[None].to(device)
                an = spectral_anchors(sem, img, fg, spacing, N_SEM, N_IMG)
                off = np.array([z0, y0, x0])
                v = an["valid"][0].cpu().numpy()
                cen = an["centre"][0].cpu().numpy().astype(int) + off
                rad = an["radius"][0].cpu().numpy()
                src = an["source"][0].cpu().numpy()
                for c, r, s_, ok in zip(cen, rad, src, v):
                    if ok:
                        anchors["ssa"].append((*c, r))
                        anchors["sem" if s_ == 0 else "img"].append((*c, r))
                # random control: the same number of anchors at random foreground positions
                flat = np.flatnonzero(fgs[w])
                if flat.size == 0:
                    flat = np.arange(int(np.prod(PATCH)))
                pick = np.unravel_index(flat[rng.integers(0, flat.size, int(v.sum()))], PATCH)
                for c in np.stack(pick, 1):
                    anchors["rand"].append((*(c + off), float(rng.choice(DEFAULT_RADII_MM))))
    return {k: np.asarray(v, dtype=np.float64).reshape(-1, 4) for k, v in anchors.items()}


def coverage(rows, lab, anchors):
    """Adds covered_<name> to every lesion row: an anchor centre inside the lesion dilated by 2 voxels with
    a radius in [r/3, 3r]."""
    objs = ndi.find_objects(lab)
    for r in rows:
        sl = objs[r["lesion"] - 1]
        lo = np.array([max(q.start - 2, 0) for q in sl])
        hi = np.array([min(q.stop + 2, n) for q, n in zip(sl, lab.shape)])
        crop = ndi.binary_dilation(lab[tuple(slice(a_, b_) for a_, b_ in zip(lo, hi))] == r["lesion"],
                                   structure=S26, iterations=2)
        for key, A in anchors.items():
            c = A[:, :3].astype(int)
            inb = np.all((c >= lo) & (c < hi), 1) & (A[:, 3] >= r["radius_mm"] / 3) & (A[:, 3] <= 3 * r["radius_mm"])
            cc = c[inb] - lo
            r[f"covered_{key}"] = bool(crop[cc[:, 0], cc[:, 1], cc[:, 2]].any()) if len(cc) else False


def cmd_run(a):
    import common as C
    sp = SPLITS[a.split]
    out = OUT / a.split
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    model = C.load_model("A0", 0, device)
    spacing = tuple(float(v) for v in model.predictor.configuration_manager.spacing)
    cases = C.shard([c for c in Path(sp["cases"]).read_text().split() if c.strip()], a.shard)
    rng = np.random.default_rng(0)
    for ci, case in enumerate(cases):
        f = out / f"{case}.json"
        if f.is_file():
            continue
        t0 = time.time()
        raw, props, seg_raw = C.read_raw(model.predictor, case, Path(sp["images"]), Path(sp["labels"]))
        data, seg, _ = C.preprocess(model.predictor, raw, props, seg_raw)
        logits = C.predict_logits(model, data)
        p = torch.softmax(logits.float(), 0)
        x, s0 = data[0].astype(np.float32), seg[0].astype(np.int16)
        pt = p[2].numpy()
        pred = (p.argmax(0) == 2).numpy()
        pred_lab, n_pred = ndi.label(pred, structure=S26)
        if n_pred:   # 4-mm post-processing of the frozen protocol
            sizes = np.bincount(pred_lab.ravel(), minlength=n_pred + 1) * float(np.prod(spacing))
            small = (6 * sizes / np.pi) ** (1 / 3) < 4.0
            small[0] = False
            pred_lab[small[pred_lab]] = 0
        lab, rows = lesion_table(x, s0, spacing, pt, pred_lab, n_pred)
        # pad to at least one patch, as nnU-Net does
        shape = tuple(max(n, q) for n, q in zip(x.shape, PATCH))
        padw = [(0, n - m) for n, m in zip(shape, x.shape)]
        xs, pts = np.pad(x, padw, constant_values=x.min()), np.pad(pt, padw)
        fgs = np.pad((p[1] + p[2]).numpy() > 0.5, padw)
        anchors = collect_anchors(xs, pts, fgs, spacing, device, rng)
        coverage(rows, lab, anchors)
        C.write_json(dict(case=case, n_anchors={k: int(len(v)) for k, v in anchors.items()}, rows=rows), f)
        print(f"[{ci + 1}/{len(cases)}] {case}: {len(rows)} lesions, {len(anchors['ssa'])} anchors, "
              f"{time.time() - t0:.0f}s", flush=True)


def _size(d):
    return "<10" if d < 10 else ("10-20" if d <= 20 else ">20")


def _cnr_tier(c):
    if not np.isfinite(c):
        return "na"
    return "low" if c < 1.2785 else ("mid" if c < 1.9208 else "high")


def cmd_summarize(a):
    import pandas as pd
    import common as C
    res, lines = {}, ["# SALT G0：谱尺度空间锚点能否覆盖 A0 漏检的病灶（docs/SALT模型设计.md §3.2）", ""]
    for split in SPLITS:
        files = sorted((OUT / split).glob("*.json"))
        if not files:
            continue
        d = pd.DataFrame([dict(case=json.loads(f.read_text())["case"], **r) for f in files
                          for r in json.loads(f.read_text())["rows"]])
        d["size"] = d.diameter_mm.map(_size)
        d["cnr_tier"] = d.cnr.map(_cnr_tier)
        d["missed"] = d.status != "detected"
        d.to_csv(OUT / f"lesions_{split}.csv", index=False)
        cov = d.groupby("status")[["covered_ssa", "covered_sem", "covered_img", "covered_rand"]].mean() * 100
        miss = d[d.missed]
        c_m = float(miss.covered_ssa.mean() * 100) if len(miss) else float("nan")
        c_r = float(miss.covered_rand.mean() * 100) if len(miss) else float("nan")
        c_d = float(d[~d.missed].covered_ssa.mean() * 100)
        ok = bool(c_m >= 50 and c_m >= 2 * c_r and c_d >= 90) if split == "msd08" else None
        res[split] = dict(n=len(d), n_missed=int(d.missed.sum()), cov_missed=c_m, cov_missed_random=c_r,
                          cov_detected=c_d, GO=ok)
        lines += [f"## {split}", "", f"{len(d)} 个病灶，A0_s0 漏检 {int(d.missed.sum())} 个"
                  f"（部分点亮 {int((d.status == 'partial').sum())}，未点亮 {int((d.status == 'unlit').sum())}）。", "",
                  "覆盖率（%）：", "", "```", cov.round(1).to_string(), "```", "",
                  "漏检病灶按大小 / CNR 的 SSA 覆盖率（%）：", "", "```",
                  (miss.groupby(["size", "cnr_tier"]).covered_ssa.agg(["mean", "size"])
                   .assign(mean=lambda t: t["mean"] * 100).round(1).to_string()) if len(miss) else "无", "```", ""]
        if ok is not None:
            lines += [f"G0（漏检覆盖 ≥ 50%、≥ 2 × 随机锚点、已检出覆盖 ≥ 90%）：漏检 {c_m:.1f}%，随机 {c_r:.1f}%，"
                      f"已检出 {c_d:.1f}% → **{'GO' if ok else 'NO-GO'}**", ""]
    C.write_json(res, OUT / "summary.json")
    (OUT / "G0.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=sorted(SPLITS), required=True)
    r.add_argument("--shard", default=None)
    r.add_argument("--device", default="cuda")
    sub.add_parser("summarize")
    a = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
