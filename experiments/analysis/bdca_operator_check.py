#!/usr/bin/env python3
"""
B-DCA 算子核对（docs/训练侧频域计划.md §3.2，判据事先登记）。

fold 0 训练集的每个病灶（标签 2 的 26 连通域，≥ 20 体素，与 DCA 相同），取以它为中心的 128³ 预处理图块，
用 5 个固定目标 t（每个病灶都用种子 0 抽取的同一组）分别做原 DCA 与 B-DCA（σ_b = 2 mm），测量中心病灶：
  实际 CNR：|μ_l − μ_r| / sd_r，环带与 DCA 相同（Chebyshev 2–6 体素、肝实质、其他病灶外扩 1 体素以外）；
  边界梯度：边界壳层（3×3×3 膨胀 − 腐蚀）上 |∇x| 的平均值，按体素间距。
判据（全部满足才能训练 B13）：
  1 等效直径 ≥ 10 mm 且 CNR0 > t 的病灶-目标对中，B-DCA 实际 CNR 在 t 的 ±20% 内的比例 ≥ 80%；
  2 等效直径 < 10 mm（且 CNR0 > t，即原 DCA 确实衰减）的病灶-目标对，边界梯度 B-DCA / 原 DCA 的中位数 ≥ 1.5；
  3 等效直径 < 10 mm 的病灶-目标对，B-DCA 实际 CNR 中位数 > 原 DCA 实际 CNR 中位数。

  python experiments/analysis/bdca_operator_check.py run --shard 0/8   # CPU，每片一个进程
  python experiments/analysis/bdca_operator_check.py summarize
输出 results/train_side/operator_check/。只读，不写 nnUNet_data。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "nnUNet"))
OUT = REPO / "results" / "train_side" / "operator_check"
SIGMA_MM = 2.0
N_TARGETS = 5
MIN_VOXELS = 20
SPLITS = "/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset102_LiTS/splits_final.json"
IMAGES = "/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/imagesTr"
LABELS = "/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset102_LiTS/labelsTr"


def measure(x, lab, m, spacing):
    """x [D, H, W] torch；lab 标签；m 中心病灶掩膜（bool）。返回 (CNR, 边界梯度)。"""
    import torch
    import torch.nn.functional as F
    mf = m.float()[None, None]
    others = ((lab == 2) & ~m).float()[None, None]
    dil = lambda t, r: F.max_pool3d(t, 2 * r + 1, 1, r)
    ring = (dil(mf, 6) - dil(mf, 1)) * (lab == 1).float()[None, None] * (1 - dil(others, 1))
    ring = ring[0, 0] > 0
    r = x[ring]
    cnr = float((x[m].mean() - r.mean()).abs() / (r.std(unbiased=False) + 1e-8)) if ring.sum() >= 10 else float("nan")
    shell = (dil(mf, 1) - (-dil(-mf, 1)))[0, 0] > 0
    g = torch.gradient(x, spacing=tuple(float(s) for s in spacing))
    mag = torch.sqrt(sum(gi ** 2 for gi in g))
    return cnr, float(mag[shell].mean())


def cmd_run(a):
    import torch
    from scipy import ndimage as ndi
    import common as C
    from freq_feature_probe import centered_patch
    from nnunetv2.training.nnUNetTrainer.phaser.lowcnr import lesion_contrast_augment
    out = OUT / "cases"
    out.mkdir(parents=True, exist_ok=True)
    m = C.load_model("A13b", 0, torch.device("cpu"))
    spacing = tuple(float(v) for v in m.predictor.configuration_manager.spacing)
    band = tuple(SIGMA_MM / s for s in spacing)
    vox_mm3 = float(np.prod(spacing))
    cases = C.shard(sorted(json.loads(Path(SPLITS).read_text())[0]["train"]), a.shard)
    for ci, case in enumerate(cases):
        f = out / f"{case}.json"
        if f.is_file():
            continue
        t0 = time.time()
        raw, props, seg_raw = C.read_raw(m.predictor, case, Path(IMAGES), Path(LABELS))
        data, seg, _ = C.preprocess(m.predictor, raw, props, seg_raw)
        s0 = seg[0].astype(np.int16)
        lab, n = ndi.label(s0 == 2, structure=ndi.generate_binary_structure(3, 3))
        rows = []
        for k, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is None or (lab[sl] == k).sum() < MIN_VOXELS:
                continue
            center = [int((s.start + s.stop) // 2) for s in sl]
            xp = torch.from_numpy(centered_patch(data[0], center, (128, 128, 128), float(data.min())).astype(np.float32))
            tp = torch.from_numpy(centered_patch(s0, center, (128, 128, 128), 0).astype(np.int64))
            mp = torch.from_numpy(centered_patch(lab, center, (128, 128, 128), 0) == k)
            nvox = int(mp.sum())
            cnr0, e0 = measure(xp, tp, mp, spacing)
            g = torch.Generator().manual_seed(0)
            from nnunetv2.training.nnUNetTrainer.phaser.lowcnr import sample_target_cnr
            for t in [sample_target_cnr(g) for _ in range(N_TARGETS)]:
                rec = dict(case=case, lesion=k, voxels=nvox, diameter_mm=(6 * nvox * vox_mm3 / np.pi) ** (1 / 3),
                           cnr0=cnr0, edge0=e0, target=t)
                for name, bs in (("plain", None), ("band", band)):
                    o, _ = lesion_contrast_augment(xp[None, None], tp[None, None], p=1.0, target_cnr=True,
                                                   fixed_target=t, band_sigma=bs)
                    rec[f"cnr_{name}"], rec[f"edge_{name}"] = measure(o[0, 0], tp, mp, spacing)
                rows.append(rec)
        C.write_json(dict(case=case, rows=rows), f)
        print(f"[{ci + 1}/{len(cases)}] {case}: {len(rows) // N_TARGETS} lesions, {time.time() - t0:.0f}s", flush=True)


def cmd_summarize(a):
    import pandas as pd
    import common as C
    rows = [r for f in sorted((OUT / "cases").glob("*.json")) for r in json.loads(f.read_text())["rows"]]
    d = pd.DataFrame(rows)
    d = d[np.isfinite(d.cnr0) & np.isfinite(d.cnr_plain) & np.isfinite(d.cnr_band)]
    d.to_csv(OUT / "pairs.csv", index=False)
    big = d[(d.diameter_mm >= 10) & (d.cnr0 > d.target)]
    small = d[d.diameter_mm < 10]
    small_att = small[small.cnr0 > small.target]
    c1 = float((np.abs(big.cnr_band - big.target) <= 0.2 * big.target).mean())
    c1_plain = float((np.abs(big.cnr_plain - big.target) <= 0.2 * big.target).mean())
    c2 = float(np.median(small_att.edge_band / small_att.edge_plain))
    c3b, c3p = float(np.median(small.cnr_band)), float(np.median(small.cnr_plain))
    res = dict(n_pairs=len(d), n_lesions=int(d.drop_duplicates(["case", "lesion"]).shape[0]),
               criterion1=dict(n=len(big), frac_within_20pct_band=c1, frac_within_20pct_plain=c1_plain, ok=c1 >= 0.8),
               criterion2=dict(n=len(small_att), median_edge_ratio=c2, ok=c2 >= 1.5),
               criterion3=dict(n=len(small), median_cnr_band=c3b, median_cnr_plain=c3p, ok=c3b > c3p))
    res["PASS"] = bool(res["criterion1"]["ok"] and res["criterion2"]["ok"] and res["criterion3"]["ok"])
    C.write_json(res, OUT / "summary.json")
    lines = ["# B-DCA 算子核对（训练侧频域计划 §3.2）", "",
             f"{res['n_lesions']} 个训练病灶 × {N_TARGETS} 个目标 = {res['n_pairs']} 对。", "",
             f"1. ≥ 10 mm、CNR0 > t（{len(big)} 对）：B-DCA 实际 CNR 在 t ± 20% 内 {c1:.1%}（原 DCA {c1_plain:.1%}），"
             f"门槛 80% → {res['criterion1']['ok']}",
             f"2. < 10 mm、CNR0 > t（{len(small_att)} 对）：边界梯度 B-DCA / 原 DCA 中位数 {c2:.2f}，门槛 1.5 → {res['criterion2']['ok']}",
             f"3. < 10 mm（{len(small)} 对）：实际 CNR 中位数 B-DCA {c3b:.2f}，原 DCA {c3p:.2f} → {res['criterion3']['ok']}",
             "", f"**总判定：{'通过' if res['PASS'] else '不通过'}**", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--shard", default=None)
    sub.add_parser("summarize")
    a = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
