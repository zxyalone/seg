#!/usr/bin/env python3
"""单元测试：missed_lesion_probability 的病例分析、阈值扫描和判据（合成数组，不需要模型或 GPU）。

    python experiments/analysis/test_missed_lesion_probability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import missed_lesion_probability as M  # noqa: E402

SPACING = (1.0, 0.78, 0.78)
SHAPE = (40, 60, 60)


def ball(c, r):
    z, y, x = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r


def synthetic():
    liver = np.zeros(SHAPE, bool)
    liver[2:38, 3:57, 3:57] = True
    gt = liver.astype(np.uint8)
    p_t = np.zeros(SHAPE, np.float32)
    lesions = [(ball((12, 15, 15), 5), 0.90),   # A：高概率，argmax 检出
               (ball((12, 45, 45), 5), 0.35),   # B：近决策边界，argmax 漏检
               (ball((30, 15, 45), 4), 0.02)]   # C：概率接近 0
    for m, p in lesions:
        gt[m] = 2
        p_t[m] = p
    small_fp = np.zeros(SHAPE, bool)
    small_fp[30:32, 45:47, 15:17] = True        # 8 体素，等效直径约 2.1 mm（被 4 mm 过滤）
    p_t[small_fp] = 0.80
    p_t[ball((30, 45, 30), 4)] = 0.25           # 较大的低概率假阳性
    p_l = np.where(liver, 1 - p_t, 0).astype(np.float32)
    p_bg = np.where(liver, 0, 1 - p_t).astype(np.float32)
    seg = np.argmax(np.stack([p_bg, p_l, p_t]), 0).astype(np.uint8)
    return p_t, p_l, seg, gt


def main():
    p_t, p_l, seg, gt = synthetic()
    r = M.analyse_case(p_t, p_l, seg, gt, SPACING, {1: 0.5, 2: 0.6, 3: 0.4})
    les = {l["lesion"]: l for l in r["lesions"]}
    assert r["n_gt"] == 3
    assert [round(les[i]["p_max"], 2) for i in (1, 2, 3)] == [0.90, 0.35, 0.02]
    assert [les[i]["det@argmax"] for i in (1, 2, 3)] == [True, False, False]
    assert [les[i]["det@0.3"] for i in (1, 2, 3)] == [True, True, False]
    assert [les[i]["det@0.05"] for i in (1, 2, 3)] == [True, True, False]
    assert les[2]["frac_liver"] == 1.0 and les[1]["frac_tumor"] == 1.0
    rl = r["rules"]
    assert (rl["argmax"]["fp"], rl["argmax_pp"]["fp"]) == (1, 0)
    assert (rl["0.3"]["fp"], rl["0.3_pp"]["fp"]) == (1, 0)
    assert (rl["0.2"]["fp"], rl["0.2_pp"]["fp"]) == (2, 1)
    print("✓ analyse_case：概率统计、各阈值检出、FP 计数、4 mm 过滤")

    # 三个"seed"使用同一结果；三个病灶都设为低 CNR
    rows_l, rows_r = [], []
    for s in range(3):
        base = dict(run=f"A0_s{s}", group="A0", seed=s, case="syn")
        rows_l += [{**base, **l} for l in r["lesions"]]
        rows_r += [{**base, "rule": k, "has_tumor": True, **v} for k, v in rl.items()]
    L, R = pd.DataFrame(rows_l), pd.DataFrame(rows_r)
    S = M.sweep(L, R, n_cases=1)
    a = S[(S.run == "A0_s0") & (S.rule == "0.3_pp")].iloc[0]
    assert abs(a.recall_low - 2 / 3) < 1e-9 and a.fp_per_scan == 0
    D = M.decide(L, S)["A0"]
    # 漏检的低 CNR 病灶为 B（0.35）和 C（0.02）：q = 0.5；0.3_pp 在 FP 预算内把低 CNR 召回从 1/3 提到 2/3
    assert abs(D["q_near_boundary"] - 0.5) < 1e-9, D
    assert abs(D["recall_low_gain_pp"] - 100 / 3) < 1e-6, D
    assert D["verdict"] == "decision-level", D
    print("✓ sweep / decide：召回、FP/scan、q、R 与判定")
    print("全部通过")


if __name__ == "__main__":
    main()
