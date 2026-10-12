#!/usr/bin/env python3
"""
诊断：A0 漏掉的低 CNR 病灶，是"没看到"还是"看到了但没判成肿瘤"？
（docs/方法重设计_可检出性计划.md §4b，判据在运行前登记）

做法
----
对每个 run（默认 A0 的 3 个 seed）在 21 例 fold-0 验证集上推理，推理设置与训练结束时的验证相同
（滑窗步长 0.5、Gaussian 加权、无 TTA，checkpoint_final.pth），并把 softmax 概率按 nnU-Net 的
导出流程还原到原始网格。病灶定义、CNR 和匹配规则全部直接调用 experiments/eval/evaluate.py：
26 连通、IoU > 0.1 的一对一匈牙利匹配、CNR 冻结三分位 1.2785 / 1.9208。

每个参考病灶记录：
  p_max / p_q90 / p_mean   病灶内肿瘤概率的最大值、90% 分位数、均值
  pl_mean                  病灶内肝脏概率均值
  frac_tumor/liver/bg      病灶内 argmax 判为肿瘤、肝脏、背景的体素比例
  det@<t>, det_pp@<t>      在判定规则 t 下是否检出（pp = 去掉等效直径 < 4 mm 的预测连通域之后）
判定规则 t ∈ {argmax, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05}：argmax 即现有输出；数值 t 表示 p_tumor > t。
每个病例和规则还记录 FP 连通域数、肿瘤体素 TP/FP/FN（原始与 pp 两种）。

汇总时先核对：argmax 规则下的逐病灶检出和 FP 数应与 results/<run>/eval_val 一致。

  # 推理（GPU 0 留给用户训练；用 1、2 号卡分片）
  CUDA_VISIBLE_DEVICES=1 python experiments/analysis/missed_lesion_probability.py run --shard 0/2 &
  CUDA_VISIBLE_DEVICES=2 python experiments/analysis/missed_lesion_probability.py run --shard 1/2 &
  wait; python experiments/analysis/missed_lesion_probability.py summarize
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
import evaluate as E  # noqa: E402

THRESHOLDS = ("argmax", 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
MIN_DIAM_PP_MM = 4.0
CONNECTIVITY = 26
IOU_THR = 0.1
PMAX_BINS = (0.0, 0.05, 0.2, 0.3, 0.5, 1.0 + 1e-9)
PMAX_LABELS = ("<0.05", "0.05-0.2", "0.2-0.3", "0.3-0.5", ">=0.5")
# 事先登记的判据（计划 §4b）
DECISION_MIN_VOXELS = 50
NEAR_BOUNDARY_P = 0.3
FP_BUDGET = 0.5


def md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown 表格；服务器上没有 tabulate 时退回等宽文本。"""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def ttag(t) -> str:
    return "argmax" if t == "argmax" else f"{t:g}"


# ---------------------------------------------------------------------------
# 单病例分析（纯函数，单元测试覆盖）
# ---------------------------------------------------------------------------

def _drop_small(mask: np.ndarray, voxel_mm3: float, min_diam_mm: float) -> np.ndarray:
    lab, n = E.components(mask, CONNECTIVITY)
    if n == 0:
        return mask
    vox = np.bincount(lab.ravel(), minlength=n + 1)
    diam = (6 * vox * voxel_mm3 / np.pi) ** (1 / 3)
    keep = diam >= min_diam_mm
    keep[0] = False
    return keep[lab]


def analyse_case(p_tumor: np.ndarray, p_liver: np.ndarray, seg_argmax: np.ndarray, gt: np.ndarray,
                 spacing_zyx, cnr: dict) -> dict:
    """p_tumor / p_liver / seg_argmax / gt 均在原始网格 (z, y, x) 上。cnr: {gt 病灶编号: CNR}。"""
    voxel_mm3 = float(np.prod(spacing_zyx))
    gt_t = gt == 2
    gt_lab, n_gt = E.components(gt_t, CONNECTIVITY)
    lesions = []
    for i, sl in enumerate(ndi.find_objects(gt_lab), start=1):
        m = gt_lab[sl] == i
        pt = p_tumor[sl][m]
        arg = seg_argmax[sl][m]
        vox = int(m.sum())
        diam = float((6 * vox * voxel_mm3 / np.pi) ** (1 / 3))
        c = float(cnr.get(i, np.nan))
        lesions.append(dict(lesion=i, voxels=vox, diameter_mm=diam, size_stratum=E.size_stratum(diam),
                            cnr=c, cnr_stratum=E.cnr_stratum(c),
                            p_max=float(pt.max()), p_q90=float(np.quantile(pt, 0.9)), p_mean=float(pt.mean()),
                            pl_mean=float(p_liver[sl][m].mean()),
                            frac_tumor=float((arg == 2).mean()), frac_liver=float((arg == 1).mean()),
                            frac_bg=float((arg == 0).mean())))
    rules = {}
    for t in THRESHOLDS:
        tag = ttag(t)
        pred = (seg_argmax == 2) if t == "argmax" else (p_tumor > t)
        for suffix, mask in (("", pred), ("_pp", _drop_small(pred, voxel_mm3, MIN_DIAM_PP_MM))):
            pr_lab, n_pr = E.components(mask, CONNECTIVITY)
            match, _ = E.match_lesions(gt_lab, n_gt, pr_lab, n_pr, IOU_THR)
            matched = {p for p in match.values() if p is not None}
            for les in lesions:
                les[f"det{suffix}@{tag}"] = match[les["lesion"]] is not None
            tp = int((mask & gt_t).sum())
            rules[f"{tag}{suffix}"] = dict(fp=n_pr - len(matched), n_pred=n_pr, tp_vox=tp,
                                           fp_vox=int(mask.sum()) - tp, fn_vox=int(gt_t.sum()) - tp)
    return dict(n_gt=n_gt, has_tumor=bool(gt_t.any()), lesions=lesions, rules=rules)


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

def _export_and_analyse(logits, plans_manager, configuration_manager, label_manager, props_pp,
                        gt, spacing_zyx, cnr, meta, out_file):
    """CPU half of one model x case: identical calls to the serial path, so the
    result does not depend on --workers."""
    from nnunetv2.inference.export_prediction import \
        convert_predicted_logits_to_segmentation_with_correct_shape
    import common as C
    seg, probs = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, plans_manager, configuration_manager, label_manager, props_pp, return_probabilities=True)
    export_dtype = str(logits.dtype)
    del logits
    if seg.shape != gt.shape:
        raise RuntimeError(f"{meta['case']}: prediction {seg.shape} vs reference {gt.shape}")
    res = analyse_case(probs[2].astype(np.float32), probs[1].astype(np.float32), seg, gt, spacing_zyx, cnr)
    del probs
    res.update(**meta, export_logits_dtype=export_dtype, export_dtype_policy='native_validation')
    C.write_json(res, out_file)
    return str(out_file)


def cmd_run(args):
    import torch
    import common as C
    from nnunetv2.inference.export_prediction import \
        convert_predicted_logits_to_segmentation_with_correct_shape

    device = torch.device(args.device)
    image_dir, label_dir, out = Path(args.image_dir), Path(args.label_dir), Path(args.out)
    models = [C.load_model(g, s, device, args.results_root) for g, s in C.parse_runs(args.runs)]
    ref = models[0].predictor
    cases = C.shard(C.read_cases(args.cases_file, image_dir), args.shard)
    pool, pending = None, []
    if args.workers > 0:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"))
    for ci, case in enumerate(cases):
        todo = [m for m in models if not (out / "cases" / m.name / f"{case}.json").is_file()]
        if not todo:
            print(f"[{ci + 1}/{len(cases)}] {case}: done, skipped", flush=True)
            continue
        t0 = time.time()
        gt_img, gt = E.read(label_dir / f"{case}.nii.gz")
        spacing_zyx = tuple(float(s) for s in gt_img.GetSpacing()[::-1])
        gt_lab, n_gt = E.components(gt == 2, CONNECTIVITY)
        cnr = {}
        if n_gt:
            ip = E.find_image(image_dir, case)
            cnr = E.raw_lesion_cnr(gt_lab, n_gt, gt_img, E.sitk.ReadImage(str(ip)))
        raw, props, _ = C.read_raw(ref, case, image_dir, None)
        data, _, props_pp = C.preprocess(ref, raw, props)
        if pool is not None:
            # GPU inference stays here; resampling, softmax/argmax and the
            # threshold analysis (CPU-bound) run in worker processes. At most
            # `workers` exports are in flight to bound memory.
            for m in todo:
                while len(pending) >= args.workers:
                    _drain(pending, wait_one=True)
                logits = C.predict_logits(m, data, preserve_dtype=True)
                meta = dict(case=case, run=m.name, group=m.group, seed=m.seed)
                pending.append(pool.submit(_export_and_analyse, logits, m.predictor.plans_manager,
                                           m.predictor.configuration_manager, m.predictor.label_manager,
                                           props_pp, gt, spacing_zyx, cnr, meta,
                                           out / "cases" / m.name / f"{case}.json"))
                del logits
            print(f"[{ci + 1}/{len(cases)}] {case}: {n_gt} lesions, {len(todo)} models submitted, "
                  f"{time.time() - t0:.0f}s", flush=True)
            continue
        for m in todo:
            logits = C.predict_logits(m, data, preserve_dtype=True)
            seg, probs = convert_predicted_logits_to_segmentation_with_correct_shape(
                logits, m.predictor.plans_manager, m.predictor.configuration_manager,
                m.predictor.label_manager, props_pp, return_probabilities=True)
            export_dtype = str(logits.dtype)
            del logits
            if seg.shape != gt.shape:
                raise RuntimeError(f"{case}: prediction {seg.shape} vs reference {gt.shape}")
            res = analyse_case(probs[2].astype(np.float32), probs[1].astype(np.float32), seg, gt, spacing_zyx, cnr)
            del probs
            res.update(case=case, run=m.name, group=m.group, seed=m.seed,
                       export_logits_dtype=export_dtype, export_dtype_policy='native_validation')
            C.write_json(res, out / "cases" / m.name / f"{case}.json")
        print(f"[{ci + 1}/{len(cases)}] {case}: {n_gt} lesions, {len(todo)} models, {time.time() - t0:.0f}s",
              flush=True)
    if pool is not None:
        while pending:
            _drain(pending, wait_one=True)
        pool.shutdown()
        print("all exports written", flush=True)


def _drain(pending, wait_one=False):
    """Collect finished exports; re-raise the first worker error."""
    from concurrent.futures import FIRST_COMPLETED, wait
    if not pending:
        return
    done, _ = wait(pending, return_when=FIRST_COMPLETED) if wait_one else (
        [f for f in pending if f.done()], None)
    for f in done:
        pending.remove(f)
        print(f"  written {f.result()}", flush=True)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def load_results(out: Path):
    les, rules = [], []
    for f in sorted((out / "cases").glob("*/*.json")):
        r = json.loads(f.read_text())
        base = dict(run=r["run"], group=r["group"], seed=r["seed"], case=r["case"])
        for l in r["lesions"]:
            les.append({**base, **l})
        for k, v in r["rules"].items():
            rules.append({**base, "rule": k, "has_tumor": r["has_tumor"], **v})
    return pd.DataFrame(les), pd.DataFrame(rules)


def sweep(L: pd.DataFrame, R: pd.DataFrame, n_cases: int) -> pd.DataFrame:
    """每个 group × 规则：seed 平均的召回（总体与各 CNR 层）、FP/scan、肿瘤 Dice（含肿瘤病例均值）、F1。"""
    rows = []
    for (g, run), Lr in L.groupby(["group", "run"]):
        Rr = R[R.run == run]
        for t in THRESHOLDS:
            for suffix in ("", "_pp"):
                key = f"{ttag(t)}{suffix}"
                det = Lr[f"det{suffix}@{ttag(t)}"]
                rr = Rr[Rr.rule == key]
                tum = rr[rr.has_tumor]
                dice = (2 * tum.tp_vox / (2 * tum.tp_vox + tum.fp_vox + tum.fn_vox).clip(lower=1)).mean()
                tp, fp = int(det.sum()), int(rr.fp.sum())
                rec, prec = tp / max(len(det), 1), tp / max(tp + fp, 1)
                row = dict(group=g, run=run, rule=key, recall=rec, fp_per_scan=fp / n_cases,
                           tumor_dice=dice, f1=2 * rec * prec / max(rec + prec, 1e-9))
                for s in ("low", "mid", "high"):
                    d = det[Lr.cnr_stratum == s]
                    row[f"recall_{s}"] = d.mean() if len(d) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def decide(L: pd.DataFrame, S: pd.DataFrame) -> dict:
    """计划 §4b 事先登记的判据，按 group 给出。"""
    out = {}
    for g, Lg in L.groupby("group"):
        miss = Lg[(Lg.cnr_stratum == "low") & (Lg.voxels >= DECISION_MIN_VOXELS) & ~Lg["det@argmax"]]
        q = float((miss.p_max >= NEAR_BOUNDARY_P).mean()) if len(miss) else float("nan")
        Sg = S[S.group == g].groupby("rule")[["recall_low", "fp_per_scan"]].mean()
        base = Sg.loc["argmax_pp"]
        ok = Sg[Sg.index.str.endswith("_pp") & (Sg.fp_per_scan <= base.fp_per_scan + FP_BUDGET)]
        best_rule = ok.recall_low.idxmax()
        gain = float(100 * (ok.recall_low.max() - base.recall_low))
        if q >= 0.5 and gain >= 10:
            verdict = "decision-level"
        elif q <= 0.2:
            verdict = "representation-level"
        else:
            verdict = "mixed"
        out[g] = dict(n_missed_low_cnr_pairs=int(len(miss)), q_near_boundary=q,
                      recall_low_argmax_pp=float(100 * base.recall_low), fp_argmax_pp=float(base.fp_per_scan),
                      best_rule_within_budget=best_rule, recall_low_best=float(100 * ok.recall_low.max()),
                      fp_best=float(ok.loc[best_rule, "fp_per_scan"]), recall_low_gain_pp=gain, verdict=verdict)
    return out


def consistency(L: pd.DataFrame, R: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """argmax 规则应复现训练结束时的 eval_val（逐病灶检出、逐病例 FP）。"""
    rows = []
    for run, Lr in L.groupby("run"):
        ev = results_dir / run / "eval_val"
        if not (ev / "lesions.csv").is_file():
            continue
        ref = pd.read_csv(ev / "lesions.csv").set_index(["case", "lesion"])["detected@0.1"]
        cur = Lr.set_index(["case", "lesion"])["det@argmax"]
        j = pd.concat([ref.rename("ref"), cur.rename("cur")], axis=1, join="inner")
        refc = pd.read_csv(ev / "cases.csv").set_index("case")["les_fp@0.1"]
        curc = R[(R.run == run) & (R.rule == "argmax")].set_index("case").fp
        jc = pd.concat([refc.rename("ref"), curc.rename("cur")], axis=1, join="inner")
        rows.append(dict(run=run, lesions_compared=len(j), lesion_detection_agree=float((j.ref == j.cur).mean()),
                         cases_compared=len(jc), fp_count_abs_diff=int((jc.ref - jc.cur).abs().sum())))
    return pd.DataFrame(rows)


def cmd_summarize(args):
    out = Path(args.out)
    L, R = load_results(out)
    if L.empty:
        sys.exit(f"no results under {out / 'cases'}")
    n_cases = R.case.nunique()
    chk = consistency(L, R, Path(args.results_dir))
    S = sweep(L, R, n_cases)
    D = decide(L, S)
    L.to_csv(out / "lesions.csv", index=False)
    S.to_csv(out / "threshold_sweep_per_run.csv", index=False)
    chk.to_csv(out / "check_argmax_vs_eval_val.csv", index=False)

    lines = ["# 漏检病灶的肿瘤概率诊断", "",
             f"runs: {', '.join(sorted(L.run.unique()))}；病例 {n_cases}；参考病灶 {L[L.run == L.run.iloc[0]].shape[0]} 个/run。",
             "", "## 0 一致性核对（argmax 规则 vs eval_val）", "", md(chk, index=False) if not chk.empty else "（无 eval_val 可比）"]

    miss = L[~L["det@argmax"]].copy()
    miss["p_max_bin"] = pd.cut(miss.p_max, PMAX_BINS, labels=PMAX_LABELS, right=False)
    for title, sub in (("全部漏检（病灶-seed 对）", miss),
                       (f"漏检且 ≥ {DECISION_MIN_VOXELS} 体素（判据所用）", miss[miss.voxels >= DECISION_MIN_VOXELS])):
        t = pd.crosstab([sub.group, sub.cnr_stratum], sub.p_max_bin, margins=True)
        lines += ["", f"## 1 漏检病灶内最大肿瘤概率分布：{title}", "", md(t)]
    cols = ["p_max", "p_q90", "p_mean", "pl_mean", "frac_tumor", "frac_liver", "frac_bg"]
    m50 = miss[miss.voxels >= DECISION_MIN_VOXELS]
    lines += ["", "## 2 漏检病灶（≥ 50 体素）的概率与 argmax 组成（中位数）", "",
              md(m50.groupby(["group", "cnr_stratum"])[cols].median().round(3))]

    agg = S.groupby(["group", "rule"])[["recall", "recall_low", "recall_mid", "recall_high", "fp_per_scan",
                                         "tumor_dice", "f1"]].mean()
    agg[["recall", "recall_low", "recall_mid", "recall_high", "tumor_dice", "f1"]] *= 100
    lines += ["", "## 3 判定阈值扫描（seed 平均；召回、Dice、F1 为 %；pp = 去掉 < 4 mm 预测连通域）", "",
              md(agg.round(2))]

    lines += ["", "## 4 事先登记的判据（计划 §4b）", "",
              f"- q：漏检的低 CNR 病灶（≥ {DECISION_MIN_VOXELS} 体素，病灶-seed 对）中 p_max ≥ {NEAR_BOUNDARY_P} 的比例",
              f"- R：在 FP/scan（pp）不超过 argmax_pp + {FP_BUDGET} 的阈值中，低 CNR 召回（pp）相对 argmax_pp 的最大增益",
              "- 决策层：q ≥ 0.5 且 R ≥ 10 个百分点；表征层：q ≤ 0.2；其余为混合", ""]
    for g, d in D.items():
        lines.append(f"- **{g}**：q = {d['q_near_boundary']:.2f}（n = {d['n_missed_low_cnr_pairs']}）；"
                     f"低 CNR 召回 {d['recall_low_argmax_pp']:.1f}% → {d['recall_low_best']:.1f}%"
                     f"（{d['best_rule_within_budget']}，FP/scan {d['fp_argmax_pp']:.2f} → {d['fp_best']:.2f}），"
                     f"R = {d['recall_low_gain_pp']:+.1f} pp；判定：**{d['verdict']}**")
    text = "\n".join(lines) + "\n"
    (out / "diag_missed_probability.md").write_text(text)
    (out / "decision.json").write_text(json.dumps(D, indent=2, ensure_ascii=False, default=float))
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=None, help="default: $RESULTS_ROOT/_analysis/missed_probability")
    run = sub.choices["run"]
    run.add_argument("--runs", nargs="+", default=["A0:0,1,2"])
    run.add_argument("--results-root", default=None)
    run.add_argument("--image-dir", default=None)
    run.add_argument("--label-dir", default=None)
    run.add_argument("--cases-file", default=str(HERE.parent / "eval" / "val_cases.txt"))
    run.add_argument("--shard", default=None, help="i/n: this process handles cases i, i+n, ...")
    run.add_argument("--device", default=None)
    run.add_argument("--workers", type=int, default=0,
                     help="CPU processes for export + analysis (0 = serial, as before)")
    summ = sub.choices["summarize"]
    summ.add_argument("--results-dir", default=str(HERE.parents[1] / "results"),
                      help="where <run>/eval_val/{lesions,cases}.csv live (for the consistency check)")
    args = ap.parse_args()
    if args.cmd == "run":
        import torch
        import common as C
        args.out = args.out or str(C.RESULTS_ROOT / "_analysis" / "missed_probability")
        args.image_dir = args.image_dir or str(C.RAW_DATASET / "imagesTr")
        args.label_dir = args.label_dir or str(C.RAW_DATASET / "labelsTr")
        args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cmd_run(args)
    else:
        if args.out is None:
            import os
            data = Path(os.environ.get("DATA", "/root/autodl-tmp/nnUNet_data"))
            args.out = str(Path(os.environ.get("RESULTS_ROOT", data / "nnUNet_results_v2")) /
                           "_analysis" / "missed_probability")
        cmd_summarize(args)


if __name__ == "__main__":
    main()
