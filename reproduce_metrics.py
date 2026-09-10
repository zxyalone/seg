#!/usr/bin/env python3
"""
PhASER-3D 精确复现脚本 — Dataset102_LiTS fold 0 测试集 24 例 Dice / IoU

复现目标（逐病例均值 ± 总体标准差，ddof=0，剔除 3 例无肿瘤 GT 的病例）：

    Liver Dice 0.951908 ± 0.027051    Liver IoU 0.909411 ± 0.045781
    Tumor Dice 0.702935 ± 0.152399    Tumor IoU 0.560872 ± 0.162564

模型：checkpoint_best-Copy2.pth（Dataset102_LiTS / 3d_fullres / fold 0，
      训练日志 2026-09-05 17:21:40 "New best EMA pseudo Dice: 0.8213"），
      推理必须关闭 TTA（--disable_tta）。

用法（必须使用 mamba 环境解释器）：

    # 仅从已保存的预测重新计算指标（快，秒级）
    /root/autodl-tmp/envs/mamba/bin/python reproduce_metrics.py

    # 先重新生成 27 例预测（无 TTA，约 40-60 分钟），再计算指标
    /root/autodl-tmp/envs/mamba/bin/python reproduce_metrics.py --predict

可选参数：
    --gpu N          指定 GPU（默认 0）
    --ckpt PATH      覆盖 checkpoint 路径
    --pred-dir DIR   覆盖预测输出目录
    --gt-dir DIR     覆盖测试集 GT 目录
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_DIR = Path(__file__).resolve().parent
NNUNET_DIR = PROJECT_DIR / "nnUNet"

DATA_ROOT = Path("/root/autodl-tmp/nnUNet_data")
RAW = DATA_ROOT / "nnUNet_raw"
PREPROCESSED = DATA_ROOT / "nnUNet_preprocessed"
RESULTS = DATA_ROOT / "nnUNet_results"

RUN_DIR = RESULTS / "Dataset102_LiTS" / "nnUNetTrainer_phaser__nnUNetPlans__3d_fullres" / "fold_0"
DEFAULT_CKPT = RUN_DIR / "_old_arch_backup" / "checkpoint_best-Copy2.pth"
DEFAULT_PRED_DIR = RUN_DIR / "test_predictions_100ep_no_tta_clean"
DEFAULT_GT_DIR = RAW / "Dataset101_LiTS" / "labelsTr"
IMAGES_TS = RAW / "Dataset102_LiTS" / "imagesTs"

PREDICT_BIN = Path("/root/autodl-tmp/envs/mamba/bin/nnUNetv2_predict")

EXCLUDE = {"LiTS_032", "LiTS_106", "LiTS_115"}  # 无肿瘤 GT，剔除
EXPECTED = {
    "liver_dice": (0.951908, 0.027051),
    "liver_iou": (0.909411, 0.045781),
    "tumor_dice": (0.702935, 0.152399),
    "tumor_iou": (0.560872, 0.162564),
}
TOL = 2e-4


def run_prediction(ckpt: Path, pred_dir: Path, gpu: str) -> None:
    """用 nnUNetv2_predict 重新生成 27 例预测（无 TTA）。"""
    if not ckpt.is_file():
        sys.exit(f"[ERROR] checkpoint 不存在: {ckpt}")
    if not PREDICT_BIN.is_file():
        sys.exit(f"[ERROR] 找不到 {PREDICT_BIN}")

    env = os.environ.copy()
    env["nnUNet_raw"] = str(RAW)
    env["nnUNet_preprocessed"] = str(PREPROCESSED)
    env["nnUNet_results"] = str(RESULTS)
    env["PYTHONPATH"] = str(NNUNET_DIR) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VOCOMNI_NNUNET_CKPT"] = ""  # 指标复现不依赖预训练权重
    env.update({
        "PHASER_PFCR": "1",
        "PHASER_PFCR_USE_PRIOR": "1",
        "PHASER_PC_SCALES": "3",
        "PHASER_PC_MAX_VOXELS": "1048576",
        "PHASER_PCSF": "0",
        "PHASER_LWFD": "0",
        "PHASER_ACG": "0",
        "PHASER_T_RAF": "none",
        "PHASER_LA_ACG": "0",
        "PHASER_SMALL_HEAD": "1",
    })

    # -chk 相对于 fold_0 目录
    ckpt_rel = os.path.relpath(ckpt, RUN_DIR)
    cmd = [
        str(PREDICT_BIN),
        "-i", str(IMAGES_TS),
        "-o", str(pred_dir),
        "-d", "Dataset102_LiTS",
        "-c", "3d_fullres",
        "-tr", "nnUNetTrainer_phaser",
        "-p", "nnUNetPlans",
        "-f", "0",
        "-chk", ckpt_rel,
        "--disable_tta",
        "-device", "cuda",
        "-npp", "1",
        "-nps", "1",
    ]
    print("[predict]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def dice_iou(pred: np.ndarray, gt: np.ndarray, label: int):
    p = pred == label
    g = gt == label
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    d = 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    i = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    return d, i


def compute_metrics(pred_dir: Path, gt_dir: Path, check_expected: bool = True) -> dict:
    preds = sorted(pred_dir.glob("*.nii.gz"))
    if len(preds) != 27:
        print(f"[WARN] 预测文件数 {len(preds)} != 27，继续计算现有文件")
    if not preds:
        sys.exit(f"[ERROR] 预测目录为空: {pred_dir}")

    rows = []
    for f in preds:
        case = f.name[: -len(".nii.gz")]
        if case in EXCLUDE:
            continue
        gt_path = gt_dir / f"{case}.nii.gz"
        if not gt_path.is_file():
            sys.exit(f"[ERROR] 缺少 GT: {gt_path}")
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        if pred.shape != gt.shape:
            sys.exit(f"[ERROR] {case} 预测/GT 形状不一致: {pred.shape} vs {gt.shape}")
        d1, i1 = dice_iou(pred, gt, 1)
        d2, i2 = dice_iou(pred, gt, 2)
        rows.append((case, d1, i1, d2, i2))

    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=np.float64)
    summary = {
        "liver_dice": (arr[:, 0].mean(), arr[:, 0].std(ddof=0)),
        "liver_iou": (arr[:, 1].mean(), arr[:, 1].std(ddof=0)),
        "tumor_dice": (arr[:, 2].mean(), arr[:, 2].std(ddof=0)),
        "tumor_iou": (arr[:, 3].mean(), arr[:, 3].std(ddof=0)),
    }

    print(f"\n{'case':10s} {'liver_dice':>11s} {'liver_iou':>10s} {'tumor_dice':>11s} {'tumor_iou':>10s}")
    for case, d1, i1, d2, i2 in rows:
        print(f"{case:10s} {d1:11.6f} {i1:10.6f} {d2:11.6f} {i2:10.6f}")

    print(f"\nN = {len(rows)} (per-case mean ± population std, ddof=0)")
    if not check_expected:
        for key in EXPECTED:
            mu, sd = summary[key]
            print(f"  {key:11s} {mu:.6f} ± {sd:.6f}")
        return {"rows": rows, "summary": summary, "ok": True}
    ok = True
    for key, (mu_exp, sd_exp) in EXPECTED.items():
        mu, sd = summary[key]
        passed = abs(mu - mu_exp) <= TOL and abs(sd - sd_exp) <= TOL
        ok &= passed
        print(f"  {key:11s} {mu:.6f} ± {sd:.6f}   expected {mu_exp:.6f} ± {sd_exp:.6f}   "
              f"[{'PASS' if passed else 'FAIL'}]")
    print(f"\n{'PASS: 指标已精确复现' if ok else 'FAIL: 指标与预期不一致'}")
    return {"rows": rows, "summary": summary, "ok": ok}


def main():
    ap = argparse.ArgumentParser(description="PhASER-3D 24-case metric reproduction")
    ap.add_argument("--predict", action="store_true", help="先重新生成预测（无 TTA）")
    ap.add_argument("--no-check", action="store_true",
                    help="只输出指标，不与 Copy2 预期值比对（用于评估新训练的模型）")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR))
    ap.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if args.predict:
        run_prediction(Path(args.ckpt), pred_dir, args.gpu)
    elif not pred_dir.is_dir():
        sys.exit(f"[ERROR] 预测目录不存在: {pred_dir}（加 --predict 重新生成）")

    result = compute_metrics(pred_dir, Path(args.gt_dir), check_expected=not args.no_check)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
