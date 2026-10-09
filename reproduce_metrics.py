#!/usr/bin/env python3
"""
PhASER-3D 精确复现脚本 — Dataset102_LiTS fold 0 测试集 Dice / IoU

默认口径（与 README 第 1/6 节主表一致）：
  - 队列：27 例测试病例中 24 例含肿瘤 GT 的病例（剔除 LiTS_032/106/115）
  - Liver  = 肝实质 label == 1（不含肿瘤）
  - Tumor  = label == 2
  - 逐病例计算全体积 Dice/IoU，再取宏平均 ± 总体标准差（ddof=0）
  - 无 TTA

预期结果（Copy2 checkpoint_best-Copy2.pth）：
  Liver(label1)  Dice 0.951908 ± 0.027051    IoU 0.909411 ± 0.045781
  Tumor(label2)  Dice 0.702935 ± 0.152399    IoU 0.560872 ± 0.162564

脚本同时输出补充口径：
  - Whole liver = 整肝 label > 0（肝实质 ∪ 肿瘤）
  - --all-cases：全部 27 例（3 例无肿瘤 GT 的病例肿瘤 Dice 记为 0；其预测有假阳性，分母非零）

用法（必须使用 mamba 环境解释器）：

    # 默认：从 Copy2 已保存预测复算 24 例指标，并与预期值逐项比对
    /root/autodl-tmp/envs/mamba/bin/python reproduce_metrics.py

    # 评估其它预测目录（只输出数值，不比对预期）
    /root/autodl-tmp/envs/mamba/bin/python reproduce_metrics.py \
      --pred-dir <dir> --no-check

    # 全部 27 例完整测试集口径
    /root/autodl-tmp/envs/mamba/bin/python reproduce_metrics.py --all-cases --no-check

可选参数：
    --predict        先重新生成 27 例预测（无 TTA，约 40-60 分钟）
    --all-cases      使用全部 27 例（默认仅 24 例含肿瘤队列）
    --no-check       不与 Copy2 预期值比对
    --gpu N          指定 GPU（默认 0）
    --ckpt PATH      覆盖 checkpoint 路径
    --pred-dir DIR   覆盖预测目录
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

EXCLUDE = {"LiTS_032", "LiTS_106", "LiTS_115"}  # GT 无肿瘤体素
EXPECTED = {  # Copy2 默认口径（24 例，label1 / label2）
    "liver1_dice": (0.951908, 0.027051),
    "liver1_iou": (0.909411, 0.045781),
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
        # Copy2 was trained with pre-V2 code: double-sigmoid prior and PC noise
        # scale from scale 0. V2 changed both, so restore them for this checkpoint.
        "PHASER_LEGACY": "1",
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


def _dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray):
    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())
    den_d = 2 * tp + fp + fn
    den_i = tp + fp + fn
    d = 2.0 * tp / den_d if den_d else float("nan")
    i = tp / den_i if den_i else float("nan")
    return d, i


def liver_label1(pred: np.ndarray, gt: np.ndarray):
    """肝实质 label == 1（不含肿瘤）。"""
    return _dice_iou(pred == 1, gt == 1)


def whole_liver(pred: np.ndarray, gt: np.ndarray):
    """整肝 label > 0（肝实质 ∪ 肿瘤）。"""
    return _dice_iou(pred > 0, gt > 0)


def tumor_label2(pred: np.ndarray, gt: np.ndarray):
    """肿瘤 label == 2；GT 无肿瘤时：预测也为空记 1，否则记 0。"""
    if int((gt == 2).sum()) == 0:
        return (1.0, 1.0) if int((pred == 2).sum()) == 0 else (0.0, 0.0)
    return _dice_iou(pred == 2, gt == 2)


def compute_metrics(pred_dir: Path, gt_dir: Path, check_expected: bool = True,
                    all_cases: bool = False) -> dict:
    preds = sorted(pred_dir.glob("*.nii.gz"))
    if len(preds) != 27:
        print(f"[WARN] 预测文件数 {len(preds)} != 27，继续计算现有文件")
    if not preds:
        sys.exit(f"[ERROR] 预测目录为空: {pred_dir}")

    rows = []
    for f in preds:
        case = f.name[: -len(".nii.gz")]
        if not all_cases and case in EXCLUDE:
            continue
        gt_path = gt_dir / f"{case}.nii.gz"
        if not gt_path.is_file():
            sys.exit(f"[ERROR] 缺少 GT: {gt_path}")
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        if pred.shape != gt.shape:
            sys.exit(f"[ERROR] {case} 预测/GT 形状不一致: {pred.shape} vs {gt.shape}")
        d1, i1 = liver_label1(pred, gt)
        dw, iw = whole_liver(pred, gt)
        d2, i2 = tumor_label2(pred, gt)
        rows.append((case, d1, i1, dw, iw, d2, i2))

    arr = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows], dtype=np.float64)
    summary = {
        "liver1_dice": (arr[:, 0].mean(), arr[:, 0].std(ddof=0)),
        "liver1_iou": (arr[:, 1].mean(), arr[:, 1].std(ddof=0)),
        "whole_liver_dice": (arr[:, 2].mean(), arr[:, 2].std(ddof=0)),
        "whole_liver_iou": (arr[:, 3].mean(), arr[:, 3].std(ddof=0)),
        "tumor_dice": (arr[:, 4].mean(), arr[:, 4].std(ddof=0)),
        "tumor_iou": (arr[:, 5].mean(), arr[:, 5].std(ddof=0)),
    }

    cohort = "all 27 cases" if all_cases else "24 tumor-bearing cases"
    print(f"\n{'case':10s} {'liver1_D':>9s} {'liver1_IoU':>10s} {'whole_D':>9s} "
          f"{'whole_IoU':>9s} {'tumor_D':>9s} {'tumor_IoU':>9s}")
    for case, d1, i1, dw, iw, d2, i2 in rows:
        print(f"{case:10s} {d1:9.6f} {i1:10.6f} {dw:9.6f} {iw:9.6f} {d2:9.6f} {i2:9.6f}")

    print(f"\nN = {len(rows)} ({cohort}; per-case mean ± population std, ddof=0)")
    print(f"  Liver parenchyma (label 1)   Dice {summary['liver1_dice'][0]:.6f} ± {summary['liver1_dice'][1]:.6f}   "
          f"IoU {summary['liver1_iou'][0]:.6f} ± {summary['liver1_iou'][1]:.6f}")
    print(f"  Whole liver (label > 0)      Dice {summary['whole_liver_dice'][0]:.6f} ± {summary['whole_liver_dice'][1]:.6f}   "
          f"IoU {summary['whole_liver_iou'][0]:.6f} ± {summary['whole_liver_iou'][1]:.6f}")
    print(f"  Tumor (label 2)              Dice {summary['tumor_dice'][0]:.6f} ± {summary['tumor_dice'][1]:.6f}   "
          f"IoU {summary['tumor_iou'][0]:.6f} ± {summary['tumor_iou'][1]:.6f}")

    if not check_expected or all_cases:
        if all_cases and check_expected:
            print("\n[跳过 Copy2 预期比对：--all-cases 为完整测试集口径]")
        return {"rows": rows, "summary": summary, "ok": True}

    ok = True
    print()
    for key, (mu_exp, sd_exp) in EXPECTED.items():
        mu, sd = summary[key]
        passed = abs(mu - mu_exp) <= TOL and abs(sd - sd_exp) <= TOL
        ok &= passed
        print(f"  {key:11s} {mu:.6f} ± {sd:.6f}   expected {mu_exp:.6f} ± {sd_exp:.6f}   "
              f"[{'PASS' if passed else 'FAIL'}]")
    print(f"\n{'PASS: 指标已精确复现' if ok else 'FAIL: 指标与预期不一致'}")
    return {"rows": rows, "summary": summary, "ok": ok}


def main():
    ap = argparse.ArgumentParser(description="PhASER-3D metric reproduction")
    ap.add_argument("--predict", action="store_true", help="先重新生成预测（无 TTA）")
    ap.add_argument("--all-cases", action="store_true",
                    help="使用全部 27 例（默认仅 24 例含肿瘤队列；该口径下自动跳过预期比对）")
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

    result = compute_metrics(pred_dir, Path(args.gt_dir),
                             check_expected=not args.no_check,
                             all_cases=args.all_cases)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
