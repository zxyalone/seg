#!/bin/bash
cd /root/autodl-tmp/nnUNet_data || exit 1
echo "=== Copy2 出处日志 header (2026_9_5_15_36_00) ==="
grep -E "PFCR: |PC boundary loss:|Small-Tumor Head:|PCSF: |Total epochs:|Aux loss warmup|Initial LR|Oversample" \
  nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_5_15_36_00.txt | head -12
echo
echo "=== 9/5 日志 前 40 行 ==="
head -40 nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_5_15_36_00.txt
echo
echo "=== 9/5 日志 中的 checkpoint 保存事件 ==="
grep -nE "New best EMA pseudo Dice|New best tumor Dice" \
  nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_5_15_36_00.txt | tail -20
