#!/usr/bin/env bash
# Run after stage 4 to summarize the original training logs for seed-0 runs.
# R and F can be overridden to check an archived run or test this command.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=${R:-$(dirname "$REPO")/nnUNet_data/nnUNet_results_v2}
F=${F:-Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0}

for g in A6 A0 A1; do
  LOG=$(ls "$R/${g}_s0/$F"/training_log_*.txt | tail -1)
  if [[ -z $LOG || ! -f $LOG ]]; then
    echo "Missing training log for $g" >&2
    exit 1
  fi
  echo "== $g  梯度跳过: $(grep -c 'non-finite gradient' "$LOG")   loss 非有限: $(grep -c 'non-finite loss' "$LOG")"
  grep -o 'non-finite gradient at epoch [0-9]*' "$LOG" | awk '{print $NF}' | sort -n | uniq -c | head -20
done
