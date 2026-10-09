#!/usr/bin/env bash
# Run the training queue of manuscript/实验清单.md back to back on one GPU.
#
#   experiments/queue.sh stage1          # E2-E4: A6, A0, A1 (seed 0) -> decision gate
#   experiments/queue.sh stage2          # after the gate: E5-E9
#   experiments/queue.sh stage2-b        # branch B: matched A0/A1/A6 seeds first
#   experiments/queue.sh smoke           # E0: every group for 2 epochs
#   GPU=1 experiments/queue.sh custom A6:1 A0:1
#
# Each entry is GROUP:SEED. Finished runs (checkpoint_final.pth present) are
# skipped, so the queue can simply be restarted after an interruption.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case ${1:-} in
  stage1) JOBS=(A6:0 A0:0 A1:0) ;;
  stage2) JOBS=(A7:0 A6:1 A0:1 A6:2 A0:2 A5:0 A8:0) ;;
  stage2-b) JOBS=(A6:1 A0:1 A1:1 A6:2 A0:2 A1:2 A7:0 A8:0) ;;
  smoke)  JOBS=(A0:0 A1:0 A5:0 A6:0 A7:0 A8:0); export EPOCHS=2
          export RESULTS_ROOT=${RESULTS_ROOT:-/root/autodl-tmp/nnUNet_data/nnUNet_results_smoke} ;;
  custom) shift; JOBS=("$@") ;;
  *) echo "usage: $0 stage1|stage2|stage2-b|smoke|custom GROUP:SEED..." >&2; exit 1 ;;
esac

for job in "${JOBS[@]}"; do
  g=${job%%:*}; s=${job##*:}
  if ! "$REPO/experiments/run.sh" "$g" "$s"; then
    echo "[$(date '+%F %T')] $g seed $s FAILED, continuing with the next job" >&2
  fi
done

# E9 (residual-encoder baseline) uses stock nnU-Net with its own plans. This
# fork predates the ResEnc M/L presets and only has ResEncUNetPlanner, whose
# GPU-memory target (8 GB) matches ResEnc-M:
#   nnUNetv2_plan_experiment -d 102 -pl ResEncUNetPlanner
#   nnUNetv2_preprocess -d 102 -plans_name nnUNetResEncUNetPlans -c 3d_fullres
#   nnUNet_results=$DATA/nnUNet_results_v2/ResEnc_s0 \
#     nnUNetv2_train 102 3d_fullres 0 -p nnUNetResEncUNetPlans -tr nnUNetTrainer_200epochs
# Report its patch size in Table 2. nnUNetTrainer_200epochs matches the
# 200-epoch budget of the PhASER groups (stock nnUNetTrainer: 1000 epochs).
