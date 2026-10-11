#!/usr/bin/env bash
# Train one ablation group with one seed, then evaluate it on the 21 fold-0
# validation cases. The test set is never touched here (see predict_test.sh).
#
#   experiments/run.sh A6 0            # group A6, seed 0, GPU 0
#   GPU=1 experiments/run.sh A0 1
#   EPOCHS=2 experiments/run.sh A6 0   # smoke test (E0): 2 epochs
#
# Outputs: $RESULTS_ROOT/<group>_s<seed>/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/
#   checkpoint_final.pth, run_manifest.json, validation/ (no TTA), eval_val/
set -euo pipefail

GROUP=${1:?group, e.g. A6}
SEED=${2:?seed, e.g. 0}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/root/autodl-tmp/nnUNet_data}
ENV_BIN=${ENV_BIN:-/root/autodl-tmp/envs/mamba/bin}
RESULTS_ROOT=${RESULTS_ROOT:-$DATA/nnUNet_results_v2}
GPU=${GPU:-0}
CFG="$REPO/experiments/configs/$GROUP.env"
[[ -f $CFG ]] || { echo "no config $CFG" >&2; exit 1; }

RUN="${GROUP}_s${SEED}"
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RESULTS_ROOT/$RUN"
export PYTHONPATH="$REPO/nnUNet${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$GPU
export VOCOMNI_NNUNET_CKPT=${VOCOMNI_NNUNET_CKPT-$REPO/VoComni_nnunet.pt}
if [[ -n $VOCOMNI_NNUNET_CKPT && ! -f $VOCOMNI_NNUNET_CKPT ]]; then
  echo "pretrained checkpoint not found: $VOCOMNI_NNUNET_CKPT" >&2
  exit 1
fi

# Clean PhASER environment, then the group config, then the seed.
for v in $(env | grep -o '^PHASER_[A-Z0-9_]*'); do unset "$v"; done
set -a; source "$CFG"; set +a
export PHASER_SEED=$SEED
[[ -n ${EPOCHS:-} ]] && export PHASER_NUM_EPOCHS=$EPOCHS

FOLD_DIR="$nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0"
mkdir -p "$nnUNet_results"
LOG="$nnUNet_results/run.log"
echo "[$(date '+%F %T')] $RUN start (GPU $GPU, commit $(git -C "$REPO" rev-parse --short HEAD))" | tee -a "$LOG"

if [[ -f $FOLD_DIR/checkpoint_final.pth ]]; then
  echo "checkpoint_final exists, skipping training" | tee -a "$LOG"
else
  CONT=()
  [[ -f $FOLD_DIR/checkpoint_latest.pth ]] && CONT=(--c)
  "$ENV_BIN/nnUNetv2_train" Dataset102_LiTS 3d_fullres 0 -tr nnUNetTrainer_phaser ${CONT[@]+"${CONT[@]}"} 2>&1 | tee -a "$LOG"
fi

[[ -f $REPO/experiments/eval/val_cases.txt ]] || \
  "$ENV_BIN/python" "$REPO/experiments/eval/make_case_lists.py" \
    --preprocessed "$nnUNet_preprocessed/Dataset102_LiTS"

# nnU-Net's final validation (no TTA, see perform_actual_validation) wrote validation/.
"$ENV_BIN/python" "$REPO/experiments/eval/evaluate.py" \
  --pred-dir "$FOLD_DIR/validation" \
  --gt-dir "$nnUNet_raw/Dataset102_LiTS/labelsTr" \
  --image-dir "$nnUNet_raw/Dataset102_LiTS/imagesTr" \
  --cases-file "$REPO/experiments/eval/val_cases.txt" \
  --workers "${EVAL_WORKERS:-4}" \
  --out-dir "$FOLD_DIR/eval_val" 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] $RUN done" | tee -a "$LOG"
