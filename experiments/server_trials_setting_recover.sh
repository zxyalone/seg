#!/usr/bin/env bash
# Recover a TriALS-setting run whose final nnU-Net validation hung after losing an
# export worker (same failure as A14 seed 0: three validations exporting at once,
# 90 GB container limit). Waits for the main runner, reruns validation only
# (--val, checkpoint_final, no TTA, 3 export processes, nothing else running),
# evaluates it, predicts the test set (checkpoint_best and checkpoint_final),
# then redoes the descriptive evaluation.
#
#   tmux new -d -s ts-recover 'experiments/server_trials_setting_recover.sh T13b 2'
set -uo pipefail
GROUP=${1:?group}; SEED=${2:?seed}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=/root/autodl-tmp/nnUNet_data
ENV_BIN=/root/autodl-tmp/envs/mamba/bin
OUT=$REPO/results/trials_setting
FOLD=Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0
VAL=$DATA/nnUNet_results_v2/${GROUP}_s$SEED/$FOLD/validation
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
export VOCOMNI_NNUNET_CKPT=   # from scratch
export nnUNet_def_n_proc=3

while pgrep -f "server_trials_setting.sh" >/dev/null; do sleep 60; done
if [[ -f $VAL/summary.json && $(ls "$VAL"/*.nii.gz | wc -l) -eq 21 ]]; then
  log "${GROUP}_s$SEED validation already complete, continuing"
else
  log "${GROUP}_s$SEED validation rerun start (previous attempt lost an export worker)"
  (
    export nnUNet_raw=$DATA/nnUNet_raw nnUNet_preprocessed=$DATA/nnUNet_preprocessed nnUNet_results=$DATA/nnUNet_results_v2/${GROUP}_s$SEED
    export PYTHONPATH=$REPO/nnUNet CUDA_VISIBLE_DEVICES=$SEED
    for v in $(env | grep -o '^PHASER_[A-Z0-9_]*'); do unset "$v"; done
    set -a; source "$REPO/experiments/configs/$GROUP.env"; set +a
    export PHASER_SEED=$SEED
    "$ENV_BIN/nnUNetv2_train" Dataset102_LiTS 3d_fullres 0 -tr nnUNetTrainer_phaser --val
  ) > "$OUT/val_rerun_${GROUP}_s$SEED.log" 2>&1 || { log "${GROUP}_s$SEED validation rerun failed"; exit 1; }
fi
# run.sh skips training (checkpoint_final exists) and evaluates validation/
GPU=$SEED "$REPO/experiments/run.sh" $GROUP $SEED >> "$OUT/train_${GROUP}_s$SEED.log" 2>&1 && log "${GROUP}_s$SEED trained (validation recovered)" \
  && GPU=$SEED CHK=checkpoint_best.pth SUFFIX=_best "$REPO/experiments/predict_test.sh" $GROUP $SEED > "$OUT/predict_${GROUP}_s${SEED}_best.log" 2>&1 \
  && GPU=$SEED "$REPO/experiments/predict_test.sh" $GROUP $SEED > "$OUT/predict_${GROUP}_s${SEED}_final.log" 2>&1 \
  && log "${GROUP}_s$SEED test predicted (best, final)" || { log "${GROUP}_s$SEED recovery failed"; exit 1; }

for g in T13b T0; do for s in 0 1 2; do
  for e in eval_val eval_test_best eval_test; do
    [[ -f $DATA/nnUNet_results_v2/${g}_s$s/$FOLD/$e/summary.json ]] || { log "${g}_s$s $e missing"; exit 1; }
  done
done; done
cd "$REPO/experiments/analysis"
for split in val test_best test_final; do
  "$ENV_BIN/python" trials_setting_eval.py metrics --split $split --out "$OUT/${split}_tables" --jobs 3 > "$OUT/metrics_$split.log" 2>&1 \
    || { log "metrics $split failed (recovery)"; exit 1; }
done
"$ENV_BIN/python" trials_setting_eval.py stats --out "$OUT" > "$OUT/stats.log" 2>&1 || { log "stats failed (recovery)"; exit 1; }
log "TriALS-setting complete (after recovery of ${GROUP}_s$SEED)"
echo "TRIALS_SETTING_EXIT 0" >> "$OUT/runner.log"
