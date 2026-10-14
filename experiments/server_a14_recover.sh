#!/usr/bin/env bash
# Recover an A14 seed whose final nnU-Net validation died (export worker lost,
# most likely the 90 GB container memory limit with three validations at once):
# wait for the other validations, rerun validation only (--val, checkpoint_final,
# no TTA, 3 export processes), evaluate it, predict the test set, then redo the
# descriptive evaluation once the main runner has stopped.
#
#   tmux new -d -s a14-recover 'experiments/server_a14_recover.sh 0'
set -uo pipefail
SEED=${1:?seed}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=/root/autodl-tmp/nnUNet_data
ENV_BIN=/root/autodl-tmp/envs/mamba/bin
OUT=$REPO/results/a14_pixel2cancer
FOLD=Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }

# wait until no other A14 validation is running (memory)
while pgrep -f "nnUNetv2_train Dataset102_LiTS 3d_fullres 0 -tr nnUNetTrainer_phaser" >/dev/null; do sleep 60; done
log "A14_s$SEED validation rerun start (previous attempt lost an export worker)"
(
  export nnUNet_raw=$DATA/nnUNet_raw nnUNet_preprocessed=$DATA/nnUNet_preprocessed nnUNet_results=$DATA/nnUNet_results_v2/A14_s$SEED
  export PYTHONPATH=$REPO/nnUNet CUDA_VISIBLE_DEVICES=$SEED nnUNet_def_n_proc=3
  export VOCOMNI_NNUNET_CKPT=/root/autodl-tmp/Downstream_2/VoComni_nnunet.pt
  for v in $(env | grep -o '^PHASER_[A-Z0-9_]*'); do unset "$v"; done
  set -a; source "$REPO/experiments/configs/A14.env"; set +a
  export PHASER_SEED=$SEED
  "$ENV_BIN/nnUNetv2_train" Dataset102_LiTS 3d_fullres 0 -tr nnUNetTrainer_phaser --val
) > "$OUT/val_rerun_A14_s$SEED.log" 2>&1 || { log "A14_s$SEED validation rerun failed"; exit 1; }
# run.sh skips training (checkpoint_final exists) and evaluates validation/
GPU=$SEED "$REPO/experiments/run.sh" A14 $SEED >> "$OUT/train_A14_s$SEED.log" 2>&1 && log "A14_s$SEED trained" \
  && GPU=$SEED "$REPO/experiments/predict_test.sh" A14 $SEED > "$OUT/predict_A14_s$SEED.log" 2>&1 \
  && log "A14_s$SEED test predicted" || { log "A14_s$SEED recovery failed"; exit 1; }

while pgrep -f "server_a14.sh" >/dev/null; do sleep 60; done
for s in 0 1 2; do
  [[ -f $DATA/nnUNet_results_v2/A14_s$s/$FOLD/eval_test/summary.json ]] || { log "A14_s$s test evaluation missing"; exit 1; }
done
cd "$REPO/experiments/analysis"
for split in val test; do
  "$ENV_BIN/python" a14_eval.py metrics --split $split --out "$OUT/${split}_tables" --jobs 3 > "$OUT/metrics_$split.log" 2>&1 \
    || { log "metrics $split failed (recovery)"; exit 1; }
done
"$ENV_BIN/python" a14_eval.py stats --out "$OUT" > "$OUT/stats.log" 2>&1 || { log "stats failed (recovery)"; exit 1; }
log "A14 complete (after recovery of seed $SEED)"
echo "A14_EXIT 0" >> "$OUT/runner.log"
