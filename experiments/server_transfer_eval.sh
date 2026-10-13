#!/usr/bin/env bash
# SegResNet architecture transfer (plan §11, 2026-10-13): one-time test-set
# prediction of S0_s{0,1} (fp16) and S13b_bf16_s{0,1} (bf16, from the config),
# then fixed-4-mm metrics and S13b_bf16 - S0 bootstrap CIs on val and test.
# Descriptive only. S0 predicts as soon as this starts; S13b_bf16 waits for
# RUN_EXIT 0 of its training.
#
#   tmux new -d -s transfer-eval 'experiments/server_transfer_eval.sh'
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=/root/autodl-tmp/nnUNet_data
PY=/root/autodl-tmp/envs/mamba/bin/python
LOGS=$DATA/nnUNet_results_v2/_segresnet_transfer
OUT=$REPO/results/segresnet_transfer
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }

predict() {  # group seed gpu
  if [[ -f $DATA/nnUNet_results_v2/$1_s$2/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/eval_test/summary.json ]]; then
    log "$1_s$2 test prediction exists, skipping"; return; fi
  log "$1_s$2 test prediction start (GPU $3)"
  GPU=$3 VOCOMNI_NNUNET_CKPT= "$REPO/experiments/predict_test.sh" "$1" "$2" > "$OUT/predict_$1_s$2.log" 2>&1
  log "$1_s$2 test prediction done"
}

wait_training() {  # run
  until grep -q 'RUN_EXIT' "$LOGS/$1.log" 2>/dev/null; do sleep 60; done
  grep -q 'RUN_EXIT 0' "$LOGS/$1.log" || { log "$1 training failed"; exit 1; }
}

log "transfer evaluation start"
( predict S0 0 2; predict S0 1 2 ) &
s0=$!
( wait_training S13b_bf16_s0; predict S13b_bf16 0 0 ) &
m0=$!
( wait_training S13b_bf16_s1; predict S13b_bf16 1 1 ) &
m1=$!
wait $s0; wait $m0; wait $m1   # each wait fails the script if that branch failed

cd "$REPO/experiments/analysis"
for split in val test; do
  log "metrics $split"
  "$PY" segresnet_transfer_eval.py metrics --split "$split" --out "$OUT/$split/tables" --jobs 4 > "$OUT/metrics_$split.log" 2>&1
  "$PY" segresnet_transfer_eval.py stats --tables "$OUT/$split/tables" --out "$OUT/$split/stats" > "$OUT/stats_$split.log" 2>&1
done
log "transfer evaluation complete"
