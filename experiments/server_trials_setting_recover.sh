#!/usr/bin/env bash
# Recovery for the TriALS-setting runs (plan §11, 2026-10-14): a final nnU-Net
# validation can die when three run at once (container memory.high throttling;
# T13b_s2 lost its export after 20 / 21 cases). Waits for the main runner to
# exit, then, one run at a time on GPU 0: reruns validation only where it is
# incomplete (--val, checkpoint_final, no TTA, 3 export processes), evaluates it,
# predicts the test set with checkpoint_best and checkpoint_final where missing,
# and finally redoes the descriptive metrics and statistics.
#
#   tmux new -d -s trials-recover 'experiments/server_trials_setting_recover.sh'
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=/root/autodl-tmp/nnUNet_data
ENV_BIN=/root/autodl-tmp/envs/mamba/bin
OUT=$REPO/results/trials_setting
FOLD=Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0
export VOCOMNI_NNUNET_CKPT=   # from scratch
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
fail() { log "recovery: $*"; exit 1; }

while pgrep -f "server_trials_setting.sh" >/dev/null; do sleep 60; done
log "recovery start, HEAD $(git -C "$REPO" rev-parse --short HEAD)"
for run in T13b_s0 T13b_s1 T13b_s2 T0_s0 T0_s1 T0_s2; do
  g=${run%_s*}; s=${run##*_s}; F=$DATA/nnUNet_results_v2/$run/$FOLD
  [[ -f $F/checkpoint_final.pth ]] || fail "$run has no checkpoint_final"
  if [[ ! -f $F/eval_val/summary.json ]]; then
    if [[ ! -f $F/validation/summary.json || $(ls "$F"/validation/*.nii.gz 2>/dev/null | wc -l) -ne 21 ]]; then
      log "$run validation rerun"
      (
        export nnUNet_raw=$DATA/nnUNet_raw nnUNet_preprocessed=$DATA/nnUNet_preprocessed nnUNet_results=$DATA/nnUNet_results_v2/$run
        export PYTHONPATH=$REPO/nnUNet CUDA_VISIBLE_DEVICES=0 nnUNet_def_n_proc=3
        for v in $(env | grep -o '^PHASER_[A-Z0-9_]*'); do unset "$v"; done
        set -a; source "$REPO/experiments/configs/$g.env"; set +a
        export PHASER_SEED=$s
        "$ENV_BIN/nnUNetv2_train" Dataset102_LiTS 3d_fullres 0 -tr nnUNetTrainer_phaser --val
      ) > "$OUT/val_rerun_$run.log" 2>&1 || fail "$run validation rerun failed"
    fi
    # run.sh skips training (checkpoint_final exists) and evaluates validation/
    GPU=0 "$REPO/experiments/run.sh" $g $s >> "$OUT/train_$run.log" 2>&1 || fail "$run validation evaluation failed"
    log "$run validation complete"
  fi
  if [[ ! -f $F/eval_test_best/summary.json ]]; then
    GPU=0 CHK=checkpoint_best.pth SUFFIX=_best "$REPO/experiments/predict_test.sh" $g $s > "$OUT/predict_${run}_best.log" 2>&1 \
      || fail "$run test prediction (best) failed"
  fi
  if [[ ! -f $F/eval_test/summary.json ]]; then
    GPU=0 "$REPO/experiments/predict_test.sh" $g $s > "$OUT/predict_${run}_final.log" 2>&1 || fail "$run test prediction (final) failed"
  fi
  log "$run test predicted (best, final)"
done

cd "$REPO/experiments/analysis"
for split in val test_best test_final; do
  "$ENV_BIN/python" trials_setting_eval.py metrics --split $split --out "$OUT/${split}_tables" --jobs 3 > "$OUT/metrics_$split.log" 2>&1 \
    || fail "metrics $split failed"
done
"$ENV_BIN/python" trials_setting_eval.py stats --out "$OUT" > "$OUT/stats.log" 2>&1 || fail "stats failed"
log "TriALS-setting complete (after recovery)"
echo "TRIALS_SETTING_EXIT 0" >> "$OUT/runner.log"
