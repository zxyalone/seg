#!/usr/bin/env bash
# Recovery for TriALS-setting runs whose final nnU-Net validation hangs after
# losing an export worker (same failure as A14 seed 0 and T13b_s2: several
# validations exporting at once, 90 GB container limit).
#
# 1. While the main runner is alive, a watchdog stops a final validation that
#    has written nothing for 25 min (checkpoint_final present, no summary.json);
#    the runner then logs that run as FAILED and moves on.
# 2. After the runner has exited, every run with an incomplete validation is
#    re-validated one at a time (--val, checkpoint_final, no TTA, 3 export
#    processes), evaluated, and predicted on the test set (checkpoint_best and
#    checkpoint_final) if those predictions are missing.
# 3. The descriptive evaluation of all six runs.
#
#   tmux new -d -s ts-recover 'experiments/server_trials_setting_recover.sh'
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=/root/autodl-tmp/nnUNet_data
ENV_BIN=/root/autodl-tmp/envs/mamba/bin
OUT=$REPO/results/trials_setting
FOLD=Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0
RUNS="T13b_s0 T13b_s1 T13b_s2 T0_s0 T0_s1 T0_s2"
STALL=1500   # seconds without a new validation file
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
fold() { echo "$DATA/nnUNet_results_v2/$1/$FOLD"; }
val_done() { [[ -f $(fold $1)/validation/summary.json && $(ls "$(fold $1)"/validation/*.nii.gz 2>/dev/null | wc -l) -eq 21 ]]; }
export VOCOMNI_NNUNET_CKPT=   # from scratch
export nnUNet_def_n_proc=3

# 1. watchdog
while pgrep -f "server_trials_setting.sh" >/dev/null; do
  for pid in $(pgrep -f "nnUNetv2_train Dataset102_LiTS 3d_fullres 0 -tr nnUNetTrainer_phaser"); do
    res=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^nnUNet_results=//p')
    run=$(basename "$res" 2>/dev/null); [[ " $RUNS " == *" $run "* ]] || continue
    f=$(fold $run); [[ -f $f/checkpoint_final.pth && ! -f $f/validation/summary.json ]] || continue
    last=$(stat -c %Y "$f/checkpoint_final.pth")
    newest=$(ls -t "$f"/validation/*.nii.gz 2>/dev/null | head -1)
    [[ -n $newest ]] && last=$(stat -c %Y "$newest")
    if (( $(date +%s) - last > STALL )); then
      log "$run validation stalled ($(ls "$f"/validation/*.nii.gz 2>/dev/null | wc -l) / 21 written); stopping it for recovery"
      kill $(pgrep -P $pid) $pid 2>/dev/null
    fi
  done
  sleep 60
done

# 2. recover incomplete runs, one at a time
for run in $RUNS; do
  g=${run%_s*}; s=${run##*_s}; f=$(fold $run)
  [[ -f $f/checkpoint_final.pth ]] || { log "$run has no checkpoint_final"; exit 1; }
  if ! val_done $run; then
    log "$run validation rerun start"
    (
      export nnUNet_raw=$DATA/nnUNet_raw nnUNet_preprocessed=$DATA/nnUNet_preprocessed nnUNet_results=$DATA/nnUNet_results_v2/$run
      export PYTHONPATH=$REPO/nnUNet CUDA_VISIBLE_DEVICES=$s
      for v in $(env | grep -o '^PHASER_[A-Z0-9_]*'); do unset "$v"; done
      set -a; source "$REPO/experiments/configs/$g.env"; set +a
      export PHASER_SEED=$s
      "$ENV_BIN/nnUNetv2_train" Dataset102_LiTS 3d_fullres 0 -tr nnUNetTrainer_phaser --val
    ) > "$OUT/val_rerun_$run.log" 2>&1 || { log "$run validation rerun failed"; exit 1; }
  fi
  if [[ ! -f $f/eval_val/summary.json ]]; then
    GPU=$s "$REPO/experiments/run.sh" $g $s >> "$OUT/train_$run.log" 2>&1 && log "$run trained (validation recovered)" \
      || { log "$run recovery failed (validation evaluation)"; exit 1; }
  fi
  if [[ ! -f $f/eval_test_best/summary.json ]]; then
    GPU=$s CHK=checkpoint_best.pth SUFFIX=_best "$REPO/experiments/predict_test.sh" $g $s > "$OUT/predict_${run}_best.log" 2>&1 \
      || { log "$run recovery failed (test best)"; exit 1; }
  fi
  if [[ ! -f $f/eval_test/summary.json ]]; then
    GPU=$s "$REPO/experiments/predict_test.sh" $g $s > "$OUT/predict_${run}_final.log" 2>&1 \
      && log "$run test predicted (best, final)" || { log "$run recovery failed (test final)"; exit 1; }
  fi
done

# 3. descriptive evaluation
cd "$REPO/experiments/analysis"
for split in val test_best test_final; do
  "$ENV_BIN/python" trials_setting_eval.py metrics --split $split --out "$OUT/${split}_tables" --jobs 3 > "$OUT/metrics_$split.log" 2>&1 \
    || { log "metrics $split failed (recovery)"; exit 1; }
done
"$ENV_BIN/python" trials_setting_eval.py stats --out "$OUT" > "$OUT/stats.log" 2>&1 || { log "stats failed (recovery)"; exit 1; }
log "TriALS-setting complete (after recovery)"
echo "TRIALS_SETTING_EXIT 0" >> "$OUT/runner.log"
