#!/usr/bin/env bash
# TriALS-setting experiment (plan §11, 2026-10-14; descriptive): T13b and T0
# (from scratch, stock nnU-Net optimiser, 100 epochs), seeds 0-2, one GPU per
# seed; test predictions with checkpoint_best (as TriALS) and checkpoint_final,
# then the descriptive evaluation.
#
#   tmux new -d -s trials-setting 'experiments/server_trials_setting.sh'
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_BIN=${ENV_BIN:-/root/autodl-tmp/envs/mamba/bin}
OUT=$REPO/results/trials_setting
export nnUNet_def_n_proc=3   # three final validations at once stay below the 90 GB container limit
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
log "TriALS-setting start, HEAD $(git -C "$REPO" rev-parse --short HEAD)"

one() {  # group seed
  local g=$1 s=$2
  GPU=$s "$REPO/experiments/run.sh" $g $s > "$OUT/train_${g}_s$s.log" 2>&1 && log "${g}_s$s trained" \
    && GPU=$s CHK=checkpoint_best.pth SUFFIX=_best "$REPO/experiments/predict_test.sh" $g $s > "$OUT/predict_${g}_s${s}_best.log" 2>&1 \
    && GPU=$s "$REPO/experiments/predict_test.sh" $g $s > "$OUT/predict_${g}_s${s}_final.log" 2>&1 \
    && log "${g}_s$s test predicted (best, final)" || { log "${g}_s$s FAILED"; return 1; }
}
for s in 0 1 2; do
  ( one T13b $s; one T0 $s ) &
done
wait

cd "$REPO/experiments/analysis"
for split in val test_best test_final; do
  "$ENV_BIN/python" trials_setting_eval.py metrics --split $split --out "$OUT/${split}_tables" --jobs 3 > "$OUT/metrics_$split.log" 2>&1 \
    || { log "metrics $split failed"; exit 1; }
done
"$ENV_BIN/python" trials_setting_eval.py stats --out "$OUT" > "$OUT/stats.log" 2>&1 || { log "stats failed"; exit 1; }
log "TriALS-setting complete"
echo "TRIALS_SETTING_EXIT 0" >> "$OUT/runner.log"
