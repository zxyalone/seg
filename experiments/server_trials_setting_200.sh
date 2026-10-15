#!/usr/bin/env bash
# 200-epoch from-scratch runs (plan §11, 2026-10-15; descriptive): T13bL and T0L
# (= T13b / T0 with 200 epochs), seeds 0-2, one GPU per seed. The GPUs start
# 15 min apart so that the final validations do not export at the same time
# (the cause of the T13b_s2 and A14 seed 0 hangs). Test predictions with
# checkpoint_best and checkpoint_final. The watchdog, recovery of incomplete
# runs and the descriptive evaluation run in server_trials_setting_recover.sh:
#
#   tmux new -d -s ts200 'experiments/server_trials_setting_200.sh'
#   tmux new -d -s ts200-recover 'TS_GROUPS="T13bL T0L" TS_OUT=results/trials_setting_200 \
#       TS_RUNNER="[s]erver_trials_setting_200.sh" TS_EVAL_GROUPS=T0L,T13bL experiments/server_trials_setting_recover.sh'
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$REPO/results/trials_setting_200
export VOCOMNI_NNUNET_CKPT=   # from scratch (run.sh checks this before it reads the config)
export nnUNet_def_n_proc=3
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
log "200-epoch from-scratch start, HEAD $(git -C "$REPO" rev-parse --short HEAD)"

one() {  # group seed
  local g=$1 s=$2
  GPU=$s "$REPO/experiments/run.sh" $g $s > "$OUT/train_${g}_s$s.log" 2>&1 && log "${g}_s$s trained" \
    && GPU=$s CHK=checkpoint_best.pth SUFFIX=_best "$REPO/experiments/predict_test.sh" $g $s > "$OUT/predict_${g}_s${s}_best.log" 2>&1 \
    && GPU=$s "$REPO/experiments/predict_test.sh" $g $s > "$OUT/predict_${g}_s${s}_final.log" 2>&1 \
    && log "${g}_s$s test predicted (best, final)" || { log "${g}_s$s FAILED"; return 1; }
}
for s in 0 1 2; do
  ( sleep $((s * 900)); one T13bL $s; one T0L $s ) &
done
wait
log "training and test prediction finished; recovery / evaluation in server_trials_setting_recover.sh"
