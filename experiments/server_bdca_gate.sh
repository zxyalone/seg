#!/usr/bin/env bash
# B13 gate (docs/训练侧频域计划.md §3.2): operator check on the fold-0 training lesions
# (CPU, 8 shards); the B13 pilot starts on $GPU only if all three registered criteria pass.
#
#   tmux new -d -s bdca-gate 'GPU=1 experiments/server_bdca_gate.sh'
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PY:-/root/autodl-tmp/envs/mamba/bin/python}
OUT=$REPO/results/train_side/operator_check
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
for f in docs/训练侧频域计划.md experiments/analysis/bdca_operator_check.py nnUNet/nnunetv2/training/nnUNetTrainer/phaser/lowcnr.py; do
  [[ -n $(git log -1 --format=%h -- "$f") ]] && git diff --quiet HEAD -- "$f" || { log "$f not committed or modified; refusing"; exit 1; }
done
log "operator check start, HEAD $(git rev-parse --short HEAD)"
export PYTHONPATH=$REPO/nnUNet OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
for i in 0 1 2 3 4 5 6 7; do
  "$PY" experiments/analysis/bdca_operator_check.py run --shard "$i/8" > "$OUT/run_$i.log" 2>&1 &
done
wait
n=$(ls "$OUT/cases/"*.json 2>/dev/null | wc -l)
[[ $n -eq 83 ]] || { log "operator check: $n/83 cases; see run_*.log"; exit 1; }
"$PY" experiments/analysis/bdca_operator_check.py summarize > "$OUT/summarize.log" 2>&1 || { log "summarize FAILED"; exit 1; }
if "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/summary.json'))['PASS'] else 1)"; then
  log "operator check PASSED; starting the B13 pilot on GPU ${GPU:-1}"
  GPU=${GPU:-1} experiments/server_train_side_pilot.sh B13 0
else
  log "operator check NOT passed (see SUMMARY.md); B13 is not trained (plan §3.2)"
fi
