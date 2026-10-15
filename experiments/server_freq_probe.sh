#!/usr/bin/env bash
# F0 feature probe (docs/频域模块计划.md §3; criteria registered before running).
# MSD08 (decision cohort) then the LiTS fold-0 validation set (descriptive);
# A13b and A0, seeds 0-2. Runs only from a checkout whose HEAD contains the
# plan and the probe script unchanged, and waits until GPU $GPU has no other
# compute process, so it never shares a card with training or prediction.
#
#   tmux new -d -s freq-probe 'GPU=0 experiments/server_freq_probe.sh'
#
# Outputs: results/freq_probe/<cohort>/{SUMMARY.md,summary.json,*.csv,cases/}.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PY:-/root/autodl-tmp/envs/mamba/bin/python}
GPU=${GPU:-0}
SHARDS=${SHARDS:-3}
OUT=$REPO/results/freq_probe
PLAN=docs/频域模块计划.md
SCRIPT=experiments/analysis/freq_feature_probe.py
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }

# Pre-registration: plan and script committed and unchanged.
for f in "$PLAN" "$SCRIPT"; do
  [[ -n $(git log -1 --format=%h -- "$f") ]] || { log "$f is not committed; refusing to run"; exit 1; }
  git diff --quiet HEAD -- "$f" || { log "$f differs from HEAD; refusing to run"; exit 1; }
done
log "F0 probe start, HEAD $(git rev-parse --short HEAD), plan $(git log -1 --format=%h -- "$PLAN"), GPU $GPU, $SHARDS shards"
"$PY" "$SCRIPT" --selftest >> "$OUT/runner.log" 2>&1 || { log "selftest FAILED"; exit 1; }

while [[ -n $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader) ]]; do
  log "GPU $GPU busy, waiting 5 min"; sleep 300
done

for cohort in msd08 lits_val; do
  mkdir -p "$OUT/$cohort"
  log "$cohort: extract"
  for ((i = 0; i < SHARDS; i++)); do
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      "$PY" "$SCRIPT" extract --cohort "$cohort" --shard "$i/$SHARDS" > "$OUT/$cohort/extract_$i.log" 2>&1 &
  done
  wait
  n_cases=$(grep -c . "$( [[ $cohort == msd08 ]] && echo /root/autodl-tmp/ext_data/msd08/Dataset104_MSD08/cases.txt \
            || echo experiments/eval/val_cases.txt)")
  n_done=$(ls "$OUT/$cohort/cases/"*.json 2>/dev/null | wc -l)
  [[ $n_done -eq $n_cases ]] || { log "$cohort: $n_done/$n_cases cases extracted; see extract_*.log"; exit 1; }
  "$PY" "$SCRIPT" summarize --cohort "$cohort" > "$OUT/$cohort/summarize.log" 2>&1 \
    || { log "$cohort: summarize FAILED"; exit 1; }
  log "$cohort: done ($n_done cases); $OUT/$cohort/SUMMARY.md"
done
log "F0 probe finished; decision in results/freq_probe/msd08/SUMMARY.md"
