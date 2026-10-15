#!/usr/bin/env bash
# F0 feature probe v2 (docs/频域模块计划.md §3.5: position-matched negatives;
# criteria registered before running). MSD08 (decision cohort) then the LiTS
# fold-0 validation set (descriptive); A13b and A0, seeds 0-2. Runs only from a
# checkout whose HEAD contains the plan and the probe script unchanged, and
# waits until every GPU in $GPUS has no other compute process.
#
#   tmux new -d -s freq-probe 'GPUS="0 1 2" experiments/server_freq_probe.sh'
#
# Outputs: results/freq_probe_v2/<cohort>/{SUMMARY.md,summary.json,*.csv,cases/}.
# (v1, invalidated by the position confound, is kept in results/freq_probe/.)
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PY:-/root/autodl-tmp/envs/mamba/bin/python}
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"
# Each shard keeps all six networks on the GPU (~4.7 GB on a 12 GB card): at most 2 shards per GPU.
PER_GPU=${PER_GPU:-2}
N=$(( ${#GPU_LIST[@]} * PER_GPU ))
OUT=$REPO/results/freq_probe_v2
PLAN=docs/频域模块计划.md
SCRIPT=experiments/analysis/freq_feature_probe.py
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }

# Pre-registration: plan and script committed and unchanged.
for f in "$PLAN" "$SCRIPT"; do
  [[ -n $(git log -1 --format=%h -- "$f") ]] || { log "$f is not committed; refusing to run"; exit 1; }
  git diff --quiet HEAD -- "$f" || { log "$f differs from HEAD; refusing to run"; exit 1; }
done
log "F0 probe v2 start, HEAD $(git rev-parse --short HEAD), plan $(git log -1 --format=%h -- "$PLAN"), GPUs ${GPU_LIST[*]}, $N shards"
"$PY" "$SCRIPT" --selftest >> "$OUT/runner.log" 2>&1 || { log "selftest FAILED"; exit 1; }

for g in "${GPU_LIST[@]}"; do
  while [[ -n $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader) ]]; do
    log "GPU $g busy, waiting 5 min"; sleep 300
  done
done

for cohort in msd08 lits_val; do
  mkdir -p "$OUT/$cohort"
  log "$cohort: extract"
  for ((i = 0; i < N; i++)); do
    g=${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      "$PY" "$SCRIPT" extract --cohort "$cohort" --shard "$i/$N" > "$OUT/$cohort/extract_$i.log" 2>&1 &
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
log "F0 probe v2 finished; decision in results/freq_probe_v2/msd08/SUMMARY.md"
