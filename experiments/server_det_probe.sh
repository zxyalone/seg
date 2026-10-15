#!/usr/bin/env bash
# D0 detection probe (docs/解码端病灶检出计划.md §3; criteria registered before
# running). LiTS fold-0 training set (head training samples) -> fit -> MSD08
# (decision cohort) -> LiTS validation (descriptive); A13b seeds 0-2. Runs only
# from a checkout whose HEAD contains the plan and the probe script unchanged,
# and waits until every GPU in $GPUS has no other compute process.
#
#   tmux new -d -s det-probe 'GPUS="0 1 2" experiments/server_det_probe.sh'
#
# Outputs: results/det_probe/{heads.npz,heads_info.json,<split>/{SUMMARY.md,summary.json,*.csv}}.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PY:-/root/autodl-tmp/envs/mamba/bin/python}
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"
PER_GPU=${PER_GPU:-2}     # three networks per shard; two shards fit a 12 GB card
N=$(( ${#GPU_LIST[@]} * PER_GPU ))
OUT=$REPO/results/det_probe
PLAN=docs/解码端病灶检出计划.md
SCRIPT=experiments/analysis/det_head_probe.py
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }

for f in "$PLAN" "$SCRIPT"; do
  [[ -n $(git log -1 --format=%h -- "$f") ]] || { log "$f is not committed; refusing to run"; exit 1; }
  git diff --quiet HEAD -- "$f" || { log "$f differs from HEAD; refusing to run"; exit 1; }
done
log "D0 start, HEAD $(git rev-parse --short HEAD), plan $(git log -1 --format=%h -- "$PLAN"), GPUs ${GPU_LIST[*]}, $N shards"
"$PY" "$SCRIPT" --selftest >> "$OUT/runner.log" 2>&1 || { log "selftest FAILED"; exit 1; }
for g in "${GPU_LIST[@]}"; do
  while [[ -n $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader) ]]; do
    log "GPU $g busy, waiting 5 min"; sleep 300
  done
done

extract() {  # split expected_count
  local split=$1 want=$2 ext
  mkdir -p "$OUT/$split"
  log "$split: extract"
  for ((i = 0; i < N; i++)); do
    g=${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      "$PY" "$SCRIPT" extract --split "$split" --shard "$i/$N" > "$OUT/$split/extract_$i.log" 2>&1 &
  done
  wait
  ext=$([[ $split == train ]] && echo npz || echo json)
  local got
  got=$(ls "$OUT/$split/cases/"*."$ext" 2>/dev/null | wc -l)
  [[ $got -eq $want ]] || { log "$split: $got/$want cases extracted; see extract_*.log"; exit 1; }
  log "$split: $got cases extracted"
}

extract train 83
"$PY" "$SCRIPT" fit > "$OUT/fit.log" 2>&1 || { log "fit FAILED"; exit 1; }
log "heads fitted (heads_info.json)"
for split in msd08 lits_val; do
  want=$(grep -c . "$( [[ $split == msd08 ]] && echo /root/autodl-tmp/ext_data/msd08/Dataset104_MSD08/cases.txt \
         || echo experiments/eval/val_cases.txt)")
  extract "$split" "$want"
  "$PY" "$SCRIPT" summarize --split "$split" > "$OUT/$split/summarize.log" 2>&1 \
    || { log "$split: summarize FAILED"; exit 1; }
  log "$split: done; $OUT/$split/SUMMARY.md"
done
log "D0 finished; decision in results/det_probe/msd08/SUMMARY.md"
