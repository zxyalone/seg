#!/usr/bin/env bash
# SALT G0 (docs/SALT模型设计.md §3): no training.
#   1. GPU preflight: check_grads.py for Q1-Q4 and F1 at 128^3, batch 2, AMP, 50 steps (peak memory, finite losses,
#      every new tensor trained, lesion query loss and fused inference exercised).
#   2. Anchor coverage probe with A0_s0 on MSD08 (303, criterion) and the LiTS validation cases (21, descriptive).
# Writes results/salt/g0/ and prints the decision; it never starts a training run.
#
#   tmux new -d -s salt-g0 'GPUS="0 1 2" experiments/server_salt_g0.sh'
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PY:-/root/autodl-tmp/envs/mamba/bin/python}
OUT=$REPO/results/salt/g0
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
for f in docs/SALT模型设计.md experiments/analysis/salt_anchor_probe.py experiments/check_grads.py \
         nnUNet/nnunetv2/training/nnUNetTrainer/phaser/lesion_query.py nnUNet/nnunetv2/training/nnUNetTrainer/phaser/freq_fusion3d.py \
         nnUNet/nnunetv2/training/nnUNetTrainer/phaser/unet.py nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_phaser.py; do
  [[ -n $(git log -1 --format=%h -- "$f") ]] && git diff --quiet HEAD -- "$f" || { log "$f not committed or modified; refusing"; exit 1; }
done
log "G0 start, HEAD $(git rev-parse --short HEAD)"
export PYTHONPATH=$REPO/nnUNet OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
read -r -a G <<< "${GPUS:-0}"
wait_idle() { while nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | grep -q .; do sleep 60; done; }

# 1. preflight on the first GPU
wait_idle "${G[0]}"
CUDA_VISIBLE_DEVICES=${G[0]} "$PY" experiments/check_grads.py --groups Q1 Q2 Q3 Q4 F1 --steps 50 --patch 128 \
  --device cuda --amp > "$OUT/preflight.log" 2>&1
grep -E "^\[|PASS|FAIL|peak CUDA|SUMMARY" "$OUT/preflight.log" | tee -a "$OUT/runner.log"
grep -q "SUMMARY:.*FAIL" "$OUT/preflight.log" && { log "preflight FAILED (see preflight.log)"; exit 1; }
peak=$(grep -o "peak CUDA allocated=[0-9.]*" "$OUT/preflight.log" | grep -o "[0-9.]*$" | sort -n | tail -1)
log "preflight passed; largest peak allocated ${peak} GiB (limit 11.0 on 12 GB cards)"
"$PY" -c "import sys; sys.exit(0 if float('${peak:-99}') <= 11.0 else 1)" \
  || { log "peak memory above 11 GiB: apply the fallback of §3.1 in a commit, then rerun"; exit 1; }

# 2. anchor coverage probe, one shard per GPU for each split
for i in "${!G[@]}"; do
  wait_idle "${G[$i]}"
  ( CUDA_VISIBLE_DEVICES=${G[$i]} "$PY" experiments/analysis/salt_anchor_probe.py run --split msd08 --shard "$i/${#G[@]}"
    CUDA_VISIBLE_DEVICES=${G[$i]} "$PY" experiments/analysis/salt_anchor_probe.py run --split val --shard "$i/${#G[@]}"
  ) > "$OUT/probe_$i.log" 2>&1 &
done
wait
n=$(ls "$OUT/msd08/"*.json 2>/dev/null | wc -l)
[[ $n -eq 303 ]] || { log "MSD08 probe: $n/303 cases; see probe_*.log"; exit 1; }
n=$(ls "$OUT/val/"*.json 2>/dev/null | wc -l)
[[ $n -eq 21 ]] || { log "validation probe: $n/21 cases; see probe_*.log"; exit 1; }
"$PY" experiments/analysis/salt_anchor_probe.py summarize > "$OUT/summarize.log" 2>&1 || { log "summarize FAILED"; exit 1; }
log "G0 done: $(grep -o 'G0（.*' "$OUT/G0.md" | head -1)"
log "next: commit results/salt/g0, then start the pilots of §4 by hand"
