#!/usr/bin/env bash
# One-off recovery for the MSD08 shard-0 OOM (plan §9, 2026-10-16).
cd /root/autodl-tmp/seg
L=results/det_probe
PY=/root/autodl-tmp/envs/mamba/bin/python
log() { echo "[$(date "+%F %T")] $*" | tee -a $L/runner.log; }
log "fix: msd08 shard 0/6 rerun on GPU 0 with the OOM fallback ($(git rev-parse --short HEAD))"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY experiments/analysis/det_head_probe.py \
  extract --split msd08 --shard 0/6 > $L/msd08/extract_0_rerun.log 2>&1 || { log "fix: shard 0 FAILED"; exit 1; }
log "fix: shard 0 done; waiting for the first runner to exit"
while pgrep -f "[s]erver_det_probe.sh" > /dev/null; do sleep 60; done
git pull -q --ff-only || { log "fix: git pull FAILED"; exit 1; }
log "fix: pulled $(git rev-parse --short HEAD); restarting the runner (completed cases and the fitted heads are kept)"
GPUS="0 1 2" experiments/server_det_probe.sh
