#!/usr/bin/env bash
# One-off recovery for the msd08 shard-1 OOM (plan §10, 2026-10-15).
cd /root/autodl-tmp/seg
L=results/freq_probe
PY=/root/autodl-tmp/envs/mamba/bin/python
log() { echo "[$(date "+%F %T")] $*" | tee -a $L/runner.log; }
log "fix: waiting for GPU 1 to rerun msd08 shard 1/3"
while [[ -n $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader) ]]; do sleep 120; done
log "fix: msd08 shard 1/3 on GPU 1"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY experiments/analysis/freq_feature_probe.py \
  extract --cohort msd08 --shard 1/3 > $L/msd08/extract_1_rerun.log 2>&1 || { log "fix: shard 1 FAILED"; exit 1; }
log "fix: shard 1 done; waiting for the first runner to exit"
while pgrep -f "[s]erver_freq_probe.sh" > /dev/null; do sleep 60; done
git pull -q --ff-only || { log "fix: git pull FAILED"; exit 1; }
log "fix: pulled $(git rev-parse --short HEAD); restarting the runner (completed cases are skipped)"
GPU=0 experiments/server_freq_probe.sh
