#!/usr/bin/env bash
# A14 (Pixel2Cancer comparator, plan §11 2026-10-14): index the offline
# variants, train seeds 0-2 in parallel (one per GPU), predict the test set
# (frozen protocol settings), then the descriptive evaluation.
#
#   tmux new -d -s a14 'experiments/server_a14.sh'
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_BIN=${ENV_BIN:-/root/autodl-tmp/envs/mamba/bin}
SYNTH=/root/autodl-tmp/nnUNet_data/p2c_synth/Dataset102_fold0
OUT=$REPO/results/a14_pixel2cancer
export VOCOMNI_NNUNET_CKPT=/root/autodl-tmp/Downstream_2/VoComni_nnunet.pt   # same file as A0 / A13b
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
log "A14 start, HEAD $(git -C "$REPO" rev-parse --short HEAD)"

"$ENV_BIN/python" "$REPO/experiments/external/p2c_synthesize.py" --out "$SYNTH" --index | tee -a "$OUT/runner.log"
cp "$SYNTH/index.json" "$OUT/p2c_index.json"; cat "$SYNTH"/stats_part*.jsonl > "$OUT/p2c_variant_stats.jsonl"

for s in 0 1 2; do
  (
    GPU=$s "$REPO/experiments/run.sh" A14 $s > "$OUT/train_A14_s$s.log" 2>&1 && log "A14_s$s trained" \
      && GPU=$s "$REPO/experiments/predict_test.sh" A14 $s > "$OUT/predict_A14_s$s.log" 2>&1 && log "A14_s$s test predicted"
  ) &
done
wait

cd "$REPO/experiments/analysis"
for split in val test; do
  "$ENV_BIN/python" a14_eval.py metrics --split $split --out "$OUT/${split}_tables" --jobs 3 > "$OUT/metrics_$split.log" 2>&1 \
    || { log "metrics $split failed"; exit 1; }
done
"$ENV_BIN/python" a14_eval.py stats --out "$OUT" > "$OUT/stats.log" 2>&1 || { log "stats failed"; exit 1; }
log "A14 complete"
echo "A14_EXIT 0" >> "$OUT/runner.log"
