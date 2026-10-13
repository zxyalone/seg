#!/usr/bin/env bash
# External validation on MSD Task08 (plan §11, 2026-10-13, registered before
# inference). Groups A0, A1, A13b x 3 seeds; inference settings of
# predict_test.sh (no TTA, checkpoint_final); reference labels (A0 majority
# liver + Task08 tumour) built after all predictions exist; then evaluate.py
# and the frozen protocol's metrics and statistics.
#
#   tmux new -d -s msd08-eval 'experiments/server_external_msd08.sh'
#   STEP=3 experiments/server_external_msd08.sh      # resume from step 3
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/root/autodl-tmp/nnUNet_data}
ENV_BIN=${ENV_BIN:-/root/autodl-tmp/envs/mamba/bin}
RESULTS_ROOT=${RESULTS_ROOT:-$DATA/nnUNet_results_v2}
RAW=${RAW:-/root/autodl-tmp/ext_data/msd08/raw/Task08_HepaticVessel}
EXT=${EXT:-/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08}
CHECK=/root/autodl-tmp/ext_data/msd08/overlap_check/overlap.json
OUT=${OUT:-$REPO/results/external_msd08}
STEP=${STEP:-1}
PY=$ENV_BIN/python
TRAINER=Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0
RUNS=(A0_s0 A0_s1 A0_s2 A1_s0 A1_s1 A1_s2 A13b_s0 A13b_s1 A13b_s2)   # A0 first: one per GPU
cd "$REPO"
grep -q '"independent": true' "$CHECK" || { echo "overlap check missing or failed" >&2; exit 1; }
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
log "MSD08 external start, HEAD $(git rev-parse --short HEAD), step $STEP"

if (( STEP <= 1 )); then
  "$PY" -I experiments/external/prepare_msd08.py stage --raw "$RAW" --out "$EXT" >> "$OUT/runner.log" 2>&1
fi

predict_one() {  # run gpu
  local run=$1 gpu=$2 group=${1%_s*} seed=${1##*_s}
  (
    export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
    export nnUNet_results="$RESULTS_ROOT/$run" PYTHONPATH="$REPO/nnUNet" CUDA_VISIBLE_DEVICES=$gpu
    for v in $(env | grep -o '^PHASER_[A-Z0-9_]*'); do unset "$v"; done
    set -a; source "experiments/configs/$group.env"; set +a
    export PHASER_SEED=$seed
    "$ENV_BIN/nnUNetv2_predict" -i "$EXT/imagesTs" -o "$RESULTS_ROOT/$run/$TRAINER/external_msd08_pred" \
      -d Dataset102_LiTS -c 3d_fullres -tr nnUNetTrainer_phaser -p nnUNetPlans -f 0 -chk checkpoint_final.pth \
      --disable_tta -device cuda -npp 2 -nps 2
  )
}
n_cases=$(wc -l < "$EXT/cases.txt")
count() { ls "$RESULTS_ROOT/$1/$TRAINER/external_msd08_pred/"*.nii.gz 2>/dev/null | wc -l; }

if (( STEP <= 2 )); then
  for g in 0 1 2; do
    (
      # GPUs 0 and 1 run the SegResNet transfer evaluation until it finishes; GPU 2 is free
      if (( g < 2 )); then while tmux has-session -t transfer-eval 2>/dev/null; do sleep 60; done; fi
      for i in "${!RUNS[@]}"; do
        (( i % 3 == g )) || continue
        run=${RUNS[$i]}
        [[ $(count "$run") == "$n_cases" ]] && { log "$run predictions exist, skipped"; continue; }
        log "$run prediction start (GPU $g)"
        predict_one "$run" "$g" > "$OUT/predict_$run.log" 2>&1
        log "$run prediction done ($(count "$run") / $n_cases)"
      done
    ) &
  done
  wait
  for run in "${RUNS[@]}"; do [[ $(count "$run") == "$n_cases" ]] || { log "$run incomplete"; exit 1; }; done
fi

if (( STEP <= 3 )); then
  "$PY" -I experiments/external/prepare_msd08.py reference --raw "$RAW" --out "$EXT" \
    --a0 "$RESULTS_ROOT/A0_s0/$TRAINER/external_msd08_pred" "$RESULTS_ROOT/A0_s1/$TRAINER/external_msd08_pred" \
         "$RESULTS_ROOT/A0_s2/$TRAINER/external_msd08_pred" > "$OUT/reference.log" 2>&1
  cp "$EXT/reference_manifest.json" "$OUT/"
  log "reference labels built"
fi

if (( STEP <= 4 )); then
  for run in "${RUNS[@]}"; do
    fold="$RESULTS_ROOT/$run/$TRAINER"
    [[ -f $fold/eval_external_msd08/summary.json ]] && continue
    "$PY" experiments/eval/evaluate.py --pred-dir "$fold/external_msd08_pred" --gt-dir "$EXT/labelsTs" \
      --image-dir "$EXT/imagesTs" --cases-file "$EXT/cases.txt" --out-dir "$fold/eval_external_msd08" \
      --workers 6 > "$OUT/eval_$run.log" 2>&1 &
    while (( $(jobs -rp | wc -l) >= 3 )); do sleep 10; done
  done
  wait
  for run in "${RUNS[@]}"; do [[ -f $RESULTS_ROOT/$run/$TRAINER/eval_external_msd08/summary.json ]] || { log "$run eval missing"; exit 1; }; done
  log "evaluation done"
fi

if (( STEP <= 5 )); then
  (cd experiments/analysis && "$PY" external_msd08_protocol.py metrics --split external_msd08 --jobs 9 \
     --out "$OUT/tables") > "$OUT/metrics.log" 2>&1
  (cd experiments/analysis && "$PY" external_msd08_protocol.py stats --tables "$OUT/tables" \
     --out "$OUT/stats") > "$OUT/stats.log" 2>&1
fi
log "MSD08 external complete"
echo "EXTERNAL_EXIT 0" >> "$OUT/runner.log"
