#!/usr/bin/env bash
# External validation on 3D-IRCADb-01 (plan §11, registered before inference).
# The 16 fold-0 models of the test protocol, the same inference settings as
# predict_test.sh (no TTA, checkpoint_final), evaluate.py, then the frozen
# protocol's metrics and statistics on the 'external' split.
#
#   experiments/server_external_eval.sh            # all steps
#   STEP=3 experiments/server_external_eval.sh     # resume from step 3
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/root/autodl-tmp/nnUNet_data}
ENV_BIN=${ENV_BIN:-/root/autodl-tmp/envs/mamba/bin}
RESULTS_ROOT=${RESULTS_ROOT:-$DATA/nnUNet_results_v2}
EXT=${EXT:-/root/autodl-tmp/ext_data/Dataset103_3DIRCADb}
OUT=${OUT:-$RESULTS_ROOT/_analysis/external_ircadb}
STEP=${STEP:-1}
PY=$ENV_BIN/python
TRAINER=Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0
RUNS=(A0_s0 A0_s1 A0_s2 A1_s0 A1_s1 A1_s2 A11_s0 A11_s1 A11_s2 A13a_s0 A13a_s1 A13a_s2 A13b_s0 A13b_s1 A13b_s2 A12_s0)
cd "$REPO"
[[ -f $EXT/cases.txt && -f $EXT/../overlap_check/overlap.json ]] || { echo "dataset or overlap check missing" >&2; exit 1; }
grep -q '"independent": true' "$EXT/../overlap_check/overlap.json" || { echo "overlap check did not pass" >&2; exit 1; }
mkdir -p "$OUT"
echo "[$(date '+%F %T')] external start, HEAD $(git rev-parse --short HEAD), step $STEP" | tee -a "$OUT/runner.log"

predict_one() {  # run gpu
  local run=$1 gpu=$2 group=${1%_s*} seed=${1##*_s}
  (
    export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
    export nnUNet_results="$RESULTS_ROOT/$run" PYTHONPATH="$REPO/nnUNet" CUDA_VISIBLE_DEVICES=$gpu
    for v in $(env | grep -o '^PHASER_[A-Z0-9_]*'); do unset "$v"; done
    set -a; source "experiments/configs/$group.env"; set +a
    export PHASER_SEED=$seed
    local fold="$RESULTS_ROOT/$run/$TRAINER"
    "$ENV_BIN/nnUNetv2_predict" -i "$EXT/imagesTs" -o "$fold/external_pred" -d Dataset102_LiTS -c 3d_fullres \
      -tr nnUNetTrainer_phaser -p nnUNetPlans -f 0 -chk checkpoint_final.pth --disable_tta -device cuda -npp 1 -nps 1
    "$PY" experiments/eval/evaluate.py --pred-dir "$fold/external_pred" --gt-dir "$EXT/labelsTs" \
      --image-dir "$EXT/imagesTs" --cases-file "$EXT/cases.txt" --out-dir "$fold/eval_external"
  )
}

if (( STEP <= 1 )); then
  n_cases=$(wc -l < "$EXT/cases.txt")
  for g in 0 1 2; do
    (
      for i in "${!RUNS[@]}"; do
        (( i % 3 == g )) || continue
        run=${RUNS[$i]}
        [[ -f $RESULTS_ROOT/$run/$TRAINER/eval_external/summary.json ]] && { echo "$run done, skipped"; continue; }
        predict_one "$run" "$g" > "$OUT/predict_$run.log" 2>&1
        echo "[$(date '+%F %T')] $run external inference + evaluation done (GPU $g)"
      done
    ) >> "$OUT/runner.log" 2>&1 &
  done
  wait
  for run in "${RUNS[@]}"; do
    n=$(ls "$RESULTS_ROOT/$run/$TRAINER/external_pred/"*.nii.gz | wc -l)
    [[ $n == "$n_cases" && -f $RESULTS_ROOT/$run/$TRAINER/eval_external/summary.json ]] || { echo "$run incomplete ($n)" >&2; exit 1; }
  done
fi

if (( STEP <= 2 )); then
  for run in "${RUNS[@]}"; do mkdir -p "results/$run"; cp -r "$RESULTS_ROOT/$run/$TRAINER/eval_external" "results/$run/"; done
fi

if (( STEP <= 3 )); then
  (cd experiments/analysis && "$PY" external_protocol.py metrics --split external --jobs 16 --out "$OUT/tables") > "$OUT/metrics.log" 2>&1
fi

if (( STEP <= 4 )); then
  (cd experiments/analysis && "$PY" external_protocol.py stats --tables "$OUT/tables" --out "$OUT/stats") > "$OUT/stats.log" 2>&1
fi
echo "[$(date '+%F %T')] external complete" | tee -a "$OUT/runner.log"
echo "EXTERNAL_EXIT 0" >> "$OUT/runner.log"
