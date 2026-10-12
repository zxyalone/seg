#!/usr/bin/env bash
# One-time test evaluation under the frozen protocol (docs/测试集冻结协议_草案.md).
# Runs only from a clean checkout whose experiments/PROTOCOL_FROZEN names an
# ancestor commit with no later change to the code, configs or protocol.
#
#   experiments/server_test_protocol.sh            # all steps
#   STEP=3 experiments/server_test_protocol.sh     # resume from step 3
#
# 1  test inference + evaluate.py for 16 runs (predict_test.sh, 3 GPUs)
# 2  archive eval_test/ of every run into results/<run>/eval_test
# 3  fixed 4-mm tables (test_protocol.py metrics --split test)
# 4  test threshold curves for the 15 primary runs + summary
# 5  curve parity report (reported as is, never used to replace results)
# 6  protocol statistics (test_protocol.py stats)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/root/autodl-tmp/nnUNet_data}
ENV_BIN=${ENV_BIN:-/root/autodl-tmp/envs/mamba/bin}
RESULTS_ROOT=${RESULTS_ROOT:-$DATA/nnUNet_results_v2}
OUT=${OUT:-$RESULTS_ROOT/_analysis/test_protocol}
STEP=${STEP:-1}
PY=$ENV_BIN/python
TRAINER=Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0
RUNS=(A0_s0 A0_s1 A0_s2 A1_s0 A1_s1 A1_s2 A11_s0 A11_s1 A11_s2 A13a_s0 A13a_s1 A13a_s2 A13b_s0 A13b_s1 A13b_s2 A12_s0)
CURVE_RUNS="A0:0,1,2 A1:0,1,2 A11:0,1,2 A13a:0,1,2 A13b:0,1,2"

cd "$REPO"
[[ -f experiments/PROTOCOL_FROZEN ]] || { echo "PROTOCOL_FROZEN missing" >&2; exit 1; }
FROZEN=$(tr -d '[:space:]' < experiments/PROTOCOL_FROZEN)
git merge-base --is-ancestor "$FROZEN" HEAD || { echo "frozen commit $FROZEN is not an ancestor of HEAD" >&2; exit 1; }
if ! git diff --quiet "$FROZEN" HEAD -- experiments nnUNet docs/测试集冻结协议_草案.md ':!experiments/PROTOCOL_FROZEN' ':!experiments/dashboard'; then
  echo "code, configs or protocol changed after the freeze ($FROZEN)" >&2; exit 1
fi
git diff --quiet HEAD -- experiments nnUNet || { echo "uncommitted changes under experiments/ or nnUNet/" >&2; exit 1; }
mkdir -p "$OUT"
echo "[$(date '+%F %T')] test protocol start, HEAD $(git rev-parse --short HEAD), frozen $FROZEN, step $STEP" | tee -a "$OUT/runner.log"

if (( STEP <= 1 )); then
  for g in 0 1 2; do
    (
      for i in "${!RUNS[@]}"; do
        (( i % 3 == g )) || continue
        run=${RUNS[$i]}; group=${run%_s*}; seed=${run##*_s}
        [[ -f $RESULTS_ROOT/$run/$TRAINER/eval_test/summary.json ]] && { echo "$run done, skipped"; continue; }
        GPU=$g experiments/predict_test.sh "$group" "$seed" > "$OUT/predict_$run.log" 2>&1
        echo "[$(date '+%F %T')] $run test inference + evaluation done (GPU $g)"
      done
    ) >> "$OUT/runner.log" 2>&1 &
  done
  wait
  for run in "${RUNS[@]}"; do
    n=$(ls "$RESULTS_ROOT/$run/$TRAINER/test_pred/"*.nii.gz | wc -l)
    [[ $n == 27 && -f $RESULTS_ROOT/$run/$TRAINER/eval_test/summary.json ]] || { echo "$run incomplete ($n masks)" >&2; exit 1; }
  done
fi

if (( STEP <= 2 )); then
  for run in "${RUNS[@]}"; do
    mkdir -p "results/$run"
    cp -r "$RESULTS_ROOT/$run/$TRAINER/eval_test" "results/$run/"
  done
fi

if (( STEP <= 3 )); then
  (cd experiments/analysis && "$PY" test_protocol.py metrics --split test --jobs 16 --out "$OUT/tables") > "$OUT/metrics.log" 2>&1
fi

if (( STEP <= 4 )); then
  for g in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$g "$PY" -u experiments/analysis/missed_lesion_probability.py run --runs $CURVE_RUNS \
      --image-dir "$DATA/nnUNet_raw/Dataset102_LiTS/imagesTs" --label-dir "$DATA/nnUNet_raw/Dataset101_LiTS/labelsTr" \
      --cases-file experiments/eval/test_cases.txt --shard "$g/3" --device cuda --workers 2 --offload-models \
      --out "$OUT/curves" > "$OUT/curves_shard$g.log" 2>&1 &
  done
  wait
  n=$(find "$OUT/curves/cases" -name '*.json' | wc -l)
  [[ $n == 405 ]] || { echo "test curves incomplete: $n / 405" >&2; exit 1; }
  "$PY" experiments/analysis/missed_lesion_probability.py summarize --out "$OUT/curves" \
    --results-dir results --eval-subdir eval_test > "$OUT/curves_summary.log" 2>&1
fi

if (( STEP <= 5 )); then
  (cd experiments/analysis && "$PY" check_curve_parity.py --jobs 15 --curves "$OUT/curves/cases" \
    --runs A0_s0 A0_s1 A0_s2 A1_s0 A1_s1 A1_s2 A11_s0 A11_s1 A11_s2 A13a_s0 A13a_s1 A13a_s2 A13b_s0 A13b_s1 A13b_s2 \
    --pred-subdir test_pred --eval-subdir eval_test --gt-dir "$DATA/nnUNet_raw/Dataset101_LiTS/labelsTr" \
    --cases-file "$REPO/experiments/eval/test_cases.txt" --eval-root "$REPO/results" --out "$OUT/curves/parity") \
    > "$OUT/parity.log" 2>&1 || echo "curve parity check did not pass; reported as is (see parity.log)" | tee -a "$OUT/runner.log"
fi

if (( STEP <= 6 )); then
  (cd experiments/analysis && "$PY" test_protocol.py stats --tables "$OUT/tables" \
    --sweep "$OUT/curves/threshold_sweep_per_run.csv" --out "$OUT/stats") > "$OUT/stats.log" 2>&1
fi
echo "[$(date '+%F %T')] test protocol complete" | tee -a "$OUT/runner.log"
