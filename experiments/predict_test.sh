#!/usr/bin/env bash
# E15: one-time prediction and evaluation of the 27 internal test cases (and,
# with EXTERNAL_IMAGES/EXTERNAL_LABELS set, the external cohort) for one run.
#
#   experiments/predict_test.sh A6 0
#
# Refuses to run until experiments/PROTOCOL_FROZEN exists. Create it once the
# evaluation protocol (Section 4.3: primary endpoint, post-processing, case
# lists) is final, and commit it:
#   git rev-parse HEAD > experiments/PROTOCOL_FROZEN && git add experiments/PROTOCOL_FROZEN && git commit
set -euo pipefail
GROUP=${1:?group}; SEED=${2:?seed}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/root/autodl-tmp/nnUNet_data}
ENV_BIN=${ENV_BIN:-/root/autodl-tmp/envs/mamba/bin}
RESULTS_ROOT=${RESULTS_ROOT:-$DATA/nnUNet_results_v2}
GPU=${GPU:-0}

[[ -f $REPO/experiments/PROTOCOL_FROZEN ]] || {
  echo "experiments/PROTOCOL_FROZEN missing: the test set stays locked until the protocol is frozen" >&2; exit 1; }

RUN="${GROUP}_s${SEED}"
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RESULTS_ROOT/$RUN"
export PYTHONPATH="$REPO/nnUNet${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=$GPU
for v in $(env | grep -o '^PHASER_[A-Z0-9_]*'); do unset "$v"; done
set -a; source "$REPO/experiments/configs/$GROUP.env"; set +a
export PHASER_SEED=$SEED
FOLD_DIR="$nnUNet_results/Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0"
[[ -f $FOLD_DIR/checkpoint_final.pth ]] || { echo "no checkpoint_final for $RUN" >&2; exit 1; }

predict() {  # input_dir output_dir
  "$ENV_BIN/nnUNetv2_predict" -i "$1" -o "$2" -d Dataset102_LiTS -c 3d_fullres \
    -tr nnUNetTrainer_phaser -p nnUNetPlans -f 0 -chk checkpoint_final.pth \
    --disable_tta -device cuda -npp 1 -nps 1
}

predict "$nnUNet_raw/Dataset102_LiTS/imagesTs" "$FOLD_DIR/test_pred"
"$ENV_BIN/python" "$REPO/experiments/eval/evaluate.py" \
  --pred-dir "$FOLD_DIR/test_pred" --gt-dir "$nnUNet_raw/Dataset101_LiTS/labelsTr" \
  --image-dir "$nnUNet_raw/Dataset102_LiTS/imagesTs" \
  --cases-file "$REPO/experiments/eval/test_cases.txt" --out-dir "$FOLD_DIR/eval_test"

if [[ -n ${EXTERNAL_IMAGES:-} && -n ${EXTERNAL_LABELS:-} ]]; then
  predict "$EXTERNAL_IMAGES" "$FOLD_DIR/external_pred"
  "$ENV_BIN/python" "$REPO/experiments/eval/evaluate.py" \
    --pred-dir "$FOLD_DIR/external_pred" --gt-dir "$EXTERNAL_LABELS" \
    --image-dir "$EXTERNAL_IMAGES" --out-dir "$FOLD_DIR/eval_external"
fi
