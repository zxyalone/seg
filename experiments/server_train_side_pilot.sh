#!/usr/bin/env bash
# Training-side pilot (docs/训练侧频域计划.md §4): train <group> seed <seed> with run.sh
# (200 epochs, fold 0, VoComni weights of A13b, validation on the 21 fold-0 cases),
# predict MSD08 (no TTA, checkpoint_final; settings of server_external_msd08.sh),
# evaluate.py, then the frozen MSD08 metrics for A13b_s0 and this run and the
# registered continue conditions. Refuses to run unless the plan and the code it
# depends on are committed and unchanged.
#
#   tmux new -d -s pilot-E13 'GPU=0 experiments/server_train_side_pilot.sh E13 0'
set -euo pipefail
GROUP=${1:?group}; SEED=${2:?seed}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
DATA=${DATA:-/root/autodl-tmp/nnUNet_data}
ENV_BIN=${ENV_BIN:-/root/autodl-tmp/envs/mamba/bin}
RESULTS_ROOT=${RESULTS_ROOT:-$DATA/nnUNet_results_v2}
EXT=${EXT:-/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08}
GPU=${GPU:-0}
RUN=${GROUP}_s${SEED}
TRAINER=Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0
OUT=$REPO/results/train_side
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner_$RUN.log"; }

for f in docs/训练侧频域计划.md "experiments/configs/$GROUP.env" nnUNet/nnunetv2/training/nnUNetTrainer/phaser/lowcnr.py \
         nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_phaser.py experiments/analysis/train_side_pilot.py; do
  [[ -n $(git log -1 --format=%h -- "$f") ]] || { log "$f is not committed; refusing to run"; exit 1; }
  git diff --quiet HEAD -- "$f" || { log "$f differs from HEAD; refusing to run"; exit 1; }
done
export VOCOMNI_NNUNET_CKPT=${VOCOMNI_NNUNET_CKPT:-/root/autodl-tmp/Downstream_2/VoComni_nnunet.pt}
[[ $(md5sum "$VOCOMNI_NNUNET_CKPT" | cut -d' ' -f1) == ef44f2fb457ea3652c3566848b40b338 ]] \
  || { log "VoComni checkpoint md5 mismatch"; exit 1; }
log "$RUN pilot start, HEAD $(git rev-parse --short HEAD), GPU $GPU"

GPU=$GPU experiments/run.sh "$GROUP" "$SEED" > "$OUT/train_$RUN.log" 2>&1 || { log "$RUN training FAILED"; exit 1; }
log "$RUN trained and validated"

PRED=$RESULTS_ROOT/$RUN/$TRAINER/external_msd08_pred
n_cases=$(wc -l < "$EXT/cases.txt")
if [[ $(ls "$PRED"/*.nii.gz 2>/dev/null | wc -l) != "$n_cases" ]]; then
  (
    export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
    export nnUNet_results="$RESULTS_ROOT/$RUN" PYTHONPATH="$REPO/nnUNet" CUDA_VISIBLE_DEVICES=$GPU
    for v in $(env | grep -o '^PHASER_[A-Z0-9_]*'); do unset "$v"; done
    set -a; source "experiments/configs/$GROUP.env"; set +a
    export PHASER_SEED=$SEED
    "$ENV_BIN/nnUNetv2_predict" -i "$EXT/imagesTs" -o "$PRED" -d Dataset102_LiTS -c 3d_fullres \
      -tr nnUNetTrainer_phaser -p nnUNetPlans -f 0 -chk checkpoint_final.pth --disable_tta -device cuda -npp 2 -nps 2
  ) > "$OUT/predict_msd08_$RUN.log" 2>&1 || { log "$RUN MSD08 prediction FAILED"; exit 1; }
fi
[[ $(ls "$PRED"/*.nii.gz | wc -l) == "$n_cases" ]] || { log "$RUN MSD08 predictions incomplete"; exit 1; }
log "$RUN MSD08 predicted"

FOLD=$RESULTS_ROOT/$RUN/$TRAINER
"$ENV_BIN/python" experiments/eval/evaluate.py --pred-dir "$PRED" --gt-dir "$EXT/labelsTs" --image-dir "$EXT/imagesTs" \
  --cases-file "$EXT/cases.txt" --out-dir "$FOLD/eval_external_msd08" --workers 6 > "$OUT/eval_msd08_$RUN.log" 2>&1 \
  || { log "$RUN MSD08 evaluation FAILED"; exit 1; }
TAB=$OUT/pilot_msd08_$RUN
"$ENV_BIN/python" experiments/analysis/train_side_pilot.py metrics --runs A13b_s0 "$RUN" --out "$TAB" --jobs 2 \
  > "$OUT/metrics_$RUN.log" 2>&1 || { log "$RUN metrics FAILED"; exit 1; }
"$ENV_BIN/python" experiments/analysis/train_side_pilot.py decide --tables "$TAB" > "$OUT/decide_$RUN.log" 2>&1 \
  || { log "$RUN decision FAILED"; exit 1; }
log "$RUN pilot done; $TAB/DECISION.md"
