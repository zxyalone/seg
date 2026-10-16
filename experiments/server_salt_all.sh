#!/usr/bin/env bash
# One-shot SALT runner (docs/SALT模型设计.md): G0 (preflight + anchor probe), then the seed-0 pilots on every
# idle GPU. Q4, Q1, F1 start first; Q2, Q3 only after Q4 passes (§4). Idempotent: finished steps are skipped,
# so simply restart it after an interruption.
#
#   tmux new -d -s salt 'bash /root/autodl-tmp/salt_work/seg/experiments/server_salt_all.sh'
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
OUT=$REPO/results/salt
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/all.log"; }

# --- resources: all GPUs, data-augmentation workers split over the CPUs --------------------------------------
mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
NG=${#GPU_LIST[@]}
NCPU=$(nproc)
MEM_GB=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
# nnU-Net keeps ~1.5-2 GB of RAM per augmentation worker at 128^3; leave 8 GB for the system.
DA=$(( NCPU / NG - 1 )); MAXDA=$(( (MEM_GB - 8) / NG / 2 )); (( DA > MAXDA )) && DA=$MAXDA; (( DA < 2 )) && DA=2
export nnUNet_n_proc_DA=$DA nnUNet_def_n_proc=$DA OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VOCOMNI_NNUNET_CKPT=${VOCOMNI_NNUNET_CKPT:-/root/autodl-tmp/Downstream_2/VoComni_nnunet.pt}
log "start, HEAD $(git rev-parse --short HEAD); GPUs ${GPU_LIST[*]}; $NCPU CPUs, $MEM_GB GB RAM; DA workers/run $DA"

# --- G0 ---------------------------------------------------------------------------------------------------
if [[ ! -f $OUT/g0/summary.json ]]; then
  GPUS="${GPU_LIST[*]}" bash experiments/server_salt_g0.sh || { log "G0 failed, see $OUT/g0/runner.log"; exit 1; }
fi
log "G0: $(grep -o 'G0（.*' "$OUT/g0/G0.md" | head -1)"
if grep -q '"GO": false' "$OUT/g0/summary.json"; then
  log "G0 NO-GO: per §3.2 the anchors are dropped for Q2-Q4; record this in §11 and edit the configs, then rerun"
  exit 0
fi

# --- pilots: a simple GPU queue --------------------------------------------------------------------------
done_run() { [[ -f $OUT/pilot_msd08_$1_s0/decision.json ]]; }
passed() { python3 -c "import json,sys; d=json.load(open('$OUT/pilot_msd08_$1_s0/decision.json')); sys.exit(0 if d['runs']['$1_s0']['CONTINUE'] else 1)"; }
declare -A PID
run_on() {   # $1 group, $2 gpu
  log "pilot $1 on GPU $2"
  GPU=$2 bash experiments/server_salt_pilot.sh "$1" 0 > "$OUT/pilot_$1.log" 2>&1 &
  PID[$2]=$!
}
free_gpu() {
  for g in "${GPU_LIST[@]}"; do
    if [[ -z ${PID[$g]:-} ]] || ! kill -0 "${PID[$g]}" 2>/dev/null; then echo "$g"; return; fi
  done
}
QUEUE=(Q4 Q1 F1)
LATE=(Q2 Q3)
while :; do
  if (( ${#QUEUE[@]} )); then
    g=$(free_gpu)
    if [[ -n $g ]]; then
      grp=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
      done_run "$grp" || run_on "$grp" "$g"
      continue
    fi
  elif (( ${#LATE[@]} )) && done_run Q4; then
    if passed Q4; then QUEUE=("${LATE[@]}"); else log "Q4 did not pass: Q2/Q3 not trained (§4)"; fi
    LATE=()
    continue
  else
    busy=0; for g in "${GPU_LIST[@]}"; do [[ -n ${PID[$g]:-} ]] && kill -0 "${PID[$g]}" 2>/dev/null && busy=1; done
    (( busy )) || break
  fi
  sleep 120
done
for grp in Q4 Q1 F1 Q2 Q3; do
  done_run "$grp" && log "$grp: $(python3 -c "import json; print(json.load(open('$OUT/pilot_msd08_${grp}_s0/decision.json'))['runs']['${grp}_s0']['CONTINUE'])")"
done
log "all pilots finished; tables in $OUT/pilot_msd08_*_s0/DECISION.md"
