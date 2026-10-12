#!/usr/bin/env bash
# Spread the last 20 model x case exports (LiTS_117, LiTS_130) over all three GPUs,
# then summarize and resume the frozen runner at step 5.
set -u
OUT=/root/autodl-tmp/nnUNet_data/nnUNet_results_v2/_analysis/test_protocol
W=/root/autodl-tmp/Downstream_2_test_protocol
PY=/root/autodl-tmp/envs/mamba/bin/python
DATA=/root/autodl-tmp/nnUNet_data
LOG=$OUT/finish3.log
for p in $(ps -eo pid,args | awk "\$2==\"bash\" && \$3 ~ /test_protocol_finish2.sh\$/ {print \$1}"); do kill $p; done
for p in $(pgrep -f "missed_lesion_probability.py run"); do pkill -P $p; kill $p; done
sleep 10
echo "[$(date "+%F %T")] stopped 2-process run; json $(find $OUT/curves/cases -name "*.json" | wc -l)" >> $LOG
$PY - <<EOF >> $LOG
import json, glob, os
bad=[]
for f in glob.glob("$OUT/curves/cases/*/*.json"):
    try: json.load(open(f))
    except Exception: bad.append(f); os.remove(f)
print("removed unreadable json:", bad)
EOF
cd $W
run() {  # gpu case workers runs...
  g=$1; c=$2; w=$3; shift 3; echo $c > $OUT/last_$c.txt
  CUDA_VISIBLE_DEVICES=$g $PY -u experiments/analysis/missed_lesion_probability.py run --runs "$@" \
    --image-dir $DATA/nnUNet_raw/Dataset102_LiTS/imagesTs --label-dir $DATA/nnUNet_raw/Dataset101_LiTS/labelsTr \
    --cases-file $OUT/last_$c.txt --device cuda --workers $w --offload-models --out $OUT/curves >> $OUT/curves_final_gpu$g.log 2>&1
}
run 0 LiTS_117 3 A1:1,2 A11:0,1,2 &
run 1 LiTS_117 3 A13a:0,1,2 A13b:0,1,2 &
run 2 LiTS_130 4 A11:0,1,2 A13a:0,1,2 A13b:0,1,2 &
wait
n=$(find $OUT/curves/cases -name "*.json" | wc -l)
echo "[$(date "+%F %T")] remaining exports done: $n / 405 json" >> $LOG
if [[ $n == 405 ]]; then
  $PY experiments/analysis/missed_lesion_probability.py summarize --out $OUT/curves --results-dir results --eval-subdir eval_test > $OUT/curves_summary.log 2>&1
  echo "[$(date "+%F %T")] summarize exit $?" >> $LOG
  STEP=5 experiments/server_test_protocol.sh >> $LOG 2>&1
  echo "FINISH_EXIT $?" >> $LOG
else
  echo "FINISH_EXIT incomplete" >> $LOG
fi
