#!/usr/bin/env bash
# Last 6 exports (LiTS_117: 219 Mvox, 75 lesions) with one worker per GPU, then
# summarize and resume the frozen runner at step 5.
set -u
OUT=/root/autodl-tmp/nnUNet_data/nnUNet_results_v2/_analysis/test_protocol
W=/root/autodl-tmp/Downstream_2_test_protocol
PY=/root/autodl-tmp/envs/mamba/bin/python
DATA=/root/autodl-tmp/nnUNet_data
LOG=$OUT/finish4.log
$PY - <<EOF >> $LOG
import json, glob, os
bad=[]
for f in glob.glob("$OUT/curves/cases/*/*.json"):
    try: json.load(open(f))
    except Exception: bad.append(f); os.remove(f)
print("removed unreadable json:", bad)
EOF
cd $W; echo LiTS_117 > $OUT/last_LiTS_117.txt
run() {  # gpu runs...
  g=$1; shift
  CUDA_VISIBLE_DEVICES=$g $PY -u experiments/analysis/missed_lesion_probability.py run --runs "$@" \
    --image-dir $DATA/nnUNet_raw/Dataset102_LiTS/imagesTs --label-dir $DATA/nnUNet_raw/Dataset101_LiTS/labelsTr \
    --cases-file $OUT/last_LiTS_117.txt --device cuda --workers 1 --offload-models --out $OUT/curves >> $OUT/curves_117_gpu$g.log 2>&1
}
run 0 A13a:0,1 & run 1 A13a:2 A13b:0 & run 2 A13b:1,2 &
wait
n=$(find $OUT/curves/cases -name "*.json" | wc -l)
echo "[$(date "+%F %T")] last exports done: $n / 405 json" >> $LOG
if [[ $n == 405 ]]; then
  $PY experiments/analysis/missed_lesion_probability.py summarize --out $OUT/curves --results-dir results --eval-subdir eval_test > $OUT/curves_summary.log 2>&1
  echo "[$(date "+%F %T")] summarize exit $?" >> $LOG
  STEP=5 experiments/server_test_protocol.sh >> $LOG 2>&1
  echo "FINISH_EXIT $?" >> $LOG
else
  echo "FINISH_EXIT incomplete" >> $LOG
fi
