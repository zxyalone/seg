#!/usr/bin/env bash
# Re-run step 4 (test threshold curves) with 3 CPU workers per GPU instead of 2,
# using the same frozen code; then resume the frozen runner from step 5.
set -u
OUT=/root/autodl-tmp/nnUNet_data/nnUNet_results_v2/_analysis/test_protocol
W=/root/autodl-tmp/Downstream_2_test_protocol
PY=/root/autodl-tmp/envs/mamba/bin/python
DATA=/root/autodl-tmp/nnUNet_data
LOG=$OUT/speedup.log
echo "[$(date "+%F %T")] waiting for step 4" >> $LOG
until ls $OUT/curves_shard0.log >/dev/null 2>&1; do sleep 10; done
sleep 15
for p in $(pgrep -f "server_test_protocol.sh"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "missed_lesion_probability.py run"); do pkill -P $p 2>/dev/null; kill $p 2>/dev/null; done
sleep 10
for p in $(pgrep -f "multiprocessing.spawn"); do kill $p 2>/dev/null; done
echo "[$(date "+%F %T")] stopped runner step 4 (2 workers); json so far: $(find $OUT/curves/cases -name "*.json" 2>/dev/null | wc -l)" >> $LOG
$PY - <<EOF >> $LOG
import json, glob, os
bad=[]
for f in glob.glob("$OUT/curves/cases/*/*.json"):
    try: json.load(open(f))
    except Exception: bad.append(f); os.remove(f)
print("removed unreadable json:", bad)
EOF
cd $W
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g $PY -u experiments/analysis/missed_lesion_probability.py run --runs A0:0,1,2 A1:0,1,2 A11:0,1,2 A13a:0,1,2 A13b:0,1,2 \
    --image-dir $DATA/nnUNet_raw/Dataset102_LiTS/imagesTs --label-dir $DATA/nnUNet_raw/Dataset101_LiTS/labelsTr \
    --cases-file experiments/eval/test_cases.txt --shard $g/3 --device cuda --workers 3 --offload-models \
    --out $OUT/curves >> $OUT/curves_shard$g.log 2>&1 &
done
wait
n=$(find $OUT/curves/cases -name "*.json" | wc -l)
echo "[$(date "+%F %T")] step 4 rerun finished: $n / 405 json" >> $LOG
if [[ $n == 405 ]]; then
  $PY experiments/analysis/missed_lesion_probability.py summarize --out $OUT/curves --results-dir results --eval-subdir eval_test > $OUT/curves_summary.log 2>&1
  echo "[$(date "+%F %T")] summarize exit $?" >> $LOG
  STEP=5 experiments/server_test_protocol.sh >> $OUT/speedup.log 2>&1
  echo "SPEEDUP_EXIT $?" >> $LOG
else
  echo "SPEEDUP_EXIT incomplete" >> $LOG
fi
