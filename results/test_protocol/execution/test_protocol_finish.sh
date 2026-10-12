#!/usr/bin/env bash
# Restart shard 1 of step 4 (process ended without a traceback while on LiTS_092)
# with 2 workers, wait for all shards, then summarize and resume the frozen runner at step 5.
set -u
OUT=/root/autodl-tmp/nnUNet_data/nnUNet_results_v2/_analysis/test_protocol
W=/root/autodl-tmp/Downstream_2_test_protocol
PY=/root/autodl-tmp/envs/mamba/bin/python
DATA=/root/autodl-tmp/nnUNet_data
LOG=$OUT/finish.log
for p in $(ps -eo pid,args | awk "\$2==\"bash\" && \$3 ~ /test_protocol_speedup.sh\$/ {print \$1}"); do kill $p; done
echo "[$(date "+%F %T")] watcher stopped; shard1 restart with 2 workers; json so far $(find $OUT/curves/cases -name "*.json" | wc -l)" >> $LOG
$PY - <<EOF >> $LOG
import json, glob, os
bad=[]
for f in glob.glob("$OUT/curves/cases/*/*.json"):
    try: json.load(open(f))
    except Exception: bad.append(f); os.remove(f)
print("removed unreadable json:", bad)
EOF
cd $W
CUDA_VISIBLE_DEVICES=1 $PY -u experiments/analysis/missed_lesion_probability.py run --runs A0:0,1,2 A1:0,1,2 A11:0,1,2 A13a:0,1,2 A13b:0,1,2 \
  --image-dir $DATA/nnUNet_raw/Dataset102_LiTS/imagesTs --label-dir $DATA/nnUNet_raw/Dataset101_LiTS/labelsTr \
  --cases-file experiments/eval/test_cases.txt --shard 1/3 --device cuda --workers 2 --offload-models \
  --out $OUT/curves >> $OUT/curves_shard1.log 2>&1
echo "[$(date "+%F %T")] shard1 restart exit $?" >> $LOG
while pgrep -f "missed_lesion_probability.py run" >/dev/null; do sleep 30; done
n=$(find $OUT/curves/cases -name "*.json" | wc -l)
echo "[$(date "+%F %T")] all shards done: $n / 405 json" >> $LOG
if [[ $n == 405 ]]; then
  $PY experiments/analysis/missed_lesion_probability.py summarize --out $OUT/curves --results-dir results --eval-subdir eval_test > $OUT/curves_summary.log 2>&1
  echo "[$(date "+%F %T")] summarize exit $?" >> $LOG
  STEP=5 experiments/server_test_protocol.sh >> $LOG 2>&1
  echo "FINISH_EXIT $?" >> $LOG
else
  echo "FINISH_EXIT incomplete" >> $LOG
fi
