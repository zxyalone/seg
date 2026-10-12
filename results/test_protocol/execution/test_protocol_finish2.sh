#!/usr/bin/env bash
# Shard 2 of step 4 ended without a traceback at 00:08 (after LiTS_109); run its two
# remaining cases on separate GPUs with 2 workers, then summarize and resume at step 5.
set -u
OUT=/root/autodl-tmp/nnUNet_data/nnUNet_results_v2/_analysis/test_protocol
W=/root/autodl-tmp/Downstream_2_test_protocol
PY=/root/autodl-tmp/envs/mamba/bin/python
DATA=/root/autodl-tmp/nnUNet_data
LOG=$OUT/finish2.log
cd $W
for pair in 0:LiTS_117 2:LiTS_130; do
  g=${pair%%:*}; c=${pair#*:}; echo $c > $OUT/last_$c.txt
  CUDA_VISIBLE_DEVICES=$g $PY -u experiments/analysis/missed_lesion_probability.py run --runs A0:0,1,2 A1:0,1,2 A11:0,1,2 A13a:0,1,2 A13b:0,1,2 \
    --image-dir $DATA/nnUNet_raw/Dataset102_LiTS/imagesTs --label-dir $DATA/nnUNet_raw/Dataset101_LiTS/labelsTr \
    --cases-file $OUT/last_$c.txt --device cuda --workers 2 --offload-models --out $OUT/curves >> $OUT/curves_last_$c.log 2>&1 &
done
wait
n=$(find $OUT/curves/cases -name "*.json" | wc -l)
echo "[$(date "+%F %T")] remaining cases done: $n / 405 json" >> $LOG
if [[ $n == 405 ]]; then
  $PY experiments/analysis/missed_lesion_probability.py summarize --out $OUT/curves --results-dir results --eval-subdir eval_test > $OUT/curves_summary.log 2>&1
  echo "[$(date "+%F %T")] summarize exit $?" >> $LOG
  STEP=5 experiments/server_test_protocol.sh >> $LOG 2>&1
  echo "FINISH_EXIT $?" >> $LOG
else
  echo "FINISH_EXIT incomplete" >> $LOG
fi
