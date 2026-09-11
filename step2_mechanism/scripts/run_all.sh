#!/bin/bash
# Step 2 机制级验证 — 一键复现
#
# 用法：
#   cd /root/autodl-tmp/Downstream_2/step2_mechanism
#   bash scripts/run_all.sh            # 全部
#   bash scripts/run_all.sh 4.2        # 只跑某一步（4.1 / 4.1b / 4.2 / 4.3 / 4.4 / 4.4b）
#
# 全部运行约 45–60 分钟，主要耗时在 4.2 的 104 例 CT/GT 重采样。
# 所有步骤只读，不写 nnUNet_data 下任何文件。

set -u
PY=/root/autodl-tmp/envs/mamba/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p out figs

STEPS="${1:-all}"

run() {
  local name="$1"; shift
  if [ "$STEPS" != "all" ] && [ "$STEPS" != "$name" ]; then return 0; fi
  echo ""
  echo "#############################################################"
  echo "#  $name"
  echo "#############################################################"
  PYTHONUNBUFFERED=1 "$PY" -u "$@" 2>&1 | tee "out/${name}.log"
  echo "[done] $name"
}

run 4.1    scripts/s41_synthetic.py
run 4.1b   scripts/s41b_profile.py
run 4.2    scripts/s42_real_pc.py
run 4.2b   scripts/s42b_stats.py
run 4.3    scripts/s43_filters.py
run 4.4    scripts/s44_routes.py
run 4.4b   scripts/s44b_pc_cap.py

echo ""
echo "============================================================="
echo "全部完成。产物："
ls -l out/*.json figs/*.png 2>/dev/null
echo "============================================================="
