#!/bin/bash
cd /root/autodl-tmp/nnUNet_data || exit 1

echo "=== preprocessed nnUNetPlans_3d_fullres ==="
ls nnUNet_preprocessed/Dataset102_LiTS/nnUNetPlans_3d_fullres/ | head -3
echo "count: $(ls nnUNet_preprocessed/Dataset102_LiTS/nnUNetPlans_3d_fullres/ | wc -l)"

echo
echo "=== ALL RUNS (config header from latest training log) ==="
for d in nnUNet_results*/Dataset102_LiTS/*/fold_0 lits_100ep_*/Dataset102_LiTS/*/fold_0; do
  [ -d "$d" ] || continue
  log=$(ls -t "$d"/training_log_*.txt 2>/dev/null | head -1)
  [ -n "$log" ] || { echo "--- $d  (no log)"; continue; }
  echo "--- $d"
  echo "    log: $(basename "$log")"
  grep -E "PFCR: |PC boundary loss:|Small-Tumor Head:|PCSF: |Total epochs:|Uncertainty|Small-Head Loss" "$log" | head -6 | sed 's/^/    /'
done

echo
echo "=== fold_0 checkpoint inventory ==="
for d in nnUNet_results*/Dataset102_LiTS/*/fold_0; do
  [ -d "$d" ] || continue
  cks=$(ls "$d"/*.pth 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
  bak=$(ls "$d"/_old_arch_backup/*.pth 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
  echo "--- $d"
  echo "    main: ${cks:-none}"
  echo "    bak : ${bak:-none}"
done
