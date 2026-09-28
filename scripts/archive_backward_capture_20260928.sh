#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
cd "$root/github/openduck-mini-"
for stage in backward_capture_r14/search backward_capture_native_r15; do
  test -f "$root/training/$stage/export_final/controller_contract.json"
  mkdir -p "models/$stage"
  cp -n "$root/training/$stage/best.json" "$root/training/$stage/search.json" "models/$stage/"
  for export in export_zero export_probe export_final; do
    if test -d "$root/training/$stage/$export"; then cp -rn "$root/training/$stage/$export" "models/$stage/"; fi
  done
  for config in zero_base.json snapshot_probe.json; do
    if test -f "$root/training/$stage/$config"; then cp -n "$root/training/$stage/$config" "models/$stage/"; fi
  done
done
cp -n "$root/training/backward_capture_r14/run.log" models/backward_capture_r14/train.log
cp -n "$root/training/backward_capture_native_r15/run.log" models/backward_capture_native_r15/train.log
for result in path_r13_k0.6_broad_60s capture_r14_final_heldout_60s capture_r15_zero_regression capture_r15_probe_heldout_60s capture_r15_final_heldout_60s stand_transition_r16_r11_60s stand_transition_r16_r15_60s; do
  if test -f "$root/outputs/$result/results.json"; then cp -n "$root/outputs/$result/results.json" "results/${result}_20260928.json"; fi
done
git diff --check
