#!/usr/bin/env bash
# Exact whitelist; preserve previously archived artifacts and unrelated changes.
set -euo pipefail
root=/data/shijinsheng/open_duck
cd "$root/github/openduck-mini-"
for stage in backward_phase_template_r10 backward_contact_balance_r11 backward_contact_heading_r12; do
  mkdir -p "models/$stage"
  cp -n "$root/training/$stage/best.json" "$root/training/$stage/search.json" "models/$stage/"
  for export in export_probe export_final; do
    cp -rn "$root/training/$stage/$export" "models/$stage/"
  done
  for config in snapshot_probe.json template_base.json; do
    if test -f "$root/training/$stage/$config"; then cp -n "$root/training/$stage/$config" "models/$stage/"; fi
  done
  cp -n "$root/tmp/$stage.log" "models/$stage/train.log"
done
mkdir -p models/backward_path_r13 models/backward_capture_r14
cp -n "$root/training/backward_path_r13/k0.3.json" "$root/training/backward_path_r13/k0.6.json" "$root/training/backward_path_r13/k1.0.json" models/backward_path_r13/
cp -n "$root/training/backward_capture_r14/capture_base.json" "$root/training/backward_capture_r14/capture_diagnosis.json" models/backward_capture_r14/
for result in phase_template_r10_probe_heldout_60s phase_template_r10_final_heldout_60s contact_balance_r11_probe_heldout_60s contact_balance_r11_parallel_regression contact_balance_r11_final_heldout_60s contact_heading_r12_probe_heldout_60s contact_heading_r12_final_heldout_60s path_r13_k0.3_60s path_r13_k0.6_60s path_r13_k1.0_60s contact_balance_r11_capture_trace_60s; do
  if test -f "$root/outputs/$result/results.json"; then
    cp -n "$root/outputs/$result/results.json" "results/${result}_20260928.json"
  fi
done
git diff --check
