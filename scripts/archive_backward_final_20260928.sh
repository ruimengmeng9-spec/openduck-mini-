#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
cd "$root/github/openduck-mini-"
mkdir -p models/backward_filtered_heading_r20
cp -n "$root/training/backward_filtered_heading_r20/best.json" "$root/training/backward_filtered_heading_r20/search.json" "$root/training/backward_filtered_heading_r20/run.log" models/backward_filtered_heading_r20/
cp -rn "$root/training/backward_filtered_heading_r20/export_final" models/backward_filtered_heading_r20/
if test -d "$root/training/backward_sim_candidate_r22"; then cp -rn "$root/training/backward_sim_candidate_r22" models/; fi
for result in filtered_heading_r20_final_heldout_60s filtered_heading_r20_manual_training_regression filtered_combined_r22_120s filtered_combined_r22_sequence backward_stop_blend_r21_direct_broad; do
  test -f "$root/outputs/$result/results.json"
  cp -n "$root/outputs/$result/results.json" "results/${result}_20260928.json"
done
if test -f "$root/outputs/filtered_heading_r20_training_regression/results.json"; then cp -n "$root/outputs/filtered_heading_r20_training_regression/results.json" results/filtered_heading_r20_training_regression_20260928.json; fi
cp -n "$root/outputs/filtered_combined_r22_120s/contact_audit.json" results/filtered_combined_r22_contact_audit_20260928.json
git diff --check
