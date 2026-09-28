#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
cd "$root/github/openduck-mini-"
mkdir -p models/backward_target_smoothing_r18
cp -n "$root/training/backward_target_smoothing_r18/tau0.01.json" models/backward_target_smoothing_r18/
for result in actuator_audit_r17_r11 actuator_audit_r17_r15 target_smoothing_r18_tau0.01_60s target_smoothing_r18_tau0.02_60s target_smoothing_r18_tau0.04_60s target_smoothing_r18_tau0.08_60s target_smoothing_r18_tau0.01_broad_60s target_smoothing_r18_tau0.01_stand_60s backward_sequence_r19_reset_tau0.01 backward_sequence_r19_regression backward_stop_blend_r21_direct backward_stop_blend_r21_aligned; do
  if test -f "$root/outputs/$result/results.json"; then cp -n "$root/outputs/$result/results.json" "results/${result}_20260928.json"; fi
  if test -f "$root/outputs/$result/actuator_audit.json"; then cp -n "$root/outputs/$result/actuator_audit.json" "results/${result}_actuators_20260928.json"; fi
done
if test -f "$root/outputs/target_smoothing_r18_tau0.01_60s/actuator_audit_seed117.json"; then cp -n "$root/outputs/target_smoothing_r18_tau0.01_60s/actuator_audit_seed117.json" results/target_smoothing_r18_tau0.01_seed117_actuators_20260928.json; fi
git diff --check
