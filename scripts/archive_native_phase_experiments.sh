#!/usr/bin/env bash
# Copy only this task's completed artifacts into the publication clone.
set -euo pipefail
ROOT=/data/shijinsheng/open_duck
cd "$ROOT/github/openduck-mini-"
for STAGE in backward_phase_r5 backward_phase_feedback_r6 backward_phase_feedback_r7 backward_phase_balance_r8 backward_phase_balance_r9_long; do
  mkdir -p "models/$STAGE"
  cp -n "$ROOT/training/$STAGE/best.json" "$ROOT/training/$STAGE/search.json" "models/$STAGE/"
  for EXPORT in export_probe export_balanced export_final; do
    if test -d "$ROOT/training/$STAGE/$EXPORT"; then
      cp -rn "$ROOT/training/$STAGE/$EXPORT" "models/$STAGE/"
    fi
  done
  cp -n "$ROOT/tmp/$STAGE.log" "models/$STAGE/train.log"
done
for RESULT in phase_r5_early_30s phase_feedback_r6_probe_heldout_30s phase_feedback_r6_balanced_gate_30s phase_feedback_r6_balanced_heldout_30s phase_feedback_r6_balanced_heldout_60s phase_feedback_r7_probe_heldout_30s phase_feedback_r7_final_heldout_60s phase_r2_zero_matched_60s phase_balance_r8_final_heldout_60s phase_balance_r9_probe_heldout_60s phase_balance_r9_final_heldout_60s; do
  cp -n "$ROOT/outputs/$RESULT/results.json" "results/${RESULT}_20260928.json"
  if test -f "$ROOT/outputs/$RESULT/failure_kinematics.json"; then
    cp -n "$ROOT/outputs/$RESULT/failure_kinematics.json" "results/${RESULT}_kinematics_20260928.json"
  fi
done
git diff --check
git status --short
