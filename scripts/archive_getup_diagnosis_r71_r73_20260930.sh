#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=965ed8c3ef609ab650aba37d0d65dc21758b822b
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
for file in audit_getup_tracking_r71.py probe_getup_terminal_timing_r72.py test_getup_terminal_timing_r72.py train_getup_prefix_pose_r73.py test_getup_prefix_pose_r73.py; do
  cp "$source_repo/diagnostics/$file" "diagnostics/$file"
  git add "diagnostics/$file"
done
cp "$source_repo/GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md" .
cp "$source_repo/scripts/archive_getup_diagnosis_r71_r73_20260930.sh" scripts/
git add GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md scripts/archive_getup_diagnosis_r71_r73_20260930.sh
for experiment in getup_fullfall_tracking_r67_left_20260930 getup_shortpath_feedback_r70_left_20260930 getup_tracking_audit_r71_20260930 getup_terminal_timing_r72_20260930; do
  test -f "$root/outputs/$experiment/results.json"
  mkdir -p "results/$experiment"
  for file in results.json contract.json search_history.json checkpoint.npz selected.npz reference.npz; do
    if test -f "$root/outputs/$experiment/$file"; then
      cp "$root/outputs/$experiment/$file" "results/$experiment/$file"
      git add -f "results/$experiment/$file"
    fi
  done
  cp "$root/outputs/$experiment.log" "results/$experiment/console.log"
  git add -f "results/$experiment/console.log"
done
for seed in nominal 769001 773000 773005; do
  experiment=getup_tracking_audit_r71_20260930
  cp "$root/outputs/$experiment/trace_$seed.npz" "results/$experiment/"
  git add -f "results/$experiment/trace_$seed.npz"
done
# Only copy the immutable launch contract from the active R73 run. Do not
# claim a running checkpoint is final or snapshot mutable logs mid-write.
experiment=getup_prefix_pose_r73_left_20260930
test -f "$root/outputs/$experiment/contract.json"
mkdir -p "results/$experiment"
cp "$root/outputs/$experiment/contract.json" "results/$experiment/"
git add -f "results/$experiment/contract.json"
git diff --cached --stat
git commit -m "Audit failed whole-fall get-up and train pre-wait contact pose"
git bundle create "$root/tmp/getup_diagnosis_r71_r73_20260930.bundle" HEAD "^$base"
git rev-parse HEAD
