#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=39ccc962158ffbeafb9acd0f7764a29298782753
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
for file in probe_getup_wait_compression_r69.py train_getup_shortpath_feedback_r70.py test_getup_wait_compression_r69.py; do
  cp "$source_repo/diagnostics/$file" "diagnostics/$file"
  git add "diagnostics/$file"
done
cp "$source_repo/GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md" .
cp "$source_repo/scripts/archive_getup_shortpath_r69_r70_20260930.sh" scripts/
git add GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md scripts/archive_getup_shortpath_r69_r70_20260930.sh
for experiment in getup_fullfall_tracking_r67_left_20260930 getup_fullfall_tracking_r67_pin_r69_20260930 getup_wait_compression_r69_20260930 getup_shortpath_feedback_r70_left_20260930; do
  test -d "$root/outputs/$experiment"
  mkdir -p "results/$experiment"
  for file in results.json contract.json search_history.json checkpoint.npz selected_prefix.npz; do
    if test -f "$root/outputs/$experiment/$file"; then
      cp "$root/outputs/$experiment/$file" "results/$experiment/$file"
      git add -f "results/$experiment/$file"
    fi
  done
  if test -f "$root/outputs/$experiment.log"; then
    cp "$root/outputs/$experiment.log" "results/$experiment/console.log"
    git add -f "results/$experiment/console.log"
  fi
done
git diff --cached --stat
git commit -m "Shorten audited get-up wait and expand whole-fall feedback training"
git bundle create "$root/tmp/getup_shortpath_r69_r70_20260930.bundle" HEAD "^$base"
git rev-parse HEAD
