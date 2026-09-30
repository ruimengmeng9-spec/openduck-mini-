#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=1e80050a35b2fed15833af016bcea8e0cc493b8b
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test "$(git rev-parse origin/main)" = "$base"
test -z "$(git status --porcelain)"
# This record describes a prepared, untested probe, not an active run.
test ! -e "$root/outputs/getup_prefix_timing_probe_r75_20261001"
for file in search_getup_prefix_timing_r75.py test_getup_prefix_timing_r75.py; do
  test ! -e "diagnostics/$file"
  cp "$source_repo/diagnostics/$file" "diagnostics/$file"
  git add "diagnostics/$file"
done
for file in GETUP_EARLY_TIMING_PREPARATION_R75_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md; do
  cp "$source_repo/$file" "$file"
  git add "$file"
done
experiment=getup_wait_pose_r74_left_20260930
test -f "$root/outputs/$experiment/results.json"
for file in results.json contract.json checkpoint.npz search_history.json training_replay.json; do
  cp "$root/outputs/$experiment/$file" "results/$experiment/$file"
  git add -f "results/$experiment/$file"
done
cp "$root/outputs/$experiment.log" "results/$experiment/console.log"
git add -f "results/$experiment/console.log"
# Paired nominal, two failures, two regressions and two rescues. All remaining
# raw paired trials stay in the immutable complete server experiment directory.
for seed in None 777000 777001 777003 777011 777015 777029; do
  for branch in baseline candidate; do
    file="${branch}_${seed}.npz"
    cp "$root/outputs/$experiment/$file" "results/$experiment/$file"
    git add -f "results/$experiment/$file"
  done
done
cp "$source_repo/scripts/archive_getup_r74_completed_r75_prepared_20261001.sh" scripts/
git add scripts/archive_getup_r74_completed_r75_prepared_20261001.sh
git diff --cached --stat
git commit -m "Archive R74 paired qualification and prepare isolated early-timing probe R75"
git bundle create "$root/tmp/getup_r74_completed_r75_prepared_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
