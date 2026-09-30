#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=0c0972c
cd "$publication"
test "$(git rev-parse --short=7 HEAD)" = "$base"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test -z "$(git status --porcelain)"
for file in train_getup_wait_pose_r74.py test_getup_wait_pose_r74.py audit_getup_wait_tilt_r74.py; do
  test ! -e "diagnostics/$file"
  cp "$source_repo/diagnostics/$file" "diagnostics/$file"
  git add "diagnostics/$file"
done
for file in GETUP_WAIT_POSE_R74_20260930.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md; do
  cp "$source_repo/$file" "$file"
  git add "$file"
done
cp "$source_repo/scripts/archive_getup_wait_pose_r74_20260930.sh" scripts/
git add scripts/archive_getup_wait_pose_r74_20260930.sh
experiment=getup_prefix_pose_r73_left_20260930
test -f "$root/outputs/$experiment/results.json"
mkdir -p "results/$experiment"
for file in results.json contract.json search_history.json checkpoint.npz; do
  cp "$root/outputs/$experiment/$file" "results/$experiment/$file"
  git add -f "results/$experiment/$file"
done
cp "$root/outputs/$experiment.log" "results/$experiment/console.log"
git add -f "results/$experiment/console.log"
# Full independent candidate traces (including failures), not only successes.
for file in "$root/outputs/$experiment"/qualification_*.npz; do
  cp "$file" "results/$experiment/"
  git add -f "results/$experiment/$(basename "$file")"
done
experiment=getup_wait_tilt_audit_r74_20260930
mkdir -p "results/$experiment"
cp "$root/outputs/$experiment/results.json" "results/$experiment/"
git add -f "results/$experiment/results.json"
# R74 is running. Publish the immutable contract, not a mutable checkpoint.
experiment=getup_wait_pose_r74_left_20260930
test -f "$root/outputs/$experiment/contract.json"
mkdir -p "results/$experiment"
cp "$root/outputs/$experiment/contract.json" "results/$experiment/"
git add -f "results/$experiment/contract.json"
git diff --cached --stat
git commit -m "Retain negative R73 qualification and launch wait-pose recovery training R74"
git bundle create "$root/tmp/getup_wait_pose_r74_20260930.bundle" HEAD "^$base"
git rev-parse HEAD
git push origin HEAD:main
