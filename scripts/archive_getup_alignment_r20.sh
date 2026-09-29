#!/usr/bin/env bash
set -euo pipefail
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ "$(git rev-parse HEAD)" == 44c3196501e2a86b34c06ebe259cf64f7cbaf9ab ]] || { echo 'Unexpected publication baseline'; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
for experiment in getup_alignment_r20_base getup_alignment_r20_height; do
  [[ -f "$training/$experiment/evaluation_50hz.json" ]] || { echo 'Evaluation unfinished'; exit 1; }
done
[[ -f "$training/getup_alignment_r20_long/results.json" ]] || { echo 'Long evaluation unfinished'; exit 1; }
[[ -f "$training/getup_range_r21_comparison/long_30s.json" ]] || { echo 'Range comparison unfinished'; exit 1; }
for file in GETUP_LOW_NOISE_ALIGNMENT_20260929.md \
    diagnostics/getup_alignment_curriculum_r20.py diagnostics/train_getup_alignment_r20.py \
    diagnostics/probe_getup_noise_r20.py diagnostics/test_getup_alignment_r20.py \
    diagnostics/validate_getup_long_r20.py scripts/launch_getup_alignment_r20.sh \
    diagnostics/probe_getup_range_r21.py diagnostics/compare_getup_range_r21.py \
    diagnostics/test_getup_comparison_r21.py \
    scripts/launch_getup_range_r21.sh scripts/validate_getup_range_r21_comparison.sh \
    scripts/validate_getup_alignment_r20_long.sh scripts/archive_getup_alignment_r20.sh; do
  cp "$runtime/$file" "$publish/$file"
  git add "$file"
done
archive=results/getup_low_noise_alignment_20260929
mkdir -p "$archive"
for experiment in getup_noise_probe_r20 getup_alignment_r20_smoke \
    getup_alignment_r20_base getup_alignment_r20_height getup_alignment_r20_long \
    getup_range_probe_r21 getup_range_r21 getup_range_r21_comparison; do
  cp -a "$training/$experiment" "$archive/$experiment"
done
for file in getup_alignment_r20_tests.log getup_alignment_r20_smoke.log \
    getup_alignment_r20_base.log getup_alignment_r20_height.log getup_r20_tests_final.log \
    getup_alignment_r20_long.log getup_range_r21.log getup_range_r21_comparison.log \
    getup_r20_r21_tests_final.log getup_range_r21_comparison_guard_missing_audit.log; do
  cp "$training/$file" "$archive/$file"
done
git add -f "$archive"
git diff --cached --check
git commit -m 'Train low-noise recovery reward and range ablations with independent long-horizon tests'
git bundle create /data/shijinsheng/open_duck/tmp/getup_alignment_r20_20260929.bundle 44c3196..main
git status --short
git log -1 --oneline
