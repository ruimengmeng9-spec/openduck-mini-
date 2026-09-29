#!/usr/bin/env bash
set -euo pipefail
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ "$(git rev-parse HEAD)" == 066fe45b310c24d1d720f13dfde0db8d36ddec57 ]] || { echo 'Unexpected publication baseline'; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
for file in GETUP_NATIVE_PPO_STAGE1_20260929.md \
    diagnostics/getup_native_curriculum.py diagnostics/train_getup_native_ppo.py \
    diagnostics/train_getup_native_ppo_v2.py diagnostics/eval_getup_native_ppo.py \
    diagnostics/eval_getup_native_ppo_v2.py diagnostics/test_getup_native_ppo.py \
    diagnostics/test_getup_native_export.py diagnostics/probe_getup_stage1_baselines.py \
    diagnostics/audit_getup_native_checkpoint.py diagnostics/select_getup_stage1_checkpoint.py \
    diagnostics/summarize_getup_native_ppo.py scripts/launch_getup_native_ppo_r10.sh \
    scripts/launch_getup_native_ppo_r11.sh scripts/validate_getup_native_ppo_r11.sh \
    scripts/archive_getup_native_ppo_20260929.sh; do
  cp "$runtime/$file" "$publish/$file"
  git add "$file"
done
archive=results/getup_native_ppo_20260929
mkdir -p "$archive"
for experiment in getup_curriculum_precheck_r10_verified getup_native_ppo_r10_smoke \
    getup_native_ppo_r10_stage1 getup_native_ppo_r11_smoke getup_native_ppo_r11_stage1; do
  [[ -d "$training/$experiment" ]] || { echo "Missing experiment: $experiment"; exit 1; }
  cp -a "$training/$experiment" "$archive/$experiment"
done
for file in getup_curriculum_precheck_r10_verified.log getup_native_ppo_r10_tests.log \
    getup_native_ppo_r10_stage1.log getup_native_ppo_r11_stage1.log \
    getup_native_ppo_r11_tests.log getup_native_ppo_r11_tests_complete.log \
    getup_native_ppo_r11_validation.log getup_native_ppo_r10_r11_summary.json; do
  cp "$training/$file" "$archive/$file"
done
[[ -f "$archive/getup_native_ppo_r11_stage1/selected_candidate/checkpoint_audit.json" ]] || { echo 'Final validation not finished'; exit 1; }
git add -f "$archive"
git diff --cached --check
git commit -m 'Train independent native get-up PPO stages and preserve matched baseline evaluations'
git bundle create /data/shijinsheng/open_duck/tmp/getup_native_ppo_20260929.bundle 066fe45..main
git status --short
git log -1 --oneline
