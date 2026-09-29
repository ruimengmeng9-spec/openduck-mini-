#!/usr/bin/env bash
set -euo pipefail
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ "$(git rev-parse HEAD)" == a847868d7acd8667b2b2915d2728ef27ee3cef51 ]] || { echo 'Unexpected publication baseline'; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
[[ -f "$training/getup_macro_ppo_r15/evaluation_50hz.json" ]] || { echo 'Final evaluation not finished'; exit 1; }
[[ -f "$training/getup_residual_ppo_r18/evaluation_50hz.json" ]] || { echo 'Residual evaluation not finished'; exit 1; }
[[ -f "$training/getup_residual_ppo_r18/selected_candidate/evaluation_50hz.json" ]] || { echo 'Selected residual evaluation not finished'; exit 1; }
[[ -f "$training/getup_residual_ppo_r18/evaluation_broad_100.json" ]] || { echo 'Broad residual evaluation not finished'; exit 1; }
for file in GETUP_MACRO_RECOVERY_20260929.md \
    diagnostics/getup_teacher_probe_r12.py diagnostics/getup_balance_search_r13.py \
    diagnostics/getup_smooth_transition_r14.py diagnostics/test_getup_balance_r13.py \
    diagnostics/getup_macro_curriculum_r15.py diagnostics/train_getup_macro_ppo_r15.py \
    diagnostics/eval_getup_macro_r15.py diagnostics/test_getup_macro_r15.py \
    diagnostics/probe_getup_solver_r16.py diagnostics/probe_getup_sole_r17.py \
    diagnostics/train_getup_residual_r18.py diagnostics/audit_getup_residual_r18.py \
    diagnostics/test_getup_residual_r18.py diagnostics/select_getup_residual_r18.py \
    diagnostics/probe_getup_goal_gates_r19.py scripts/launch_getup_residual_r18.sh \
    scripts/validate_getup_residual_r18.sh \
    scripts/launch_getup_macro_r15.sh scripts/archive_getup_macro_20260929.sh; do
  cp "$runtime/$file" "$publish/$file"
  git add "$file"
done
archive=results/getup_macro_recovery_20260929
mkdir -p "$archive"
for experiment in getup_teacher_probe_r12 getup_balance_search_r13 \
    getup_smooth_transition_r14 getup_macro_ppo_r15_smoke getup_macro_ppo_r15 \
    getup_solver_probe_r16 getup_sole_probe_r17 getup_residual_ppo_r18_smoke \
    getup_residual_ppo_r18 getup_goal_gates_r19; do
  [[ -d "$training/$experiment" ]] || { echo "Missing experiment: $experiment"; exit 1; }
  cp -a "$training/$experiment" "$archive/$experiment"
done
for file in getup_balance_search_r13.log getup_smooth_transition_r14.log \
    getup_macro_ppo_r15.log getup_r12_r15_tests.log getup_r12_r17_tests_final.log \
    getup_residual_ppo_r18_smoke.log getup_residual_ppo_r18.log getup_r12_r18_tests_final.log; do
  cp "$training/$file" "$archive/$file"
done
git add -f "$archive"
git diff --cached --check
git commit -m 'Test recovery teachers and train temporally coherent native get-up PPO'
git bundle create /data/shijinsheng/open_duck/tmp/getup_macro_recovery_20260929.bundle a847868..main
git status --short
git log -1 --oneline
