#!/usr/bin/env bash
set -euo pipefail
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ "$(git rev-parse HEAD)" == "$(git rev-parse f7bf6dc)" ]] || { echo 'Unexpected publication baseline'; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
for file in GETUP_LOADED_SUPPORT_20260928.md \
    diagnostics/audit_getup_crouch.py diagnostics/audit_getup_load_support.py \
    diagnostics/getup_anatomical_search.py diagnostics/getup_load_search.py diagnostics/getup_ik_lift.py \
    diagnostics/test_getup_load_support.py diagnostics/test_getup_ik_lift.py diagnostics/summarize_getup_load.py \
    scripts/launch_getup_anatomical_r7.sh scripts/launch_getup_load_r8.sh \
    scripts/archive_getup_loaded_support_20260928.sh; do
    cp "$runtime/$file" "$publish/$file"
    git add "$file"
done
archive=results/getup_loaded_support_20260928
mkdir -p "$archive"
for experiment in getup_crouch_audit_r7 getup_load_audit_r7 getup_anatomical_r7_prone \
    getup_load_r8_supine getup_load_r8_prone getup_ik_r9_prone_verified getup_ik_r9_supine; do
    mkdir -p "$archive/$experiment"
    for file in results.json search_progress.json best_reference.npz best_trajectory.npz; do
        if [[ -f "$training/$experiment/$file" ]]; then cp "$training/$experiment/$file" "$archive/$experiment/$file"; fi
    done
done
for file in getup_load_summary_r7_r9.json getup_anatomical_r7.log getup_load_r8.log \
    getup_ik_r9_prone_verified.log getup_ik_r9_supine.log getup_load_r7_r9_tests.log; do
    cp "$training/$file" "$archive/$file"
done
git add -f "$archive"
git diff --cached --check
git commit -m 'Audit knee-supported get-up local optima and test load-aware recovery methods'
git bundle create /data/shijinsheng/open_duck/tmp/getup_loaded_support_20260928.bundle f7bf6dc..main
git status --short
git log -1 --oneline
