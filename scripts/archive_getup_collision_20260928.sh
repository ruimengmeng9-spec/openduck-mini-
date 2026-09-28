#!/usr/bin/env bash
set -euo pipefail
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ "$(git rev-parse HEAD)" == "$(git rev-parse bce977d)" ]] || { echo 'Unexpected publication baseline'; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Publication checkout is dirty'; exit 1; }
for file in GETUP_COLLISION_AND_CROUCH_20260928.md \
    diagnostics/build_getup_decomposed.py diagnostics/audit_getup_support_workspace.py \
    diagnostics/probe_getup_workspace_reference.py diagnostics/getup_dynamic_beam.py \
    diagnostics/qualify_getup_collision.py diagnostics/getup_crouch_extension.py \
    diagnostics/getup_aligned_extension.py diagnostics/validate_getup_aligned.py \
    diagnostics/test_getup_collision_search.py diagnostics/render_getup_dynamic.py \
    scripts/build_getup_decomposed_r4.sh scripts/audit_getup_workspace_r4.sh \
    scripts/probe_getup_workspace_r4.sh scripts/launch_getup_dynamic_r4.sh \
    scripts/qualify_getup_collision_r4.sh scripts/launch_getup_crouch_r5.sh \
    scripts/launch_getup_aligned_r6.sh scripts/archive_getup_collision_20260928.sh; do
    cp "$runtime/$file" "$publish/$file"
    git add "$file"
done
archive=results/getup_collision_20260928
mkdir -p "$archive/getup_decomposed_r4"
cp -r "$training/getup_decomposed_r4/model" "$archive/getup_decomposed_r4/model"
for experiment in getup_workspace_r4 getup_workspace_reference_r4 \
    getup_dynamic_r4b_supine getup_dynamic_r4b_prone getup_crouch_r5_prone getup_aligned_r6_prone; do
    mkdir -p "$archive/$experiment"
    for file in results.json verified_results.json search_progress.json best_reference.npz \
        best_trajectory.npz verified_trajectory.npz; do
        if [[ -f "$training/$experiment/$file" ]]; then
            cp "$training/$experiment/$file" "$archive/$experiment/$file"
        fi
    done
done
for log in getup_decomposed_r4.log getup_workspace_r4.log getup_workspace_reference_r4.log \
    getup_dynamic_r4.log getup_dynamic_r4b.log getup_crouch_r5.log getup_aligned_r6.log getup_aligned_r6_verified.log; do
    if [[ -f "$training/$log" ]]; then cp "$training/$log" "$archive/$log"; fi
done
test -f "$archive/getup_aligned_r6_prone/verified_results.json"
git add -f "$archive"
mkdir -p media/getup_collision_20260928
cp "$training/getup_dynamic_r4b_prone/recovery_search_preview.mp4" \
    media/getup_collision_20260928/prone_failed_handoff.mp4
git add -f media/getup_collision_20260928/prone_failed_handoff.mp4
# Preserve generated artifacts exactly; only edited source is whitespace-checked.
git diff --cached --check -- . ':!results/getup_collision_20260928'
git commit -m 'Investigate get-up self-collision, supported crouch and strict standing handoff'
git bundle create /data/shijinsheng/open_duck/tmp/getup_collision_20260928.bundle bce977d..main
git status --short
git log -1 --oneline
