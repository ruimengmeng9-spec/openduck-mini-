#!/usr/bin/env bash
set -euo pipefail
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
cd "$publish"
finish_archive() {
    # Preserve generated XML byte-for-byte so archived audit hashes remain true.
    # Upstream XML contains whitespace-only lines; check edited source separately.
    git diff --cached --check -- . ':!results/getup_independent_20260928'
    git commit -m 'Develop independent get-up reference search and strict simulation validation'
    git bundle create /data/shijinsheng/open_duck/tmp/getup_independent_20260928.bundle b76932d..main
    git status --short
    git log -1 --oneline
}
if [[ "${1:-}" == --resume ]]; then
    while IFS= read -r path; do
        case "$path" in
            GETUP_INDEPENDENT_20260928.md|diagnostics/getup_*.py|diagnostics/audit_getup_self_collision.py|diagnostics/validate_getup_independent.py|diagnostics/test_getup_independent_native.py|diagnostics/render_getup_verified.py|scripts/launch_getup_independent_r*.sh|scripts/validate_getup_independent_20260928.sh|scripts/archive_getup_independent_20260928.sh|results/getup_independent_20260928/*|media/getup_independent_20260928/*) ;;
            *) echo "Unexpected staged path: $path" >&2; exit 1 ;;
        esac
    done < <(git diff --cached --name-only)
    if [[ -n "$(git diff --name-only)" ]]; then
        echo 'Unexpected unstaged changes; refusing resume.' >&2
        exit 1
    fi
    cp "$runtime/scripts/archive_getup_independent_20260928.sh" scripts/archive_getup_independent_20260928.sh
    git add scripts/archive_getup_independent_20260928.sh
    finish_archive
    exit 0
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo 'Publication checkout is not clean; refusing to mix unrelated changes.' >&2
    exit 1
fi
for file in GETUP_INDEPENDENT_20260928.md \
    diagnostics/getup_independent_native.py diagnostics/getup_feedback_reference.py \
    diagnostics/getup_beam_reference.py diagnostics/audit_getup_self_collision.py \
    diagnostics/validate_getup_independent.py diagnostics/test_getup_independent_native.py \
    diagnostics/render_getup_verified.py \
    scripts/launch_getup_independent_r1.sh scripts/launch_getup_independent_r2.sh \
    scripts/launch_getup_independent_r3.sh scripts/validate_getup_independent_20260928.sh \
    scripts/archive_getup_independent_20260928.sh; do
    cp "$runtime/$file" "$publish/$file"
    git add "$file"
done
archive=results/getup_independent_20260928
mkdir -p "$archive"
for experiment in getup_independent_precheck_r1 getup_independent_r1_supine getup_independent_r1_prone \
    getup_independent_r2_supine getup_independent_r2_prone getup_independent_r3_supine getup_independent_r3_prone; do
    source_dir="/data/shijinsheng/open_duck/training/$experiment"
    mkdir -p "$archive/$experiment"
    for file in precheck.json results.json verified_results.json search_progress.json \
        self_collision_audit.json best_reference.npz best_trajectory.npz verified_trajectory.npz; do
        if [[ -f "$source_dir/$file" ]]; then
            cp "$source_dir/$file" "$archive/$experiment/$file"
        fi
    done
    cp -r "$source_dir/model" "$archive/$experiment/model"
done
for round in r1 r2 r3; do
    cp "/data/shijinsheng/open_duck/training/getup_independent_${round}.log" "$archive/"
done
git add -f "$archive"
mkdir -p media/getup_independent_20260928
cp /data/shijinsheng/open_duck/training/getup_independent_r2_supine/recovery_search_preview.mp4 \
    media/getup_independent_20260928/supine_failed_search.mp4
git add -f media/getup_independent_20260928/supine_failed_search.mp4
finish_archive
