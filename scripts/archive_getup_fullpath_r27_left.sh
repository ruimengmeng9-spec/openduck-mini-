#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat PAGER=cat
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training/getup_fullpath_r27/left_side
cd "$publish"
[[ "$(git rev-parse HEAD)" == 37a37e7fff396baff8e91316522e74f81ff0d983 ]] || { echo 'Unexpected baseline'; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
"$runtime/.venv/bin/python" -c 'import json,pathlib; s=json.loads(pathlib.Path("/data/shijinsheng/open_duck/training/getup_fullpath_r27/left_side/results.json").read_text()); assert s["validation_runs"]==20 and not s["full_task_completed"]; print("Completed left-side experiment verified; full task not complete")'
archive=results/getup_fullpath_r27_left_side_20260929
[[ ! -e "$archive" ]] || { echo 'Existing result archive'; exit 1; }
mkdir -p "$archive"
cp -a "$training" "$archive/"
cp "$runtime/GETUP_FULLPATH_R27_20260929.md" .
cp "$runtime/scripts/archive_getup_fullpath_r27_left.sh" scripts/
git add GETUP_FULLPATH_R27_20260929.md scripts/archive_getup_fullpath_r27_left.sh
git add -f "$archive"
git diff --cached --check
git commit -m 'Preserve full fallen left-side search failure and independent replay results'
git bundle create /data/shijinsheng/open_duck/tmp/getup_fullpath_r27_20260929.bundle 0f27e4a..main
git --no-pager log -1 --oneline
