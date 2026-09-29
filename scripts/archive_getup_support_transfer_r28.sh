#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat PAGER=cat
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
[[ "$(git rev-parse --short HEAD)" == dca2fc3 ]] || { echo 'Unexpected publication baseline'; exit 1; }
for file in GETUP_SUPPORT_TRANSFER_R28_R29_20260929.md \
    diagnostics/probe_getup_support_transfer_r28.py diagnostics/search_getup_reverse_path_r29.py \
    diagnostics/test_getup_support_transfer_r28.py scripts/archive_getup_support_transfer_r28.sh; do
  [[ ! -e "$file" ]] || { echo "Existing source $file"; exit 1; }
  cp "$runtime/$file" "$file"
  git add "$file"
done
archive=results/getup_support_transfer_r28_20260929
[[ ! -e "$archive" ]] || { echo 'Existing result archive'; exit 1; }
mkdir -p "$archive"
cp -a "$training/getup_support_transfer_r28" "$archive/"
cp -a "$training/getup_fullpath_r27/right_side" "$archive/right_side_r27"
cp "$training/getup_support_transfer_r28.log" "$archive/"
"$runtime/.venv/bin/python" -c 'import pathlib,json; r=pathlib.Path("results/getup_support_transfer_r28_20260929"); s=json.loads((r/"getup_support_transfer_r28/results.json").read_text()); assert len(s["rows"])==206 and not s["full_task_completed"]; assert all(p.stat().st_size<95*1024**2 for p in r.rglob("*") if p.is_file())'
git add -f "$archive"
git diff --cached --check
git commit -m 'Test support transfer hypotheses and propose inverse paths from real dynamic falls'
git bundle create /data/shijinsheng/open_duck/tmp/getup_support_transfer_r28_20260929.bundle 0f27e4a..main
git --no-pager log -1 --oneline
