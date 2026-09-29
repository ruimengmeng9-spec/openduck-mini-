#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat PAGER=cat
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
[[ "$(git rev-parse --short HEAD)" == 0e3394b ]] || { echo 'Unexpected publication baseline'; exit 1; }
for file in GETUP_FULL_FALL_BRIDGE_R31_20260929.md diagnostics/bridge_getup_contact_prefixes_r31.py \
    diagnostics/test_getup_bridge_r31.py scripts/launch_getup_bridge_r31.sh scripts/archive_getup_bridge_r31_source.sh; do
  [[ ! -e "$file" ]] || { echo "Existing source $file"; exit 1; }
  cp "$runtime/$file" "$file"
  git add "$file"
done
archive=results/getup_bridge_r31_sources_20260929
[[ ! -e "$archive" ]] || { echo 'Existing result archive'; exit 1; }
mkdir -p "$archive"
cp -a "$training/getup_contact_archive_r30/prone" "$archive/prone_r30"
"$runtime/.venv/bin/python" -c 'import pathlib,json; r=pathlib.Path("results/getup_bridge_r31_sources_20260929"); s=json.loads((r/"prone_r30/results.json").read_text()); assert s["search_controls"]==223823 and s["archive_cells"]==772 and not s["full_task_completed"]; assert all(p.stat().st_size<95*1024**2 for p in r.rglob("*") if p.is_file())'
git add -f "$archive"
git diff --cached --check
git commit -m 'Connect dynamically reached fallen prefixes to verified rescue sequences; preserve prone archive failure'
git bundle create /data/shijinsheng/open_duck/tmp/getup_bridge_r31_sources_20260929.bundle 0f27e4a..main
git --no-pager log -1 --oneline
