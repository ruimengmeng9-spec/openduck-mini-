#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat PAGER=cat
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
[[ "$(git rev-parse --short HEAD)" == c1ef3c0 ]] || { echo 'Unexpected publication baseline'; exit 1; }
for file in GETUP_CONTACT_ARCHIVE_R30_20260929.md diagnostics/search_getup_contact_archive_r30.py \
    diagnostics/test_getup_contact_archive_r30.py diagnostics/run_getup_spawn.py \
    scripts/launch_getup_contact_archive_r30.sh scripts/archive_getup_contact_archive_r30_source.sh; do
  [[ ! -e "$file" ]] || { echo "Existing source $file"; exit 1; }
  cp "$runtime/$file" "$file"
  git add "$file"
done
archive=results/getup_contact_archive_r30_sources_20260929
[[ ! -e "$archive" ]] || { echo 'Existing result archive'; exit 1; }
mkdir -p "$archive"
cp -a "$training/getup_reverse_path_r29" "$archive/"
cp "$training/getup_reverse_path_r29.log" "$archive/"
cp -a "$training/getup_contact_archive_r30_smoke" "$archive/"
cp -a "$training/getup_contact_archive_r30_spawn_smoke" "$archive/"
cp "$training/getup_contact_archive_r30_spawn_smoke_spawn_launcher.json" "$archive/"
"$runtime/.venv/bin/python" -c 'import pathlib,json; r=pathlib.Path("results/getup_contact_archive_r30_sources_20260929"); s=json.loads((r/"getup_reverse_path_r29/results.json").read_text()); assert len(s["rows"])==256 and not s["full_task_completed"]; a=json.loads((r/"getup_contact_archive_r30_smoke/results.json").read_text()); b=json.loads((r/"getup_contact_archive_r30_spawn_smoke/results.json").read_text()); assert a["search_controls"]==b["search_controls"]==1073 and a["archive_cells"]==b["archive_cells"]==9; assert all(p.stat().st_size<95*1024**2 for p in r.rglob("*") if p.is_file())'
git add -f "$archive"
git diff --cached --check
git commit -m 'Explore contact-diverse fallen prefixes; archive inverse-path failures and process-start parity'
git bundle create /data/shijinsheng/open_duck/tmp/getup_contact_archive_r30_sources_20260929.bundle 0f27e4a..main
git --no-pager log -1 --oneline
