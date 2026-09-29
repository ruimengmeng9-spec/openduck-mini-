#!/usr/bin/env bash
set -euo pipefail
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ "$(git rev-parse HEAD)" == 290b66a0e7004730e54141e69fb0075363ae37e6 ]] || { echo 'Publication baseline changed; inspect before archive'; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
"$runtime/.venv/bin/python" -c 'import json,pathlib; r=pathlib.Path("/data/shijinsheng/open_duck/training/getup_expanded_r26"); s=json.loads((r/"results.json").read_text()); assert s["complete"] and not s["hardware_readiness"] and not s["default_controller_replaced"]; print("Completed simulation-only run verified")'
for file in GETUP_EXPANDED_R26_20260929.md diagnostics/train_getup_expanded_r26.py \
    diagnostics/test_getup_expanded_r26.py diagnostics/freeze_getup_expanded_r26.py scripts/launch_getup_expanded_r26.sh \
    scripts/archive_getup_expanded_r26.sh; do
  [[ ! -e "$publish/$file" ]] || { echo "Existing archive target: $file"; exit 1; }
  cp "$runtime/$file" "$publish/$file"
  git add "$file"
done
archive=results/getup_expanded_r26_20260929
[[ ! -e "$archive" ]] || { echo 'Existing result archive'; exit 1; }
mkdir -p "$archive"
cp -a "$training/getup_expanded_r26" "$archive/"
cp "$training/getup_expanded_r26.log" "$archive/"
"$runtime/.venv/bin/python" -c 'import hashlib,json,pathlib; r=pathlib.Path("results/getup_expanded_r26_20260929/getup_expanded_r26"); s=json.loads((r/"results.json").read_text()); assert hashlib.sha256((r/"trajectory_library.npz").read_bytes()).hexdigest()==s["library_sha256"]; assert all(hashlib.sha256((r/"executed_sources"/pathlib.Path(p).name).read_bytes()).hexdigest()==h for p,h in s["hashes"].items() if pathlib.Path(p).suffix==".py"); assert all(f.stat().st_size < 95*1024**2 for f in r.rglob("*") if f.is_file()); print("Source/library hashes and artifact sizes verified")'
git add -f "$archive"
git diff --cached --check
git commit -m 'Expand verified near-standing rescue atlas and compare frozen policy against R25 on fresh paired states'
git bundle create /data/shijinsheng/open_duck/tmp/getup_expanded_r26_20260929.bundle 290b66a..main
git status --short
git --no-pager log -1 --oneline
