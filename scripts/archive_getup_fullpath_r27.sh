#!/usr/bin/env bash
set -euo pipefail
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
phase=${1:-source}
cd "$publish"
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
if [[ "$phase" == source ]]; then
  [[ "$(git rev-parse HEAD)" == 0f27e4aeeb10af63063f70fa82b68198def8ed16 ]] || { echo 'Unexpected source publication baseline'; exit 1; }
  for file in GETUP_FULLPATH_R27_20260929.md diagnostics/train_getup_fullpath_r27.py \
      diagnostics/validate_getup_fullpath_r27.py diagnostics/audit_getup_fullpath_r27.py \
      diagnostics/test_getup_fullpath_r27.py scripts/launch_getup_fullpath_r27.sh \
      scripts/archive_getup_fullpath_r27.sh; do
    [[ ! -e "$file" ]] || { echo "Existing source: $file"; exit 1; }
    cp "$runtime/$file" "$file"
    git add "$file"
  done
  archive=results/getup_fullpath_r27_audit_20260929
  [[ ! -e "$archive" ]] || { echo 'Existing audit archive'; exit 1; }
  mkdir -p "$archive"
  cp "$training/getup_fullpath_r27_audit/results.json" "$archive/"
  cp "$training/getup_fullpath_r27/left_side/contract.json" "$archive/search_contract.json"
  git add -f "$archive"
  git diff --cached --check
  git commit -m 'Search true fallen get-up paths with substep collision audits and independent strict acceptance'
elif [[ "$phase" == results ]]; then
  "$runtime/.venv/bin/python" -c 'import pathlib; r=pathlib.Path("/data/shijinsheng/open_duck/training/getup_fullpath_r27"); assert all((r/p/"results.json").exists() for p in ("prone","supine","left_side","right_side")); assert pathlib.Path("/data/shijinsheng/open_duck/training/getup_fullpath_r27_strict/results.json").exists()'
  archive=results/getup_fullpath_r27_20260929
  [[ ! -e "$archive" ]] || { echo 'Existing results archive'; exit 1; }
  mkdir -p "$archive"
  cp -a "$training/getup_fullpath_r27" "$archive/"
  cp -a "$training/getup_fullpath_r27_strict" "$archive/"
  cp "$training/getup_fullpath_r27.log" "$archive/"
  "$runtime/.venv/bin/python" -c 'import pathlib,hashlib,json; r=pathlib.Path("results/getup_fullpath_r27_20260929"); assert all(p.stat().st_size<95*1024**2 for p in r.rglob("*") if p.is_file()); [(lambda c,p: c["hashes"][next(k for k in c["hashes"] if k.endswith("train_getup_fullpath_r27.py"))]==hashlib.sha256(p.read_bytes()).hexdigest() or (_ for _ in ()).throw(AssertionError("source mismatch")))(json.loads(c.read_text()),c.parent/"executed_sources/train_getup_fullpath_r27.py") for c in r.glob("getup_fullpath_r27/*/contract.json")]'
  git add -f "$archive"
  git diff --cached --check
  git commit -m 'Archive full fallen trajectory search and strict held-out recovery failures or successes'
else
  echo 'Expected source or results' >&2
  exit 1
fi
git bundle create /data/shijinsheng/open_duck/tmp/getup_fullpath_r27_20260929.bundle 0f27e4a..main
git --no-pager log -1 --oneline
