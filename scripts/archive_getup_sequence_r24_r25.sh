#!/usr/bin/env bash
set -euo pipefail
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ "$(git rev-parse HEAD)" == 4d3323865317594733b1a4185dc1017cc5dcfafb ]] || { echo 'Unexpected publication baseline'; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
"$runtime/.venv/bin/python" -c 'import json,pathlib; r=pathlib.Path("/data/shijinsheng/open_duck/training"); assert all(json.loads((r/n/"results.json").read_text())["complete"] for n in ("getup_sequence_r24_fixed","getup_transfer_r25","getup_sequence_r25_broad")); assert (r/"getup_transfer_r25/frozen_policy.json").exists()'
for file in GETUP_SEQUENCE_ATLAS_20260929.md \
    diagnostics/train_getup_sequence_r24.py diagnostics/test_getup_sequence_r24.py \
    diagnostics/train_getup_transfer_r25.py diagnostics/validate_getup_sequence_r25.py \
    diagnostics/freeze_getup_sequence_r25.py scripts/launch_getup_sequence_r24.sh \
    scripts/launch_getup_transfer_r25.sh scripts/validate_getup_sequence_r25.sh \
    scripts/archive_getup_sequence_r24_r25.sh; do
  cp "$runtime/$file" "$publish/$file"
  git add "$file"
done
archive=results/getup_sequence_atlas_20260929
mkdir -p "$archive"
for experiment in getup_sequence_r24 getup_sequence_r24_fixed getup_transfer_r25 \
    getup_sequence_r25_broad getup_sequence_r24_failed_source; do
  cp -a "$training/$experiment" "$archive/$experiment"
done
for file in getup_sequence_r24.log getup_sequence_r24_fixed.log getup_transfer_r25.log \
    getup_sequence_r25_broad.log getup_sequence_r24_r25_tests.log; do
  cp "$training/$file" "$archive/$file"
done
"$runtime/.venv/bin/python" -c 'import hashlib,json,pathlib; a=pathlib.Path("results/getup_sequence_atlas_20260929"); cases=[(a/"getup_sequence_r24/results.json",a/"getup_sequence_r24_failed_source/train_getup_sequence_r24.py","train_getup_sequence_r24.py"),(a/"getup_sequence_r24_fixed/results.json",pathlib.Path("diagnostics/train_getup_sequence_r24.py"),"train_getup_sequence_r24.py"),(a/"getup_transfer_r25/results.json",pathlib.Path("diagnostics/train_getup_transfer_r25.py"),"train_getup_transfer_r25.py"),(a/"getup_sequence_r25_broad/results.json",pathlib.Path("diagnostics/validate_getup_sequence_r25.py"),"validate_getup_sequence_r25.py")]; assert all(hashlib.sha256(source.read_bytes()).hexdigest()==json.loads(report.read_text())["hashes"]["/data/shijinsheng/open_duck/projects/Open_Duck_Playground/diagnostics/"+name] for report,source,name in cases); print("Executed source hashes verified")'
git add -f "$archive"
git diff --cached --check
git commit -m 'Fit conservative whole-sequence recovery atlas with counterfactual teacher transfer and frozen held-out validation'
git bundle create /data/shijinsheng/open_duck/tmp/getup_sequence_r24_r25_20260929.bundle 4d33238..main
git status --short
git log -1 --oneline
