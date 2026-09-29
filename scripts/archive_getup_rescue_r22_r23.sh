#!/usr/bin/env bash
set -euo pipefail
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ "$(git rev-parse HEAD)" == 60e557d8d55ecb6b2fc2b0663d00d684e811d722 ]] || { echo 'Unexpected publication baseline'; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty publication checkout'; exit 1; }
[[ -f "$training/getup_rescue_r22/results.json" && -f "$training/getup_rescue_r22_expanded/results.json" ]] || { echo 'Search unfinished'; exit 1; }
"$runtime/.venv/bin/python" -c 'import json,pathlib; p=pathlib.Path("/data/shijinsheng/open_duck/training/getup_distill_r23"); assert all(json.loads((p/n).read_text())["complete"] for n in ("closed_loop_evaluation.json","home_tail_evaluation.json")); assert (p/"checkpoint_audit.json").exists()'
for file in GETUP_RESCUE_DISTILL_20260929.md \
    diagnostics/search_getup_rescue_r22.py diagnostics/test_getup_rescue_r22.py \
    diagnostics/distill_getup_rescue_r23.py diagnostics/test_getup_distill_r23.py \
    diagnostics/eval_getup_distill_r23.py diagnostics/clean_getup_bc_metadata_r23.py \
    scripts/launch_getup_rescue_r22.sh scripts/launch_getup_distill_r23.sh \
    scripts/validate_getup_distill_home_tail_r23.sh scripts/archive_getup_rescue_r22_r23.sh; do
  cp "$runtime/$file" "$publish/$file"
  git add "$file"
done
archive=results/getup_rescue_distill_20260929
mkdir -p "$archive/executed_sources"
for experiment in getup_rescue_r22 getup_rescue_r22_expanded getup_distill_r23; do
  cp -a "$training/$experiment" "$archive/$experiment"
done
cp "$training/getup_rescue_executed_sources_r22_r23/"*.py "$archive/executed_sources/"
for file in getup_rescue_r22.log getup_rescue_r22_expanded.log getup_distill_r23.log \
    getup_distill_home_tail_r23.log getup_rescue_distill_r22_r23_tests.log; do
  cp "$training/$file" "$archive/$file"
done
"$runtime/.venv/bin/python" -c 'import hashlib,json,pathlib; a=pathlib.Path("results/getup_rescue_distill_20260929"); d=a/"getup_distill_r23"; expected=[("distill_getup_rescue_r23.py",json.loads((d/"training_summary.json").read_text())["source_hashes"]["/data/shijinsheng/open_duck/projects/Open_Duck_Playground/diagnostics/distill_getup_rescue_r23.py"]),("eval_getup_distill_r23_initial.py",json.loads((d/"closed_loop_evaluation.json").read_text())["hashes"]["/data/shijinsheng/open_duck/projects/Open_Duck_Playground/diagnostics/eval_getup_distill_r23.py"]),("eval_getup_distill_r23_home_tail.py",json.loads((d/"home_tail_evaluation.json").read_text())["hashes"]["/data/shijinsheng/open_duck/projects/Open_Duck_Playground/diagnostics/eval_getup_distill_r23.py"])]; assert all(hashlib.sha256((a/"executed_sources"/n).read_bytes()).hexdigest()==h for n,h in expected); print("Executed source hashes verified")'
git add -f "$archive"
git diff --cached --check
git commit -m 'Search executable near-standing rescue teachers and reject regressing BC actors after paired native tests'
git bundle create /data/shijinsheng/open_duck/tmp/getup_rescue_r22_r23_20260929.bundle 60e557d..main
git status --short
git log -1 --oneline
