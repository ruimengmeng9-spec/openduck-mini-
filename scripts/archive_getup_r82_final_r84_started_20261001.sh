#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=807ed217fe4e4db134826311ade510c9691755e1
previous=getup_sensor_clock_r82_left_20261001
audit=getup_clock_selector_audit_r83_20261001
current=getup_initial_settling_r84_left_20261001
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
test -f "$root/outputs/$previous/results.json"
test -f "$root/outputs/$audit/results.json"
test -f "$root/outputs/$current/contract.json"
test ! -e "results/$previous/final"
test ! -e "results/$audit"
test ! -e "results/$current"
mkdir "results/$previous/final" "results/$audit" "results/$current"
cp -r "$root/outputs/$previous/." "results/$previous/final/"
cp "$root/outputs/$previous.log" "results/$previous/final/console.log"
cp -r "$root/outputs/$audit/." "results/$audit/"
cp "$root/outputs/$current/contract.json" "results/$current/contract.json"
cp "$root/tmp/getup_r84_unit_tests_20261001.txt" "results/$current/unit_tests.txt"
cp -r "$root/outputs/$current/executed_sources" "results/$current/executed_sources"
cd "$source_repo"
.venv/bin/python - "$publication" "$previous" "$audit" "$current" <<'PY'
import hashlib,json
from pathlib import Path
import sys
repo=Path(sys.argv[1]);previous,audit,current=sys.argv[2:]
old=repo/'results'/previous/'final'; audit_dir=repo/'results'/audit; new=repo/'results'/current
r=json.loads((old/'results.json').read_text()); a=json.loads((audit_dir/'results.json').read_text())
assert r['training_grid'][0]['successes']==13 and not r['qualification_run']
assert max(x['successes'] for x in r['training_grid'])==13
assert a['oracle_union']==20 and not a['promotion_allowed'] and not a['new_rollouts']
assert a['input_sha256']==hashlib.sha256((old/'results.json').read_bytes()).hexdigest()
for directory in (old,audit_dir,new):
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}
    (directory/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('Preserved negative clock grid, rejected causal selector and R84 launch contract.')
PY
cd "$publication"
for file in GETUP_SENSOR_CLOCK_R82_20261001.md GETUP_INITIAL_SETTLING_R84_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/audit_getup_clock_selector_r83.py diagnostics/probe_getup_initial_settling_r84.py diagnostics/test_getup_initial_settling_r84.py scripts/launch_getup_initial_settling_r84_20261001.sh scripts/archive_getup_r82_final_r84_started_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$previous/final" "results/$audit" "results/$current"
git commit -m "Preserve negative clock and selector evidence; probe physical pre-motion settling R84" | tail -n 5
git bundle create "$root/tmp/getup_r82_final_r84_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
