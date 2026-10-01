#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=c8657e3077ef5c3f58ebdc7e5952e671b00234cc
r89=getup_head_distillation_r89_left_20261001
r90=getup_centered_head_r90_left_20261001
r91=getup_head_tolerance_r91_left_20261001
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
test -f "$root/outputs/$r89/results.json"
test -f "$root/outputs/$r90/results.json"
test -f "$root/outputs/$r91/contract.json"
test ! -e "results/$r89/final"
test ! -e "results/$r90"
test ! -e "results/$r91"
mkdir "results/$r89/final" "results/$r90" "results/$r91"
cp -r "$root/outputs/$r89/." "results/$r89/final/"
cp "$root/outputs/$r89.log" "results/$r89/final/console.log"
cp -r "$root/outputs/$r90/." "results/$r90/"
cp "$root/outputs/$r90.log" "results/$r90/console.log"
cp "$root/tmp/getup_r90_unit_tests_20261001.txt" "results/$r90/unit_tests.txt"
cp "$root/outputs/$r91/contract.json" "results/$r91/contract.json"
cp -r "$root/outputs/$r91/executed_sources" "results/$r91/executed_sources"
cp "$root/tmp/getup_r91_unit_tests_20261001.txt" "results/$r91/unit_tests.txt"
cd "$source_repo"
.venv/bin/python - "$publication" "$r89" "$r90" "$r91" <<'PY'
import hashlib,json,shutil,time
from pathlib import Path
import sys
repo=Path(sys.argv[1]);r89,r90,r91=sys.argv[2:]
old=repo/'results'/r89/'final';center=repo/'results'/r90;new=repo/'results'/r91
a=json.loads((old/'results.json').read_text());b=json.loads((center/'results.json').read_text())
assert a['training_best']['successes']==9 and not a['qualification_run']
assert b['training_best']['successes']==7 and not b['qualification_run']
assert all(s['nominal']['nominal_exact_identity'] for s in b['training_screen'])
assert json.loads((new/'contract.json').read_text())['r90_result_sha256']==hashlib.sha256((center/'results.json').read_bytes()).hexdigest()
live=Path('/data/shijinsheng/open_duck/outputs')/r91
if (live/'progress.json').exists():
    for attempt in range(5):
        try:progress=json.loads((live/'progress.json').read_text());break
        except json.JSONDecodeError:time.sleep(.2)
    else:raise RuntimeError('Cannot read complete diagnostic progress')
    snapshot=new/f'progress_arms{len(progress):02d}';snapshot.mkdir()
    (snapshot/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    for index in range(len(progress)):
        shutil.copytree(live/f'grid_{index:02d}',snapshot/f'grid_{index:02d}')
    if (live/'results.json').exists():shutil.copy2(live/'results.json',snapshot/'results.json')
    (snapshot/'archive_stage.json').write_text(json.dumps({'snapshot_only':True,
        'completed_prescribed_arms':len(progress),'diagnostic_not_qualification':True},indent=2)+'\n')
for directory in (old,center,new):
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}
    (directory/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('R89_BEST',a['training_best']['successes'],'R90_BEST',b['training_best']['successes'],flush=True)
PY
cd "$publication"
for file in GETUP_HEAD_DISTILLATION_R88_R89_20261001.md GETUP_HEAD_TOLERANCE_R91_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/probe_getup_centered_head_r90.py diagnostics/test_getup_centered_head_r90.py diagnostics/probe_getup_head_tolerance_r91.py diagnostics/test_getup_head_tolerance_r91.py scripts/launch_getup_centered_head_r90_20261001.sh scripts/launch_getup_head_tolerance_r91_20261001.sh scripts/archive_getup_r89_r90_final_r91_started_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$r89/final" "results/$r90" "results/$r91"
git commit -m "Preserve negative getup distillation and centering and audit tiny head-command tolerance R91" | tail -n 5
git bundle create "$root/tmp/getup_r89_r90_final_r91_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
