#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=56f564bc943e4117740b06519b0700006a1edc1b
audit=getup_micro_contact_r92_20261001
current=getup_gyro_attenuation_r93_left_20261001
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
test -f "$root/outputs/$audit/results.json"
test -f "$root/outputs/$current/contract.json"
test ! -e "results/$audit"
test ! -e "results/$current"
mkdir "results/$audit" "results/$current"
cp -r "$root/outputs/$audit/." "results/$audit/"
cp "$root/outputs/$current/contract.json" "results/$current/contract.json"
cp -r "$root/outputs/$current/executed_sources" "results/$current/executed_sources"
cp "$root/tmp/getup_r93_unit_tests_20261001.txt" "results/$current/unit_tests.txt"
cd "$source_repo"
.venv/bin/python - "$publication" "$audit" "$current" <<'PY'
import hashlib,json,shutil,time
from pathlib import Path
import sys
repo=Path(sys.argv[1]);audit,current=sys.argv[2:];old=repo/'results'/audit;new=repo/'results'/current
r=json.loads((old/'results.json').read_text())
assert len(r['records'])==9 and all(x['unchanged_outcome_and_trace'] for x in r['records'])
assert json.loads((new/'contract.json').read_text())['r92_result_sha256']==hashlib.sha256((old/'results.json').read_bytes()).hexdigest()
live=Path('/data/shijinsheng/open_duck/outputs')/current
if (live/'grid_progress.json').exists():
    for attempt in range(5):
        try:progress=json.loads((live/'grid_progress.json').read_text());break
        except json.JSONDecodeError:time.sleep(.2)
    else:raise RuntimeError('Cannot read completed grid progress')
    d=new/f'progress_grid{len(progress):02d}';d.mkdir()
    (d/'grid_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    for i in range(len(progress)):shutil.copytree(live/f'grid_{i:02d}',d/f'grid_{i:02d}')
    (d/'archive_stage.json').write_text(json.dumps({'snapshot_only':True,'grid_settings_completed':len(progress),
        'qualification_not_established':True},indent=2)+'\n')
for directory in (old,new):
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}
    (directory/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('R92_EXACT_REPLAYS',len(r['records']),flush=True)
PY
cd "$publication"
for file in GETUP_MICRO_CONTACT_R92_GYRO_R93_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/audit_getup_micro_contact_r92.py diagnostics/test_getup_micro_contact_r92.py diagnostics/train_getup_gyro_attenuation_r93.py diagnostics/test_getup_gyro_attenuation_r93.py scripts/launch_getup_gyro_attenuation_r93_20261001.sh scripts/archive_getup_r92_r93_started_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$audit" "results/$current"
git commit -m "Audit exact getup substep contact divergence and fit bounded early gyro attenuation R93" | tail -n 5
git bundle create "$root/tmp/getup_r92_r93_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
