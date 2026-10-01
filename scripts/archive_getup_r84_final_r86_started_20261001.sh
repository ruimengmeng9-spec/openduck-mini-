#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=2d7a4a1e39b8fe70e8ce51fadde013d5d5059282
previous=getup_initial_settling_r84_left_20261001
audit=getup_substep_contact_audit_r85_20261001
current=getup_early_head_r86_left_20261001
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
cp "$root/tmp/getup_r86_unit_tests_20261001.txt" "results/$current/unit_tests.txt"
cp -r "$root/outputs/$current/executed_sources" "results/$current/executed_sources"
cd "$source_repo"
.venv/bin/python - "$publication" "$previous" "$audit" "$current" <<'PY'
from io import BytesIO
import hashlib,json,time
from pathlib import Path
import sys
import numpy as np
repo=Path(sys.argv[1]);previous,audit,current=sys.argv[2:]
old=repo/'results'/previous/'final';audit_dir=repo/'results'/audit;new=repo/'results'/current
r=json.loads((old/'results.json').read_text());a=json.loads((audit_dir/'results.json').read_text())
assert r['training_grid'][0]['successes']==13 and not r['qualification_run']
assert all(not x['nominal_success'] and not x['cases'] for x in r['training_grid'][1:])
assert a['observer_identity_contract_passed'] and a['physics_commands_and_audit_unchanged']
assert a['r84_result_sha256']==hashlib.sha256((old/'results.json').read_bytes()).hexdigest()
live=Path('/data/shijinsheng/open_duck/outputs')/current
if (live/'checkpoint.npz').exists():
    for attempt in range(5):
        raw=(live/'checkpoint.npz').read_bytes()
        with np.load(BytesIO(raw),allow_pickle=False) as z:generation=int(z['r86_generation'])
        try:
            history=json.loads((live/'search_history.json').read_text())
            if len(history)>=generation:break
        except json.JSONDecodeError:pass
        time.sleep(.2)
    else:raise RuntimeError('Cannot snapshot coherent checkpoint')
    d=new/f'progress_gen{generation:03d}';d.mkdir()
    (d/'checkpoint.npz').write_bytes(raw)
    (d/'search_history.json').write_text(json.dumps(history[:generation],indent=2)+'\n')
    (d/'summary.json').write_text(json.dumps({'snapshot_only':True,'qualification_pending':True,
        'generation':generation,'training_successes':history[generation-1]['best']['successes'],
        'nominal_success':history[generation-1]['best']['nominal_success']},indent=2)+'\n')
for directory in (old,audit_dir,new):
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}
    (directory/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('Preserved negative settling, direct substep contact evidence and early head search.')
PY
cd "$publication"
for file in GETUP_INITIAL_SETTLING_R84_20261001.md GETUP_EARLY_HEAD_R86_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/audit_getup_substep_contact_r85.py diagnostics/train_getup_early_head_r86.py diagnostics/test_getup_early_head_r86.py scripts/launch_getup_early_head_r86_20261001.sh scripts/archive_getup_r84_final_r86_started_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$previous/final" "results/$audit" "results/$current"
git commit -m "Preserve failed settling and transient contact audit; train early head coordination R86" | tail -n 5
git bundle create "$root/tmp/getup_r84_final_r86_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
