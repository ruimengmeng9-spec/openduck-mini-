#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=ffe11f89f8e27fc272171e3cea68be3e9ebf5dfe
previous=getup_early_head_r86_left_20261001
current=getup_head_feedback_r87_left_20261001
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
test -f "$root/outputs/$previous/results.json"
test -f "$root/outputs/$current/contract.json"
test ! -e "results/$previous/final"
test ! -e "results/$current"
mkdir "results/$previous/final" "results/$current"
cp -r "$root/outputs/$previous/." "results/$previous/final/"
cp "$root/outputs/$previous.log" "results/$previous/final/console.log"
cp "$root/outputs/$current/contract.json" "results/$current/contract.json"
cp "$root/tmp/getup_r87_unit_tests_20261001.txt" "results/$current/unit_tests.txt"
cp -r "$root/outputs/$current/executed_sources" "results/$current/executed_sources"
cd "$source_repo"
.venv/bin/python - "$publication" "$previous" "$current" <<'PY'
from io import BytesIO
import hashlib,json,time
from pathlib import Path
import sys
import numpy as np
repo=Path(sys.argv[1]);previous,current=sys.argv[2:]
old=repo/'results'/previous/'final';new=repo/'results'/current
r=json.loads((old/'results.json').read_text())
assert r['training_best']['successes']==13 and not r['qualification_run']
with np.load(old/'checkpoint.npz',allow_pickle=False) as z:
    assert not np.any(z['r86_head_delta'])
rows=[json.loads(f.read_text()) for f in (old/'candidates').glob('*/summary.json')]
eligible=[x for x in rows if x['nominal_success']]
baseline={x['seed'] for x in r['training_best']['cases'] if x['success']}
rescues=sorted({x['seed'] for c in eligible for x in c.get('cases',[]) if x['success'] and x['seed'] not in baseline})
nonidentity=[x for x in eligible if np.any(x['params'])]
summary={'candidate_count':len(rows),'nominal_eligible':len(eligible),
         'best_nonidentity_training_successes':max(x['successes'] for x in nonidentity),
         'rescued_seeds_across_different_candidates':rescues,'oracle_not_policy':True,
         'independent_qualification_run':False}
(old/'candidate_frontier_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert len(rows)==312 and len(eligible)==98 and len(rescues)==11
contract=json.loads((new/'contract.json').read_text())
assert contract['r86_result_sha256']==hashlib.sha256((old/'results.json').read_bytes()).hexdigest()
live=Path('/data/shijinsheng/open_duck/outputs')/current
if (live/'checkpoint.npz').exists():
    for attempt in range(5):
        raw=(live/'checkpoint.npz').read_bytes()
        with np.load(BytesIO(raw),allow_pickle=False) as z:generation=int(z['r87_generation'])
        try:
            history=json.loads((live/'search_history.json').read_text())
            if len(history)>=generation:break
        except json.JSONDecodeError:pass
        time.sleep(.2)
    else:raise RuntimeError('Cannot snapshot coherent checkpoint/history')
    d=new/f'progress_gen{generation:03d}';d.mkdir()
    (d/'checkpoint.npz').write_bytes(raw)
    (d/'search_history.json').write_text(json.dumps(history[:generation],indent=2)+'\n')
    (d/'summary.json').write_text(json.dumps({'snapshot_only':True,'qualification_pending':True,
        'generation':generation,'training_successes':history[generation-1]['best']['successes'],
        'nominal_success':history[generation-1]['best']['nominal_success']},indent=2)+'\n')
for directory in (old,new):
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}
    (directory/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps(summary),flush=True)
PY
cd "$publication"
for file in GETUP_EARLY_HEAD_R86_20261001.md GETUP_HEAD_FEEDBACK_R87_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/train_getup_head_feedback_r87.py diagnostics/test_getup_head_feedback_r87.py scripts/launch_getup_head_feedback_r87_20261001.sh scripts/archive_getup_r86_final_r87_started_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$previous/final" "results/$current"
git commit -m "Preserve negative R86 candidate frontier and train causal head feedback R87" | tail -n 5
git bundle create "$root/tmp/getup_r86_final_r87_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
