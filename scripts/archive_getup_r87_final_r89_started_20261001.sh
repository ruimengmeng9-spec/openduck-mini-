#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=613298385bc81ab1477639da9486328950cf5806
previous=getup_head_feedback_r87_left_20261001
audit=getup_head_selector_audit_r88_20261001
current=getup_head_distillation_r89_left_20261001
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
test -f "$root/outputs/$previous/results.json"
test -f "$root/outputs/$audit/results.json"
test -f "$root/outputs/$current/teacher_review.json"
test ! -e "results/$previous/final"
test ! -e "results/$audit"
test ! -e "results/$current"
mkdir "results/$previous/final" "results/$audit" "results/$current"
cp -r "$root/outputs/$previous/." "results/$previous/final/"
cp "$root/outputs/$previous.log" "results/$previous/final/console.log"
cp -r "$root/outputs/$audit/." "results/$audit/"
for file in contract.json teacher_review.json teacher_dataset.npz dataset_summary.json fit_history.json network_epoch050.npz network_epoch150.npz network_epoch300.npz; do
    cp "$root/outputs/$current/$file" "results/$current/$file"
done
cp -r "$root/outputs/$current/teachers" "$root/outputs/$current/executed_sources" "results/$current/"
cp "$root/tmp/getup_r89_unit_tests_20261001.txt" "results/$current/unit_tests.txt"
cd "$source_repo"
.venv/bin/python - "$publication" "$previous" "$audit" "$current" <<'PY'
import hashlib,json
from pathlib import Path
import sys
import numpy as np
repo=Path(sys.argv[1]);previous,audit,current=sys.argv[2:]
old=repo/'results'/previous/'final';screen=repo/'results'/audit;new=repo/'results'/current
r=json.loads((old/'results.json').read_text());a=json.loads((screen/'results.json').read_text())
assert r['training_best']['successes']==13 and not r['qualification_run']
with np.load(old/'checkpoint.npz',allow_pickle=False) as z:assert not np.any(z['r87_head_gains'])
rows=[json.loads(f.read_text()) for f in (old/'candidates').glob('*/summary.json')]
base_seeds={c['seed'] for c in r['training_best']['cases'] if c['success']}
rescues=sorted({c['seed'] for row in rows for c in row.get('cases',[]) if c['success'] and c['seed'] not in base_seeds})
summary={'candidate_count':len(rows),'nominal_eligible':sum(c['nominal_success'] for c in rows),
         'nonidentity_best':max(c['successes'] for c in rows if np.any(c['params'])),
         'rescued_seeds_across_different_candidates':rescues,'oracle_not_policy':True,
         'qualification_run':False}
(old/'candidate_frontier_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert len(rows)==312 and len(rescues)==11
assert a['best_screen']['successes']==9 and not a['promotion_allowed']
t=json.loads((new/'teacher_review.json').read_text());assert len(t)==25 and all(x['success'] for x in t)
assert json.loads((new/'contract.json').read_text())['r87_result_sha256']==hashlib.sha256((old/'results.json').read_bytes()).hexdigest()
(new/'archive_stage.json').write_text(json.dumps({'snapshot_only':True,'teacher_short_gate_reproduced':25,
    'student_closed_loop_pending_at_launch_snapshot':True,'qualification_not_yet_established':True},indent=2)+'\n')
for directory in (old,screen,new):
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}
    (directory/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps(summary),flush=True)
PY
cd "$publication"
for file in GETUP_HEAD_FEEDBACK_R87_20261001.md GETUP_HEAD_DISTILLATION_R88_R89_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/audit_getup_head_selector_r88.py diagnostics/test_getup_head_selector_r88.py diagnostics/train_getup_head_distillation_r89.py diagnostics/test_getup_head_distillation_r89.py scripts/launch_getup_head_distillation_r89_20261001.sh scripts/archive_getup_r87_final_r89_started_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$previous/final" "results/$audit" "results/$current"
git commit -m "Preserve negative R87 and R88 audits and start causal getup distillation R89" | tail -n 5
git bundle create "$root/tmp/getup_r87_final_r89_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
