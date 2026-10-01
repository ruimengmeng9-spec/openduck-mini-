#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=2b6e52c2b14652017075b6a77910f56a842c9933
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
test -f "$root/outputs/getup_preload_r95_left_20261001/results.json"
test -f "$root/outputs/getup_initial_encoder_r96_left_20261001/contract.json"
test ! -e results/getup_preload_r95_left_20261001/final
test ! -e results/getup_initial_encoder_r96_left_20261001
cd "$source_repo"
.venv/bin/python - "$publication" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
repo=Path(sys.argv[1]);root=Path('/data/shijinsheng/open_duck')
old=root/'outputs/getup_preload_r95_left_20261001'
new=root/'outputs/getup_initial_encoder_r96_left_20261001'
r=json.loads((old/'results.json').read_text())
assert r['generations_completed']==17 and not r['qualification_run']
assert r['training_best']['successes']==13 and not np.any(r['training_best']['params'])
assert all(v['successes']==13 and v['nominal_success'] for v in r['training_review'])
dst=repo/'results'/old.name/'final';dst.mkdir()
for f in old.iterdir():
    if f.is_file():shutil.copy2(f,dst/f.name)
shutil.copytree(old/'executed_sources',dst/'executed_sources')
for name in ('review_baseline','review_candidate'):
    shutil.copytree(old/name,dst/name)
# Full experiment traces stay on the server. Publish all candidate summaries
# and three named contact-diagnostic trajectories per evaluated candidate.
for folder in sorted(old.glob('gen_*_candidate_*')):
    target=dst/folder.name;target.mkdir()
    for name in ('summary.json','case_None.npz','case_769003.npz','case_773014.npz','failed_reference.npz'):
        if (folder/name).is_file():shutil.copy2(folder/name,target/name)
hashes={str(f.relative_to(old)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in old.rglob('*') if f.is_file()}
(dst/'full_server_artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
(dst/'archive_scope.json').write_text(json.dumps({
    'server_full_trace_directory':str(old),'server_originals_preserved':True,
    'all_candidate_summaries_published':True,'all_final_review_traces_published':True,
    'representative_candidate_trace_cases':[None,769003,773014],
    'prior_generation_six_archive_preserved':True,'final_generations':17,
    'qualification_run':False},indent=2)+'\n')
shutil.copy2(root/'tmp/getup_r95_unit_tests_20261001.txt',dst/'unit_tests.txt')
current=repo/'results'/new.name;current.mkdir()
complete=(new/'results.json').is_file()
shutil.copy2(new/'contract.json',current/'contract.json')
shutil.copytree(new/'executed_sources',current/'executed_sources')
shutil.copy2(root/'tmp/getup_r96_unit_tests_20261001.txt',current/'unit_tests.txt')
# Snapshot only completed arms; workers create summary last, after all traces.
folders=[f for f in new.iterdir() if f.is_dir() and (f/'summary.json').is_file()]
for folder in folders:shutil.copytree(folder,current/folder.name)
if complete:
    for name in ('results.json','checkpoint.npz','grid_progress.json','robust_progress.json'):
        if (new/name).is_file():shutil.copy2(new/name,current/name)
else:
    # Do not mix a later final checkpoint with earlier captured grid arms.
    snapshot=[]
    for folder in folders:
        if folder.name.startswith('grid_'):
            value=json.loads((current/folder.name/'summary.json').read_text())
            value['grid_index']=int(folder.name.split('_')[1]);snapshot.append(value)
    (current/'grid_progress.json').write_text(json.dumps(sorted(snapshot,key=lambda v:v['grid_index']),indent=2)+'\n')
(current/'archive_stage.json').write_text(json.dumps({
    'snapshot_only':not complete,'complete_arms':sorted(f.name for f in folders),
    'qualification_not_established':True},indent=2)+'\n')
for directory in (dst,current):
    artifacts={str(f.relative_to(directory)):hashlib.sha256(f.read_bytes()).hexdigest()
               for f in directory.rglob('*') if f.is_file()}
    (directory/'artifact_hashes.json').write_text(json.dumps(artifacts,indent=2)+'\n')
print('R95_FINAL_AND_R96_SNAPSHOT_SAVED',len(folders),flush=True)
PY
cd "$publication"
for file in GETUP_COORDINATED_PRELOAD_R95_20261001.md GETUP_INITIAL_ENCODER_COMPENSATION_R96_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/probe_getup_initial_encoder_r96.py diagnostics/test_getup_initial_encoder_r96.py scripts/launch_getup_initial_encoder_r96_20261001.sh scripts/archive_getup_r95_final_r96_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f results/getup_preload_r95_left_20261001/final results/getup_initial_encoder_r96_left_20261001
git -c gc.auto=0 commit -m "Record rejected R95 preload and test causal initial encoder compensation R96" | tail -n 5
git bundle create "$root/tmp/getup_r95_final_r96_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
