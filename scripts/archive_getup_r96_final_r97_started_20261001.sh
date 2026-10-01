#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=b04450d69235dce21933ba000a9454edb7e68eeb
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
test -f "$root/outputs/getup_initial_encoder_r96_left_20261001/results.json"
test -f "$root/outputs/getup_vertical_imu_r97_left_20261001/search_history.json"
test ! -e results/getup_initial_encoder_r96_left_20261001/final
test ! -e results/getup_vertical_imu_r97_left_20261001
cd "$source_repo"
.venv/bin/python - "$publication" <<'PY'
import hashlib,io,json,shutil,sys,time
from pathlib import Path
import numpy as np
repo=Path(sys.argv[1]);root=Path('/data/shijinsheng/open_duck')
old=root/'outputs/getup_initial_encoder_r96_left_20261001'
new=root/'outputs/getup_vertical_imu_r97_left_20261001'
r=json.loads((old/'results.json').read_text())
assert len(r['grid_results'])==73 and not r['qualification_run'] and not r['promoted']
assert r['training_best']['successes']==13 and not np.any(r['training_best']['scales'])
dst=repo/'results'/old.name/'final';dst.mkdir()
for f in old.iterdir():
    if f.is_file():shutil.copy2(f,dst/f.name)
for folder in sorted(old.glob('grid_*')):
    if not folder.is_dir():continue
    target=dst/folder.name;target.mkdir()
    names=('summary.json','case_None.npz','case_769003.npz','case_773014.npz')
    for f in folder.iterdir():
        if f.name in names or folder.name in ('grid_00','grid_09'):shutil.copy2(f,target/f.name)
hashes={str(f.relative_to(old)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in old.rglob('*') if f.is_file()}
(dst/'full_server_artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
(dst/'archive_scope.json').write_text(json.dumps({
    'server_full_trace_directory':str(old),'server_originals_preserved':True,
    'all_grid_summaries_published':True,'complete_baseline_and_best_nonidentity_traces_published':True,
    'other_grid_representative_cases':[None,769003,773014],
    'prior_complete_arm_snapshot_preserved':True,'qualification_run':False},indent=2)+'\n')
current=repo/'results'/new.name;current.mkdir()
shutil.copy2(new/'contract.json',current/'contract.json')
shutil.copytree(new/'executed_sources',current/'executed_sources')
shutil.copy2(root/'tmp/getup_r97_unit_tests_20261001.txt',current/'unit_tests.txt')
for attempt in range(10):
    try:
        history=json.loads((new/'search_history.json').read_text())
        checkpoint=(new/'checkpoint.npz').read_bytes()
        with np.load(io.BytesIO(checkpoint),allow_pickle=False) as z:generation=int(z['r97_generation'])
        if generation==history[-1]['generation']:break
    except (json.JSONDecodeError,FileNotFoundError):pass
    time.sleep(.5)
else:raise RuntimeError('No coherent completed generation and checkpoint')
(current/'checkpoint.npz').write_bytes(checkpoint)
(current/'search_history.json').write_text(json.dumps(history,indent=2)+'\n')
for gen in range(1,generation+1):
    for i in range(18):
        folder=new/f'gen_{gen:03d}_candidate_{i:03d}'
        target=current/folder.name;target.mkdir()
        for f in folder.iterdir():
            if f.name in ('summary.json','vertical_reference.npz','case_None.npz','case_769003.npz','case_773014.npz',
                          'vertical_error_None.npz','vertical_error_769003.npz','vertical_error_773014.npz'):
                shutil.copy2(f,target/f.name)
(current/'archive_stage.json').write_text(json.dumps({'snapshot_only':True,'generations_completed':generation,
    'all_candidate_summaries_published':True,'representative_cases':[None,769003,773014],
    'server_full_trace_directory':str(new),'server_originals_preserved':True,
    'qualification_not_established':True},indent=2)+'\n')
for directory in (dst,current):
    artifacts={str(f.relative_to(directory)):hashlib.sha256(f.read_bytes()).hexdigest()
               for f in directory.rglob('*') if f.is_file()}
    (directory/'artifact_hashes.json').write_text(json.dumps(artifacts,indent=2)+'\n')
print('R96_FINAL_R97_GENERATION',generation,flush=True)
PY
cd "$publication"
for file in GETUP_INITIAL_ENCODER_COMPENSATION_R96_20261001.md GETUP_VERTICAL_IMU_FEEDBACK_R97_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/train_getup_vertical_imu_r97.py diagnostics/test_getup_vertical_imu_r97.py scripts/launch_getup_vertical_imu_r97_20261001.sh scripts/archive_getup_r96_final_r97_started_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f results/getup_initial_encoder_r96_left_20261001/final results/getup_vertical_imu_r97_left_20261001
git -c gc.auto=0 commit -m "Reject initial encoder compensation R96 and train vertical IMU recovery feedback R97" | tail -n 5
git bundle create "$root/tmp/getup_r96_final_r97_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
