#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=843e33c310f05f596062de624c36fb7aefedc5bd
prior=getup_gyro_attenuation_r93_left_20261001
phase=getup_head_phase_r94_left_20261001
current=getup_preload_r95_left_20261001
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
test -f "$root/outputs/$prior/results.json"
test -f "$root/outputs/$phase/results.json"
test -f "$root/outputs/$current/contract.json"
test ! -e "results/$prior/final"
test ! -e "results/$phase"
test ! -e "results/$current"
mkdir "results/$prior/final" "results/$phase" "results/$current"
cp -r "$root/outputs/$prior/." "results/$prior/final/"
cp -r "$root/outputs/$phase/." "results/$phase/"
cp "$root/outputs/$current/contract.json" "results/$current/contract.json"
cp -r "$root/outputs/$current/executed_sources" "results/$current/executed_sources"
cp "$root/tmp/getup_r94_unit_tests_20261001.txt" "results/$phase/unit_tests.txt"
cp "$root/tmp/getup_r95_unit_tests_20261001.txt" "results/$current/unit_tests.txt"
cp "$root/outputs/$current/identity_regression.json" "results/$current/identity_regression.json"
cd "$source_repo"
.venv/bin/python - "$publication" "$prior" "$phase" "$current" <<'PY'
import hashlib,json,shutil,time
from pathlib import Path
import sys
import numpy as np
repo=Path(sys.argv[1]);prior,phase,current=sys.argv[2:]
old=repo/'results'/prior/'final';second=repo/'results'/phase;new=repo/'results'/current
for d in (old,second):
    r=json.loads((d/'results.json').read_text())
    assert len(r['grid_results'])==25 and not r['qualification_run']
    assert r['training_best']['grid_index']==0
    assert r['grid_results'][0]['successes']==13
live=Path('/data/shijinsheng/open_duck/outputs')/current
if (live/'search_history.json').exists():
    for attempt in range(5):
        try:
            history=json.loads((live/'search_history.json').read_text())
            checkpoint=(live/'checkpoint.npz').read_bytes()
            import io
            with np.load(io.BytesIO(checkpoint),allow_pickle=False) as z:generation=int(z['r95_generation'])
            if generation==history[-1]['generation']:break
        except (json.JSONDecodeError,FileNotFoundError):pass
        time.sleep(.5)
    else:raise RuntimeError('Cannot obtain matching completed generation and checkpoint')
    snap=new/f'progress_gen{generation:03d}';snap.mkdir()
    (snap/'checkpoint.npz').write_bytes(checkpoint)
    (snap/'search_history.json').write_text(json.dumps(history,indent=2)+'\n')
    for gen in range(1,generation+1):
        for candidate in range(24):
            shutil.copytree(live/f'gen_{gen:03d}_candidate_{candidate:03d}',snap/f'gen_{gen:03d}_candidate_{candidate:03d}')
    (snap/'archive_stage.json').write_text(json.dumps({'snapshot_only':True,'generations_completed':generation,
        'qualification_not_established':True},indent=2)+'\n')
    print('R95_SNAPSHOT_GEN',generation,flush=True)
for directory in (old,second,new):
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}
    (directory/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('R93_R94_FINAL_CHECKED',flush=True)
PY
cd "$publication"
for file in GETUP_MICRO_CONTACT_R92_GYRO_R93_20261001.md GETUP_HEAD_RELATIVE_PHASE_R94_20261001.md GETUP_COORDINATED_PRELOAD_R95_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/probe_getup_head_phase_r94.py diagnostics/test_getup_head_phase_r94.py diagnostics/train_getup_preload_r95.py diagnostics/test_getup_preload_r95.py scripts/launch_getup_head_phase_r94_20261001.sh scripts/launch_getup_preload_r95_20261001.sh scripts/archive_getup_r93_r94_final_r95_started_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$prior/final" "results/$phase" "results/$current"
git -c gc.auto=0 commit -m "Reject gyro and relative head timing grids; train coordinated pre-contact preload R95" | tail -n 5
git bundle create "$root/tmp/getup_r93_r94_final_r95_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
