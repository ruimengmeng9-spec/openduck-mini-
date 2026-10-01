#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=5c89871eaae555ed674647af6ddfd64a51931880
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
cd "$source_repo"
.venv/bin/python - "$publication" <<'PY'
from io import BytesIO
import hashlib,json,time
from pathlib import Path
import sys
import numpy as np
repo=Path(sys.argv[1]);live=Path('/data/shijinsheng/open_duck/outputs/getup_head_feedback_r87_left_20261001')
for attempt in range(5):
    raw=(live/'checkpoint.npz').read_bytes()
    with np.load(BytesIO(raw),allow_pickle=False) as z:
        generation=int(z['r87_generation']);gain_max=float(abs(z['r87_head_gains']).max())
    try:
        history=json.loads((live/'search_history.json').read_text())
        if len(history)>=generation:break
    except json.JSONDecodeError:pass
    time.sleep(.2)
else:raise RuntimeError('No coherent live checkpoint/history')
dest=repo/'results/getup_head_feedback_r87_left_20261001'/f'progress_gen{generation:03d}'
dest.mkdir(exist_ok=False)
(dest/'checkpoint.npz').write_bytes(raw)
(dest/'search_history.json').write_text(json.dumps(history[:generation],indent=2)+'\n')
summary={'snapshot_only':True,'simulation_only':True,'hardware_readiness':False,
         'generation':generation,'training_successes':history[generation-1]['best']['successes'],
         'nominal_success':history[generation-1]['best']['nominal_success'],
         'maximum_head_gain':gain_max,'qualification_pending':True,
         'checkpoint_sha256':hashlib.sha256(raw).hexdigest()}
(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()}
(dest/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps(summary),flush=True)
PY
cd "$publication"
cp "$source_repo/GETUP_HEAD_FEEDBACK_R87_20261001.md" .
cp "$source_repo/scripts/archive_getup_r87_progress_20261001.sh" scripts/
git add GETUP_HEAD_FEEDBACK_R87_20261001.md scripts/archive_getup_r87_progress_20261001.sh
git add -f results/getup_head_feedback_r87_left_20261001/progress_gen*
git commit -m "Save first live head-feedback checkpoint without claiming qualification" | tail -n 5
git bundle create "$root/tmp/getup_r87_progress_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
