#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=85c3c0945765748d908765eb700b9ed0ae107f8d
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
cd "$source_repo"
.venv/bin/python - "$publication" <<'PY'
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time
import numpy as np
repo = Path(sys.argv[1])
source = Path('/data/shijinsheng/open_duck/outputs/getup_early_pose_r80_left_20261001')
for attempt in range(5):
    raw = (source / 'checkpoint.npz').read_bytes()
    with np.load(BytesIO(raw), allow_pickle=False) as z:
        generation = int(z['r80_generation'])
    try:
        history = json.loads((source / 'search_history.json').read_text())
        if len(history) >= generation:
            break
    except json.JSONDecodeError:
        pass
    time.sleep(.2)
else:
    raise RuntimeError('No coherent checkpoint/history snapshot; keep live files untouched')
history = history[:generation]
assert history[-1]['generation'] == generation
destination = repo / 'results/getup_early_pose_r80_left_20261001' / f'progress_gen{generation:03d}'
destination.mkdir(exist_ok=False)
(destination / 'checkpoint.npz').write_bytes(raw)
(destination / 'search_history.json').write_text(json.dumps(history, indent=2) + '\n')
summary = {'snapshot_only': True, 'simulation_only': True, 'hardware_readiness': False,
           'generation': generation, 'training_successes': history[-1]['best']['successes'],
           'nominal_success': history[-1]['best']['nominal_success'],
           'qualification_pending': True, 'source_run': str(source),
           'checkpoint_sha256': hashlib.sha256(raw).hexdigest()}
(destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}
(destination / 'artifact_hashes.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(summary), flush=True)
PY
cd "$publication"
cp "$source_repo/GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md" .
cp "$source_repo/scripts/archive_getup_r80_progress_20261001.sh" scripts/
git add GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md scripts/archive_getup_r80_progress_20261001.sh
git add -f results/getup_early_pose_r80_left_20261001/progress_gen*
git commit -m "Preserve live R80 training checkpoint and update continuation handoff" | tail -n 5
git bundle create "$root/tmp/getup_r80_progress_20261001.bundle" HEAD '^53e76d5323b0d817acfc416b4c0fca81a7e7e8a0'
git rev-parse HEAD
