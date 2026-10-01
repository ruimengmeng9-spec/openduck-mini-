#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=7e6c7cc5de8fbcefe41e76de75087d6475330ca5
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
cd "$source_repo"
.venv/bin/python - "$publication" <<'PY'
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
repo = Path(sys.argv[1])
live = Path('/data/shijinsheng/open_duck/outputs/getup_early_feedback_r81_left_20261001')
for attempt in range(5):
    raw = (live / 'checkpoint.npz').read_bytes()
    with np.load(BytesIO(raw), allow_pickle=False) as z:
        generation = int(z['r81_generation'])
        gain_max = float(np.abs(z['r81_early_gains']).max())
    try:
        history = json.loads((live / 'search_history.json').read_text())
        if len(history) >= generation: break
    except json.JSONDecodeError:
        pass
    time.sleep(.2)
else:
    raise RuntimeError('No coherent live checkpoint/history; leave live files untouched')
history = history[:generation]
assert history[-1]['generation'] == generation
dest = repo / 'results/getup_early_feedback_r81_left_20261001' / f'progress_gen{generation:03d}'
dest.mkdir(exist_ok=False)
(dest / 'checkpoint.npz').write_bytes(raw)
(dest / 'search_history.json').write_text(json.dumps(history, indent=2) + '\n')
summary = {'snapshot_only': True, 'simulation_only': True, 'hardware_readiness': False,
           'generation': generation, 'training_successes': history[-1]['best']['successes'],
           'nominal_success': history[-1]['best']['nominal_success'],
           'maximum_learned_gain': gain_max, 'qualification_pending': True,
           'checkpoint_sha256': hashlib.sha256(raw).hexdigest()}
(dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()}
(dest / 'artifact_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(json.dumps(summary), flush=True)
PY
cd "$publication"
cp "$source_repo/scripts/archive_getup_r81_progress_20261001.sh" scripts/
git add scripts/archive_getup_r81_progress_20261001.sh
git add -f results/getup_early_feedback_r81_left_20261001/progress_gen*
git commit -m "Preserve live R81 sensor-feedback checkpoint without claiming qualification" | tail -n 5
git bundle create "$root/tmp/getup_r81_progress_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
