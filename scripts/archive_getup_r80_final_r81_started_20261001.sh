#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=df3856c56ad48207e0c76e571d4e0c44f4ff07b3
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
previous=getup_early_pose_r80_left_20261001
current=getup_early_feedback_r81_left_20261001
test -f "$root/outputs/$previous/results.json"
test -f "$root/outputs/$current/contract.json"
test ! -e "results/$previous/final"
test ! -e "results/$current"
mkdir "results/$previous/final" "results/$current"
cp -r "$root/outputs/$previous/." "results/$previous/final/"
cp "$root/outputs/$previous.log" "results/$previous/final/console.log"
cp "$root/outputs/$current/contract.json" "results/$current/contract.json"
cp "$root/tmp/getup_r81_unit_tests_20261001.txt" "results/$current/unit_tests.txt"
cp -r "$root/outputs/$current/executed_sources" "results/$current/executed_sources"
cd "$source_repo"
.venv/bin/python - "$publication" "$previous" "$current" <<'PY'
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
repo = Path(sys.argv[1]); previous, current = sys.argv[2:]
old = repo / 'results' / previous / 'final'
new = repo / 'results' / current
r = json.loads((old / 'results.json').read_text())
assert r['heldout_successes'] == r['baseline_successes'] == 13
assert not r['paired_rescues'] and not r['paired_regressions']
with np.load(old / 'checkpoint.npz', allow_pickle=False) as z:
    assert not np.any(z['r80_early_delta'])
contract = json.loads((new / 'contract.json').read_text())
assert contract['r80_result_sha256'] == hashlib.sha256((old / 'results.json').read_bytes()).hexdigest()
assert not set(contract['heldout_seeds']) & set(r['heldout_seeds'])
live = Path('/data/shijinsheng/open_duck/outputs') / current
if (live / 'checkpoint.npz').exists():
    for attempt in range(5):
        raw = (live / 'checkpoint.npz').read_bytes()
        with np.load(BytesIO(raw), allow_pickle=False) as z:
            generation = int(z['r81_generation'])
        try:
            history = json.loads((live / 'search_history.json').read_text())
            if len(history) >= generation: break
        except json.JSONDecodeError:
            pass
        time.sleep(.2)
    else:
        raise RuntimeError('Cannot preserve a coherent live checkpoint')
    snapshot = new / f'progress_gen{generation:03d}'; snapshot.mkdir()
    (snapshot / 'checkpoint.npz').write_bytes(raw)
    (snapshot / 'search_history.json').write_text(json.dumps(history[:generation], indent=2) + '\n')
    (snapshot / 'summary.json').write_text(json.dumps({'snapshot_only': True,
        'qualification_pending': True, 'generation': generation,
        'training_successes': history[generation-1]['best']['successes'],
        'nominal_success': history[generation-1]['best']['nominal_success']}, indent=2) + '\n')
for directory in (old, new):
    hashes = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in directory.rglob('*') if p.is_file()}
    (directory / 'artifact_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print('R80 negative 13/40 result, all failure traces, and R81 launch contract archived.')
PY
cd "$publication"
for file in GETUP_EARLY_POSE_R79_R80_20261001.md GETUP_EARLY_FEEDBACK_R81_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/train_getup_early_feedback_r81.py diagnostics/test_getup_early_feedback_r81.py scripts/launch_getup_early_feedback_r81_20261001.sh scripts/archive_getup_r80_final_r81_started_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$previous/final" "results/$current"
git commit -m "Preserve negative R80 qualification and start causal early IMU recovery feedback R81" | tail -n 5
git bundle create "$root/tmp/getup_r80_final_r81_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
