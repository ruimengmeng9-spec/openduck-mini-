#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=13ec36f0da8d69c808a6b71c12437a1d617a91c7
previous=getup_early_feedback_r81_left_20261001
current=getup_sensor_clock_r82_left_20261001
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
cp "$root/tmp/getup_r82_unit_tests_20261001.txt" "results/$current/unit_tests.txt"
cp -r "$root/outputs/$current/executed_sources" "results/$current/executed_sources"
cd "$source_repo"
.venv/bin/python - "$publication" "$previous" "$current" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
repo = Path(sys.argv[1]); previous, current = sys.argv[2:]
old = repo / 'results' / previous / 'final'; new = repo / 'results' / current
r = json.loads((old / 'results.json').read_text())
assert r['heldout_successes'] == r['baseline_successes'] == 8
assert not r['paired_rescues'] and not r['paired_regressions']
with np.load(old / 'checkpoint.npz', allow_pickle=False) as z:
    assert not np.any(z['r81_early_gains'])
contract = json.loads((new / 'contract.json').read_text())
assert contract['r81_result_sha256'] == hashlib.sha256((old / 'results.json').read_bytes()).hexdigest()
assert not set(contract['heldout_seeds']) & set(r['heldout_seeds'])
for directory in (old, new):
    hashes = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in directory.rglob('*') if p.is_file()}
    (directory / 'artifact_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print('Archived negative R81 full qualification and R82 launch contract.')
PY
cd "$publication"
for file in GETUP_EARLY_FEEDBACK_R81_20261001.md GETUP_SENSOR_CLOCK_R82_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md diagnostics/probe_getup_sensor_clock_r82.py diagnostics/test_getup_sensor_clock_r82.py scripts/launch_getup_sensor_clock_r82_20261001.sh scripts/archive_getup_r81_final_r82_started_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$previous/final" "results/$current"
git commit -m "Archive negative R81 qualification and test causal sensor-clock alignment R82" | tail -n 5
git bundle create "$root/tmp/getup_r81_final_r82_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
