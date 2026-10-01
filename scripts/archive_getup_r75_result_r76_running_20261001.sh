#!/usr/bin/env bash
# Archive finished R75 evidence and a clearly labelled running R76 snapshot.
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=7160cefe0de5cd59388119fb88534f600ca147df
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
cmp diagnostics/search_getup_prefix_timing_r75.py "$source_repo/diagnostics/search_getup_prefix_timing_r75.py"
cmp diagnostics/test_getup_prefix_timing_r75.py "$source_repo/diagnostics/test_getup_prefix_timing_r75.py"
probe=getup_prefix_timing_probe_r75_20261001
train=getup_prefix_timing_cem_r76_20261001
test -f "$root/outputs/$probe/results.json"
test -f "$root/outputs/$train/contract.json"
test ! -e "$root/outputs/$train/results.json"
test ! -e "results/$probe"
test ! -e "results/$train"
mkdir -p "results/$probe" "results/$train"
for file in contract.json results.json selected.npz; do
    cp "$root/outputs/$probe/$file" "results/$probe/$file"
done
cp "$root/outputs/$probe.log" "results/$probe/console.log"
cp "$root/outputs/$train/contract.json" "results/$train/contract.json"
cp "$root/outputs/$train/search_history.json" "results/$train/running_history_snapshot.json"
cp "$root/outputs/$train.log" "results/$train/running_console_snapshot.log"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
"$source_repo/.venv/bin/python" -m unittest \
    diagnostics.test_getup_prefix_timing_r75 \
    diagnostics.test_getup_wait_pose_r74 \
    diagnostics.test_getup_prefix_pose_r73 \
    diagnostics.test_getup_reference_feedback_r64 -v \
    >"results/$probe/unit_tests.txt" 2>&1
"$source_repo/.venv/bin/python" - "$root" "$publication" "$probe" "$train" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
root, publication = map(Path, sys.argv[1:3])
probe, train = sys.argv[3:5]
source = root / 'projects/Open_Duck_Playground/diagnostics/search_getup_prefix_timing_r75.py'
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
result = json.loads((publication / 'results' / probe / 'results.json').read_text())
contract = json.loads((publication / 'results' / train / 'contract.json').read_text())
assert result['source_sha256'] == contract['source_sha256'] == source_hash
assert result['selected']['factors'] == [1., 1., 1., 1.]
assert result['baseline']['successes'] == result['selected']['successes'] == 13
assert len(result['results']) == 15
assert result['training_only'] and not contract['training_only']
assert contract['required_strict_seconds'] == 30 and contract['hold_seconds'] == 35
assert not set(contract['training_seeds']) & set(contract['heldout_seeds'])
history = json.loads((publication / 'results' / train / 'running_history_snapshot.json').read_text())
snapshot = {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'running_snapshot_not_final_qualification',
            'source_sha256': source_hash,
            'generations_in_snapshot': len(history),
            'best_training_successes': history[-1]['best']['successes'],
            'nominal_success': history[-1]['best']['nominal_success'],
            'full_task_completed': False, 'hardware_readiness': False}
(publication / 'results' / train / 'running_snapshot.json').write_text(
    json.dumps(snapshot, indent=2) + '\n')
print(json.dumps(snapshot, indent=2))
PY
for file in GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md GETUP_EARLY_TIMING_R75_R76_20261001.md; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
cp "$source_repo/scripts/launch_getup_prefix_timing_r75_20261001.sh" scripts/
cp "$source_repo/scripts/archive_getup_r75_result_r76_running_20261001.sh" scripts/
git add scripts/launch_getup_prefix_timing_r75_20261001.sh scripts/archive_getup_r75_result_r76_running_20261001.sh
git add -f "results/$probe" "results/$train"
git diff --cached --stat
git commit -m "Archive negative R75 timing probe and record existing R76 training"
git bundle create "$root/tmp/getup_r75_result_r76_running_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
