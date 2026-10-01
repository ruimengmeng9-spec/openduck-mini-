#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=53e76d5323b0d817acfc416b4c0fca81a7e7e8a0
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
audit=getup_failure_audit_r79_20261001
training=getup_early_pose_r80_left_20261001
test ! -e "results/$audit"
test ! -e "results/$training"
test -f "$root/outputs/$audit/results.json"
test -f "$root/outputs/$training/contract.json"
mkdir "results/$audit" "results/$training"
cp "$root/outputs/$audit/results.json" "results/$audit/results.json"
cp "$root/outputs/$training/contract.json" "results/$training/contract.json"
cp "$root/tmp/getup_r80_unit_tests_20261001.txt" "results/$training/unit_tests.txt"
cp -r "$root/outputs/$training/executed_sources" "results/$training/executed_sources"
cd "$source_repo"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
    .venv/bin/python -m unittest diagnostics.test_getup_wait_failures_r79 -v \
    >"$publication/results/$audit/unit_tests.txt" 2>&1
.venv/bin/python - "$publication" "$audit" "$training" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
repo = Path(sys.argv[1])
for name in sys.argv[2:]:
    directory = repo / 'results' / name
    manifest = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file()}
    (directory / 'artifact_hashes.json').write_text(json.dumps(manifest, indent=2) + '\n')
audit = json.loads((repo / 'results' / sys.argv[2] / 'results.json').read_text())
contract = json.loads((repo / 'results' / sys.argv[3] / 'contract.json').read_text())
assert contract['r79_result_sha256'] == hashlib.sha256((repo / 'results' / sys.argv[2] / 'results.json').read_bytes()).hexdigest()
assert audit['simulation_only'] and audit['no_new_rollout_or_training']
assert contract['simulation_only'] and not contract['hardware_readiness']
assert not set(contract['training_seeds']) & set(contract['heldout_seeds'])
print('Archive contract verified; R80 still running, no qualification claimed.')
PY
cd "$publication"
for file in GETUP_EARLY_POSE_R79_R80_20261001.md diagnostics/audit_getup_wait_failures_r79.py diagnostics/test_getup_wait_failures_r79.py diagnostics/train_getup_early_pose_r80.py diagnostics/test_getup_early_pose_r80.py scripts/launch_getup_early_pose_r80_20261001.sh scripts/archive_getup_r79_r80_started_20261001.sh; do
    test ! -e "$file"
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$audit" "results/$training"
git commit -m "Archive R79 early failure evidence and launch R80 early-pose recovery fitting" | tail -n 5
git bundle create "$root/tmp/getup_r79_r80_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
