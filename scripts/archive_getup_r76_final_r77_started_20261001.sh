#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=718a53aae4b158e01023f76ec791c171565cecdb
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
finished=getup_prefix_timing_cem_r76_20261001
probe=getup_wait_library_r77_20261001
test -f "$root/outputs/$finished/results.json"
test -f "$root/outputs/$probe/contract.json"
test ! -e "results/$finished/final"
test ! -e "results/$probe"
mkdir "results/$finished/final" "results/$probe"
# Preserve the previously published running snapshots; final evidence is separate.
for file in "$root/outputs/$finished/"*; do
    test -f "$file"
    cp "$file" "results/$finished/final/"
done
cp "$root/outputs/$finished.log" "results/$finished/final/console.log"
cp "$root/outputs/getup_r76_resume_20261001_032348_706806/resume.json" "results/$finished/final/resume_record.json"
cp "$root/outputs/$probe/contract.json" "results/$probe/contract.json"
cp "$root/outputs/$probe.log" "results/$probe/launch_console_snapshot.log"
cp "$root/tmp/getup_r77_unit_tests_20261001.txt" "results/$probe/unit_tests.txt"
if test -f "$root/outputs/$probe/progress.json"; then
    cp "$root/outputs/$probe/progress.json" "results/$probe/launch_progress_snapshot.json"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
"$source_repo/.venv/bin/python" - "$root" "$publication" "$finished" "$probe" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
root, publication = map(Path, sys.argv[1:3])
finished, probe = sys.argv[3:5]
final = publication / 'results' / finished / 'final'
result = json.loads((final / 'results.json').read_text())
assert result['generations_completed'] == 13
assert result['training_best']['successes'] == 13
assert result['baseline_successes'] == result['heldout_successes'] == 13
assert len(result['heldout']) == 40
assert all(r['baseline'] == r['candidate'] for r in result['heldout'])
assert sum(r['candidate']['valid'] for r in result['heldout']) == 38
assert result['required_strict_seconds'] == 30
assert not set(result['training_seeds']) & set(result['heldout_seeds'])
assert len(list(final.glob('baseline_*.npz'))) == 41
assert len(list(final.glob('candidate_*.npz'))) == 41
contract = json.loads((root / 'outputs' / probe / 'contract.json').read_text())
assert contract['training_only'] and contract['heldout_seeds_used'] == []
assert contract['no_midpath_state_reset'] and not contract['hardware_readiness']
assert contract['source_sha256'] == hashlib.sha256((root / 'projects/Open_Duck_Playground/diagnostics/probe_getup_wait_library_r77.py').read_bytes()).hexdigest()
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in final.iterdir() if p.is_file()}
(final / 'artifact_hashes.json').write_text(json.dumps(manifest, indent=2) + '\n')
snapshot = {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'launched_training_only_diagnostic_not_final_validation',
            'run_directory': str(root / 'outputs' / probe),
            'source_sha256': contract['source_sha256'],
            'full_task_completed': False, 'hardware_readiness': False}
(publication / 'results' / probe / 'launch_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
print(json.dumps({'r76_independent': '13/40 baseline 13/40', 'r77': snapshot}, indent=2))
PY
for file in GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md GETUP_WAIT_LIBRARY_R77_20261001.md diagnostics/probe_getup_wait_library_r77.py diagnostics/test_getup_wait_library_r77.py scripts/launch_getup_wait_library_r77_20261001.sh scripts/archive_getup_r76_final_r77_started_20261001.sh scripts/resume_getup_r76_20261001.py; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$finished/final" "results/$probe"
git diff --cached --stat
git commit -m "Archive completed R76 full-fall evidence and start R77 wait library probe"
git bundle create "$root/tmp/getup_r76_final_r77_started_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
