#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=df5919ab76646a219a45b7e3f776a44e8e592f44
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
library=getup_wait_library_r77_20261001
selector=getup_wait_selector_r78_20261001
test ! -e "results/$library/final"
test ! -e "results/$selector"
test -f "$root/outputs/$library/results.json"
test -f "$root/outputs/$selector/results.json"
mkdir "results/$library/final" "results/$selector"
cp -r "$root/outputs/$library/." "results/$library/final/"
cp -r "$root/outputs/$selector/." "results/$selector/"
cp "$root/outputs/$library.log" "results/$library/final/console.log"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
cd "$source_repo"
.venv/bin/python -m unittest diagnostics.test_getup_wait_selector_r78 -v \
    >"$publication/results/$selector/unit_tests.txt" 2>&1
.venv/bin/python - "$publication" "$library" "$selector" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
publication = Path(sys.argv[1])
library, selector = sys.argv[2:]
libdir = publication / 'results' / library / 'final'
fitdir = publication / 'results' / selector
lib = json.loads((libdir / 'results.json').read_text())
fit = json.loads((fitdir / 'results.json').read_text())
assert len(lib['results']) == 18 and lib['selected_wait_s'] == 2.
assert lib['baseline_successes'] == lib['selected_training_successes'] == 13
assert lib['library_training_oracle_successes'] == 16
assert fit['selected']['cv_successes'] == fit['baseline_training_successes'] == 13
assert not fit['needs_prospective_fullpath_validation']
assert fit['heldout_seeds_used'] == lib['heldout_seeds_used'] == []
assert fit['library_result_sha256'] == hashlib.sha256((libdir / 'results.json').read_bytes()).hexdigest()
for directory in (libdir, fitdir):
    manifest = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file()}
    (directory / 'artifact_hashes.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('R77 fixed wait 13/24, retrospective union 16/24; R78 training CV 13/24. No independent skill pass.')
PY
cd "$publication"
for file in GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md GETUP_WAIT_LIBRARY_R77_20261001.md diagnostics/fit_getup_wait_selector_r78.py diagnostics/test_getup_wait_selector_r78.py scripts/archive_getup_r77_r78_finished_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$library/final" "results/$selector"
git diff --cached --stat | tail -n 5
git commit -m "Preserve completed R77 wait library and negative R78 sensor selector fitting" | tail -n 5
git bundle create "$root/tmp/getup_r77_r78_finished_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
