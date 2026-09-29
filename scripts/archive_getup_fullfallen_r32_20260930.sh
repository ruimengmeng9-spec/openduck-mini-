#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat PAGER=cat
runtime=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
publish=/data/shijinsheng/open_duck/github/openduck-mini-
training=/data/shijinsheng/open_duck/training
cd "$publish"
[[ -z "$(git status --porcelain)" ]] || { echo 'Publication checkout dirty; refusing'; exit 1; }
files=(GETUP_FULLFALLEN_PPO_R32_20260929.md GETUP_R32_RUNNING_20260930.md
  diagnostics/getup_fullfallen_contract_r32.py diagnostics/getup_fullfallen_env_r32.py
  diagnostics/train_getup_fullfallen_ppo_r32.py diagnostics/validate_getup_fullfallen_r32.py
  diagnostics/test_getup_fullfallen_r32.py scripts/launch_getup_fullfallen_r32.sh
  scripts/archive_getup_fullfallen_r32_20260930.sh)
archive=results/getup_fullfallen_r32_start_20260930
[[ ! -e "$archive" ]] || { echo 'Archive already exists'; exit 1; }
for file in "${files[@]}"; do
  [[ ! -e "$file" && -f "$runtime/$file" ]] || { echo "Refusing source overwrite or missing file: $file"; exit 1; }
done
for task in getup_fullpath_r27 getup_fullpath_r27_strict getup_contact_archive_r30 getup_bridge_r31; do
  [[ -d "$training/$task" ]] || { echo "Missing result $task"; exit 1; }
done
smoke="$training/getup_fullfallen_r32_20260930_003407_smoke"
[[ -f "$smoke/training_summary.json" ]] || { echo 'Smoke has not finished'; exit 1; }
mkdir -p "$archive"
for file in "${files[@]}"; do cp "$runtime/$file" "$file"; git add "$file"; done
for task in getup_fullpath_r27 getup_fullpath_r27_strict getup_contact_archive_r30 getup_bridge_r31; do
  cp -a "$training/$task" "$archive/"
  cp "$training/$task.log" "$archive/"
done
cp -a "$smoke" "$archive/"
"$runtime/.venv/bin/python" - <<'PY'
from pathlib import Path
import hashlib,json
archive=Path('results/getup_fullfallen_r32_start_20260930')
strict=json.loads((archive/'getup_fullpath_r27_strict/results.json').read_text())
assert not strict['full_task_completed']
smoke=json.loads((archive/'getup_fullfallen_r32_20260930_003407_smoke/training_summary.json').read_text())
assert smoke['completed_iterations']==2 and not smoke['full_task_completed']
assert all(p.stat().st_size<95*1024**2 for p in archive.rglob('*') if p.is_file())
manifest=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
    main_training_output='/data/shijinsheng/open_duck/training/getup_fullfallen_r32_20260930_003407',
    main_training_not_yet_terminal_at_archive=True,
    gate='>=18/20 independent starts per each of four full-fallen orientations; continuous 30s strict loaded standing',
    hashes={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file()})
(archive/'archive_manifest.json').write_text(json.dumps(manifest,indent=2))
PY
git add -f "$archive"
git diff --cached --check
git commit -m 'Train full-fallen closed-loop getup PPO; preserve four-pose search and bridge failures'
git bundle create /data/shijinsheng/open_duck/tmp/getup_fullfallen_r32_start_20260930.bundle 0f27e4a..main
git --no-pager log -1 --oneline
