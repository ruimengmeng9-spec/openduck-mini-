#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
run_dir=/data/shijinsheng/open_duck/outputs/getup_early_head_r86_left_20261001
test ! -e "$run_dir"
test ! -e "$run_dir.log"
test -f /data/shijinsheng/open_duck/outputs/getup_substep_contact_audit_r85_20261001/results.json
if pgrep -u shijinsheng -f '^[^ ]*python[0-9.]* -u -m diagnostics\.[^ ]*getup' >/dev/null; then
    echo 'Existing get-up experiment; refusing duplicate launch.';exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
.venv/bin/python -m unittest diagnostics.test_getup_early_head_r86 diagnostics.test_getup_initial_settling_r84 \
    diagnostics.test_getup_early_pose_r80 diagnostics.test_getup_reference_feedback_r64 -v \
    >/data/shijinsheng/open_duck/tmp/getup_r86_unit_tests_20261001.txt 2>&1
tail -n 4 /data/shijinsheng/open_duck/tmp/getup_r86_unit_tests_20261001.txt
nohup flock -n /data/shijinsheng/open_duck/outputs/getup_r75_execution.lock \
    .venv/bin/python -u -m diagnostics.train_getup_early_head_r86 \
    --checkpoint /data/shijinsheng/open_duck/outputs/getup_prefix_pose_r73_left_20260930/checkpoint.npz \
    --output "$run_dir" --workers 6 --generations 24 --population 24 --seed 186 \
    >"$run_dir.log" 2>&1 </dev/null &
echo "R86_WRAPPER_PID=$!"
sleep 3
tail -n 5 "$run_dir.log"
