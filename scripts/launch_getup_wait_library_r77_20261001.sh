#!/usr/bin/env bash
# Simulation only; no existing experiment or hardware controller is modified.
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
run_dir=/data/shijinsheng/open_duck/outputs/getup_wait_library_r77_20261001
checkpoint=/data/shijinsheng/open_duck/outputs/getup_prefix_pose_r73_left_20260930/checkpoint.npz
test -f "$checkpoint"
test ! -e "$run_dir"
test ! -e "$run_dir.log"
if pgrep -u shijinsheng -f '^[^ ]*python[0-9.]* -u -m diagnostics\.[^ ]*getup' >/dev/null; then
    echo 'Another get-up experiment is running; refusing duplicate launch.'
    exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
test_log=/data/shijinsheng/open_duck/tmp/getup_r77_unit_tests_20261001.txt
.venv/bin/python -m unittest \
    diagnostics.test_getup_wait_library_r77 \
    diagnostics.test_getup_prefix_timing_r75 \
    diagnostics.test_getup_wait_pose_r74 \
    diagnostics.test_getup_prefix_pose_r73 \
    diagnostics.test_getup_reference_feedback_r64 -v >"$test_log" 2>&1
# Share the earlier execution guard; a stale file is not an active lock.
nohup flock -n /data/shijinsheng/open_duck/outputs/getup_r75_execution.lock \
    .venv/bin/python -u -m diagnostics.probe_getup_wait_library_r77 \
    --checkpoint "$checkpoint" --output "$run_dir" --workers 6 \
    >"$run_dir.log" 2>&1 </dev/null &
echo "R77_WRAPPER_PID=$!"
sleep 3
tail -n 8 "$run_dir.log"
