#!/usr/bin/env bash
# Simulation-only R75; preserve every existing experiment and controller.
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
run_dir=/data/shijinsheng/open_duck/outputs/getup_prefix_timing_probe_r75_20261001
checkpoint=/data/shijinsheng/open_duck/outputs/getup_prefix_pose_r73_left_20260930/checkpoint.npz
test -f "$checkpoint"
test ! -e "$run_dir"
test ! -e "$run_dir.log"
# Anchor to the executable, so the containing shell never matches itself.
if pgrep -u shijinsheng -f '^[^ ]*python[0-9.]* -u -m diagnostics\.search_getup_prefix_timing_r75([[:space:]]|$)' >/dev/null; then
    echo 'R75 already running; refusing duplicate launch.'
    exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
.venv/bin/python -m unittest \
    diagnostics.test_getup_prefix_timing_r75 \
    diagnostics.test_getup_wait_pose_r74 \
    diagnostics.test_getup_prefix_pose_r73 \
    diagnostics.test_getup_reference_feedback_r64 -v
nohup .venv/bin/python -u -m diagnostics.search_getup_prefix_timing_r75 \
    --checkpoint "$checkpoint" --output "$run_dir" --workers 6 \
    >"$run_dir.log" 2>&1 </dev/null &
run_pid=$!
echo "R75_PID=$run_pid"
sleep 2
ps -p "$run_pid" -o pid,etime,args
tail -n 6 "$run_dir.log"
