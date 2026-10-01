#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
run_dir=/data/shijinsheng/open_duck/outputs/getup_gyro_attenuation_r93_left_20261001
test ! -e "$run_dir"
test ! -e "$run_dir.log"
test -f /data/shijinsheng/open_duck/outputs/getup_micro_contact_r92_20261001/results.json
if pgrep -u shijinsheng -f '^[^ ]*python[0-9.]* -u -m diagnostics\.[^ ]*getup' >/dev/null; then
    echo 'Another get-up experiment is live; refusing duplicate launch.';exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
.venv/bin/python -m unittest diagnostics.test_getup_gyro_attenuation_r93 diagnostics.test_getup_micro_contact_r92 \
    diagnostics.test_getup_head_tolerance_r91 diagnostics.test_getup_centered_head_r90 \
    diagnostics.test_getup_head_distillation_r89 diagnostics.test_getup_head_selector_r88 \
    diagnostics.test_getup_head_feedback_r87 -v > /data/shijinsheng/open_duck/tmp/getup_r93_unit_tests_20261001.txt 2>&1
tail -n 4 /data/shijinsheng/open_duck/tmp/getup_r93_unit_tests_20261001.txt
nohup flock -n /data/shijinsheng/open_duck/outputs/getup_r75_execution.lock \
    .venv/bin/python -u -m diagnostics.train_getup_gyro_attenuation_r93 \
    --checkpoint /data/shijinsheng/open_duck/outputs/getup_prefix_pose_r73_left_20260930/checkpoint.npz \
    --output "$run_dir" --workers 6 >"$run_dir.log" 2>&1 </dev/null &
echo "R93_WRAPPER_PID=$!"
sleep 3
tail -n 5 "$run_dir.log"
