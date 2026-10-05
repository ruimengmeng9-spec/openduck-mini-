#!/usr/bin/env bash
set -euo pipefail
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
run=/data/shijinsheng/open_duck/outputs/getup_nominal_anchor_r101b_left_20261005
test ! -e "$run"
test ! -e "$run.log"
if pgrep -u "$(id -u)" -f '^.venv/bin/python -u -m diagnostics.probe_getup_anchor_r101' >/dev/null; then
  echo 'Previous anchor process still alive; refusing overlap';exit 1
fi
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup .venv/bin/python -u -m diagnostics.probe_getup_anchor_r101b --output "$run" --workers 4 >"$run.log" 2>&1 </dev/null &
echo "R101b anchor probe started PID $!"
