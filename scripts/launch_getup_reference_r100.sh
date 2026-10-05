#!/usr/bin/env bash
set -euo pipefail
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
run=/data/shijinsheng/open_duck/outputs/getup_reference_residual_r100_left_20261005
test ! -e "$run"
test ! -e "$run.log"
if pgrep -u "$(id -u)" -f '^.venv/bin/python -u -m diagnostics.run_getup_reference_r100' >/dev/null; then
  echo 'Existing R100 runner; refusing duplicate'
  exit 1
fi
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nohup .venv/bin/python -u -m diagnostics.run_getup_reference_r100 --output "$run" --iterations 64 --workers 4 >"$run.log" 2>&1 </dev/null &
echo "R100 runner started PID $!"
