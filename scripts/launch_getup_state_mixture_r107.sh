#!/usr/bin/env bash
set -euo pipefail
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if pgrep -af '[d]iagnostics.train_getup_state_mixture_r107|[d]iagnostics.probe_getup_temporal_gate_r104|[d]iagnostics.train_getup_joint_anchor_r102' >/dev/null; then
  echo 'Existing experiment process; refusing duplicate launch'; exit 1
fi
output=/data/shijinsheng/open_duck/outputs/getup_state_mixture_r107_left_20261005
test ! -e "$output"
test ! -e "$output.log"
nohup .venv/bin/python -u -m diagnostics.train_getup_state_mixture_r107 --output "$output" --workers 6 --generations 24 --population 12 > "$output.log" 2>&1 < /dev/null &
echo "R107 started PID $!"
