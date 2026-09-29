#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
for experiment in getup_residual_ppo_r18 getup_alignment_r20_base \
    getup_alignment_r20_height getup_range_r21; do
  [[ -f "/data/shijinsheng/open_duck/training/$experiment/checkpoint_audit.json" ]] || { echo 'Model audit missing'; exit 1; }
done
.venv/bin/python -u -m diagnostics.compare_getup_range_r21 \
  --output /data/shijinsheng/open_duck/training/getup_range_r21_comparison
