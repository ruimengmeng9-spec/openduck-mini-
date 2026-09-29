#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -u -m diagnostics.validate_getup_long_r20 \
  --base /data/shijinsheng/open_duck/training/getup_alignment_r20_base \
  --height /data/shijinsheng/open_duck/training/getup_alignment_r20_height \
  --previous /data/shijinsheng/open_duck/training/getup_residual_ppo_r18 \
  --output /data/shijinsheng/open_duck/training/getup_alignment_r20_long
