#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -u -m diagnostics.getup_anatomical_search \
  --output /data/shijinsheng/open_duck/training/getup_anatomical_r7_prone \
  --depth 10 --width 6 --workers 4 --seed 197
