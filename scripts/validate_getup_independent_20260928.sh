#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
for run in r1_supine r1_prone r2_supine r2_prone r3_supine r3_prone; do
    .venv/bin/python -m diagnostics.validate_getup_independent \
        --experiment "/data/shijinsheng/open_duck/training/getup_independent_$run"
done
