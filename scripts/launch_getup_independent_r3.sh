#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
for pose in prone supine; do
    .venv/bin/python -u -m diagnostics.getup_beam_reference \
        --pose "$pose" --depth 7 --width 8 --workers 4 --duration .8 --seed 94 \
        --output "/data/shijinsheng/open_duck/training/getup_independent_r3_${pose}"
done
