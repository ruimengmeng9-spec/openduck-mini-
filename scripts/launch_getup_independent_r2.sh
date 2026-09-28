#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
for pose in supine prone; do
    .venv/bin/python -u -m diagnostics.getup_feedback_reference \
        --pose "$pose" --generations 24 --population 96 --workers 4 \
        --knots 5 --duration 4 --seed 83 \
        --warm-reference "/data/shijinsheng/open_duck/training/getup_independent_r1_${pose}/best_reference.npz" \
        --output "/data/shijinsheng/open_duck/training/getup_independent_r2_${pose}"
done
