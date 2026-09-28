#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -u -m diagnostics.getup_crouch_extension \
    --scene /data/shijinsheng/open_duck/training/getup_decomposed_r4/model/scene.xml \
    --prefix /data/shijinsheng/open_duck/training/getup_dynamic_r4b_prone/best_reference.npz \
    --output /data/shijinsheng/open_duck/training/getup_crouch_r5_prone \
    --depth 12 --width 6 --workers 4 --seed 162
