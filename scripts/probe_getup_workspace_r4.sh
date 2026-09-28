#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -u -m diagnostics.probe_getup_workspace_reference \
    --workspace /data/shijinsheng/open_duck/training/getup_workspace_r4 \
    --scene /data/shijinsheng/open_duck/training/getup_independent_r1_supine/model/scene.xml \
    --output /data/shijinsheng/open_duck/training/getup_workspace_reference_r4
