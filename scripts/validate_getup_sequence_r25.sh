#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m unittest diagnostics.test_getup_sequence_r24 diagnostics.test_getup_rescue_r22 -v
.venv/bin/python -u -m diagnostics.validate_getup_sequence_r25 \
  --experiment /data/shijinsheng/open_duck/training/getup_transfer_r25 \
  --output /data/shijinsheng/open_duck/training/getup_sequence_r25_broad
