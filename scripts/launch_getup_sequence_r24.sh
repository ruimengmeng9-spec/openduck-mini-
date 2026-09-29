#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m unittest diagnostics.test_getup_sequence_r24 -v
.venv/bin/python -u -m diagnostics.train_getup_sequence_r24 \
  --contract /data/shijinsheng/open_duck/training/getup_alignment_r20_base/controller_contract.json \
  --search /data/shijinsheng/open_duck/training/getup_rescue_r22 /data/shijinsheng/open_duck/training/getup_rescue_r22_expanded \
  --output /data/shijinsheng/open_duck/training/getup_sequence_r24
