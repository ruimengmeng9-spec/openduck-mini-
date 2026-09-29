#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -u -m diagnostics.train_getup_native_ppo \
  --precheck /data/shijinsheng/open_duck/training/getup_curriculum_precheck_r10_verified/results.json \
  --output /data/shijinsheng/open_duck/training/getup_native_ppo_r10_stage1 \
  --iterations 64 --envs 8 --workers 4 --horizon 128 --tilt-max .25 --seed 311
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.eval_getup_native_ppo \
  --experiment /data/shijinsheng/open_duck/training/getup_native_ppo_r10_stage1 --seeds 20 --tilt-max .25
