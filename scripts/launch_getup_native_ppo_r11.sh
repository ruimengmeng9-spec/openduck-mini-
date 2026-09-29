#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -u -m diagnostics.train_getup_native_ppo_v2 \
  --precheck /data/shijinsheng/open_duck/training/getup_curriculum_precheck_r10_verified/results.json \
  --output /data/shijinsheng/open_duck/training/getup_native_ppo_r11_stage1 \
  --resume-params /data/shijinsheng/open_duck/training/getup_native_ppo_r10_stage1/final.msgpack \
  --iterations 256 --envs 8 --workers 4 --horizon 128 --tilt-max .55 --seed 312 \
  --initial-std .03 --reward-scale .05 --learning-rate .0001
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.eval_getup_native_ppo_v2 \
  --experiment /data/shijinsheng/open_duck/training/getup_native_ppo_r11_stage1 --seeds 20 --seed-base 74000 --tilt-max .55
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.eval_getup_native_ppo_v2 \
  --experiment /data/shijinsheng/open_duck/training/getup_native_ppo_r11_stage1 --seeds 20 --seed-base 74000 --tilt-max .25 --output-name evaluation_tilt025.json
