#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
experiment=/data/shijinsheng/open_duck/training/getup_residual_ppo_r18
.venv/bin/python -u -m diagnostics.train_getup_residual_r18 \
  --precheck /data/shijinsheng/open_duck/training/getup_curriculum_precheck_r10_verified/results.json \
  --output "$experiment" --iterations 128 --envs 8 --workers 4 --horizon 64 \
  --tilt-max .55 --seed 418 --repeat 5 --initial-std .3 --residual-scale .15 --reward-scale .01
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.audit_getup_residual_r18 --experiment "$experiment"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.eval_getup_macro_r15 \
  --experiment "$experiment" --seeds 20 --seed-base 210000 --repeat 5 --tilt-max .55
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.eval_getup_macro_r15 \
  --experiment "$experiment" --seeds 20 --seed-base 210000 --repeat 1 --tilt-max .55 --output-name evaluation_50hz.json
