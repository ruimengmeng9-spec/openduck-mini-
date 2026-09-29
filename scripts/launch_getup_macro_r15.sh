#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
experiment=/data/shijinsheng/open_duck/training/getup_macro_ppo_r15
.venv/bin/python -u -m diagnostics.train_getup_macro_ppo_r15 \
  --precheck /data/shijinsheng/open_duck/training/getup_curriculum_precheck_r10_verified/results.json \
  --output "$experiment" --resume-params /data/shijinsheng/open_duck/training/getup_native_ppo_r11_stage1/selected_candidate/final.msgpack \
  --iterations 256 --envs 8 --workers 4 --horizon 64 --tilt-max .55 --seed 415 --repeat 5 --initial-std .1 --reward-scale .01
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.audit_getup_native_checkpoint --experiment "$experiment"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.eval_getup_macro_r15 \
  --experiment "$experiment" --seeds 20 --seed-base 180000 --repeat 5 --tilt-max .55
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.eval_getup_macro_r15 \
  --experiment "$experiment" --seeds 20 --seed-base 180000 --repeat 1 --tilt-max .55 --output-name evaluation_50hz.json
