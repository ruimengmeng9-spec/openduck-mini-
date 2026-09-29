#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
mode=${1:?Specify base or height}
gpu=${2:?Specify allocated GPU}
[[ "$mode" == base || "$mode" == height ]] || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
experiment=/data/shijinsheng/open_duck/training/getup_alignment_r20_${mode}
.venv/bin/python -u -m diagnostics.train_getup_alignment_r20 \
  --precheck /data/shijinsheng/open_duck/training/getup_curriculum_precheck_r10_verified/results.json \
  --output "$experiment" --iterations 128 --envs 8 --workers 4 --horizon 64 \
  --tilt-max .55 --seed 420 --repeat 5 --initial-std .03 --residual-scale .15 \
  --reward-scale .01 --reward-mode "$mode"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.audit_getup_residual_r18 --experiment "$experiment"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.eval_getup_macro_r15 \
  --experiment "$experiment" --seeds 100 --seed-base 250000 --repeat 5 --tilt-max .55 --output-name evaluation_100.json
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.eval_getup_macro_r15 \
  --experiment "$experiment" --seeds 20 --seed-base 260000 --repeat 1 --tilt-max .55 --output-name evaluation_50hz.json
