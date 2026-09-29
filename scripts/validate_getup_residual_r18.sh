#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
experiment=${1:-/data/shijinsheng/open_duck/training/getup_residual_ppo_r18}
[[ -f "$experiment/training_summary.json" ]] || { echo 'Training has not finished'; exit 1; }
# Each program refuses to overwrite existing results. Use a new experiment
# directory for a reproduction rather than deleting prior evidence.
.venv/bin/python -u -m diagnostics.select_getup_residual_r18 --experiment "$experiment" --stride 16 --seeds 8
.venv/bin/python -u -m diagnostics.audit_getup_residual_r18 --experiment "$experiment/selected_candidate"
.venv/bin/python -u -m diagnostics.eval_getup_macro_r15 \
  --experiment "$experiment/selected_candidate" --seed-base 230000 --seeds 20 --repeat 5 --tilt-max .55
.venv/bin/python -u -m diagnostics.eval_getup_macro_r15 \
  --experiment "$experiment/selected_candidate" --seed-base 230000 --seeds 20 --repeat 1 --tilt-max .55 --output-name evaluation_50hz.json
.venv/bin/python -u -m diagnostics.eval_getup_macro_r15 \
  --experiment "$experiment" --seed-base 240000 --seeds 100 --repeat 5 --tilt-max .55 --output-name evaluation_broad_100.json
