#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
experiment=/data/shijinsheng/open_duck/training/getup_native_ppo_r11_stage1
for attempt in {1..240}; do
  [[ -f "$experiment/training_summary.json" ]] && break
  sleep 5
done
[[ -f "$experiment/training_summary.json" ]] || { echo 'Training did not complete in validation wait window'; exit 1; }
.venv/bin/python -u -m diagnostics.audit_getup_native_checkpoint --experiment "$experiment"
.venv/bin/python -u -m diagnostics.select_getup_stage1_checkpoint --experiment "$experiment" --seeds 8 --stride 32
.venv/bin/python -u -m diagnostics.eval_getup_native_ppo_v2 \
  --experiment "$experiment/selected_candidate" --seeds 20 --seed-base 95000 --tilt-max .55
.venv/bin/python -u -m diagnostics.eval_getup_native_ppo_v2 \
  --experiment "$experiment/selected_candidate" --seeds 20 --seed-base 95000 --tilt-max .25 --output-name evaluation_tilt025.json
.venv/bin/python -u -m diagnostics.audit_getup_native_checkpoint --experiment "$experiment/selected_candidate"
