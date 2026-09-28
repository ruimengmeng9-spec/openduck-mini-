#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
for option in direct aligned; do
  extra=()
  if test "$option" = aligned; then extra=(--align-double-support); fi
  .venv/bin/python -u -m diagnostics.backward_skill_sequence --contract "$root/training/backward_target_smoothing_r18/tau0.01.json" --model "$root/training/backward_contact_balance_r11/export_final/corrector.onnx" --output "$root/outputs/backward_stop_blend_r21_$option" --seed-start 1000 --seed-count 20 --workers 8 --stop-blend-s 1 "${extra[@]}"
  .venv/bin/python -m diagnostics.summarize_backward_sequence "$root/outputs/backward_stop_blend_r21_$option/results.json"
done
