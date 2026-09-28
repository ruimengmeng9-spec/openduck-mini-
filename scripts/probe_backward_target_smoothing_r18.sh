#!/usr/bin/env bash
# No extra torque/speed and no friction change; test reachability-aware filtering.
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
controller="$root/training/backward_contact_balance_r11/export_final"
for tau in 0.01 0.02 0.04 0.08; do
  .venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/target_smoothing_r18_tau${tau}_60s" --validate "$controller/controller_contract.json" --corrector-onnx "$controller/corrector.onnx" --duration 60 --seed-start 110 --seed-count 10 --workers 8 --target-tau-s "$tau"
  .venv/bin/python -m diagnostics.summarize_phase_result "$root/outputs/target_smoothing_r18_tau${tau}_60s/results.json"
done
