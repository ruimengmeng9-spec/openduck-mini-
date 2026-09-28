#!/usr/bin/env bash
# Distinguish a warm stand->backward switch from cold home-pose startup.
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
for model in r11 r15; do
  if test "$model" = r11; then controller="$root/training/backward_contact_balance_r11/export_final"; else controller="$root/training/backward_capture_native_r15/export_probe"; fi
  .venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/stand_transition_r16_${model}_60s" --validate "$controller/controller_contract.json" --corrector-onnx "$controller/corrector.onnx" --duration 60 --seed-start 600 --seed-count 10 --workers 8 --stand-warmup-s 3
  .venv/bin/python -m diagnostics.summarize_phase_result "$root/outputs/stand_transition_r16_${model}_60s/results.json"
done
