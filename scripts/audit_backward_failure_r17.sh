#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
for model in r11 r15; do
  if test "$model" = r11; then controller="$root/training/backward_contact_balance_r11/export_final"; seed=117; else controller="$root/training/backward_capture_native_r15/export_final"; seed=306; fi
  out="$root/outputs/actuator_audit_r17_$model"
  .venv/bin/python -u -m diagnostics.backward_phase_search --output "$out" --validate "$controller/controller_contract.json" --corrector-onnx "$controller/corrector.onnx" --duration 60 --seed-start "$seed" --seed-count 1
  .venv/bin/python -m diagnostics.audit_backward_execution_trace "$out/seed_$seed.npz" "$out/actuator_audit.json"
done
