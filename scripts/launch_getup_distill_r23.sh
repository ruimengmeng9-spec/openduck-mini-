#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
training=/data/shijinsheng/open_duck/training
[[ -f "$training/getup_rescue_r22/results.json" && -f "$training/getup_rescue_r22_expanded/results.json" ]] || { echo 'Teacher search incomplete'; exit 1; }
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -m unittest diagnostics.test_getup_rescue_r22 diagnostics.test_getup_distill_r23 -v
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -u -m diagnostics.distill_getup_rescue_r23 \
  --contract "$training/getup_alignment_r20_base/controller_contract.json" \
  --search "$training/getup_rescue_r22" "$training/getup_rescue_r22_expanded" \
  --output "$training/getup_distill_r23" --epochs 200
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -m diagnostics.audit_getup_residual_r18 --experiment "$training/getup_distill_r23"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.eval_getup_distill_r23 --experiment "$training/getup_distill_r23"
