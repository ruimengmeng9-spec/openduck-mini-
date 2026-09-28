#!/usr/bin/env bash
# Explicit privileged-state simulation experiment, not a deployable gait claim.
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
ROOT=/data/shijinsheng/open_duck
cd "$ROOT/projects/Open_Duck_Playground"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
INNER="$ROOT/training/backward_contact_balance_r11/export_probe"
.venv/bin/python -m unittest diagnostics.test_backward_phase_search -v
for PATH_K in 0.3 0.6 1.0; do
  CONTRACT="$ROOT/training/backward_path_r13/k${PATH_K}.json"
  .venv/bin/python -m diagnostics.make_backward_path_probe \
    --controller "$INNER/controller_contract.json" --output "$CONTRACT" --gain "$PATH_K"
  .venv/bin/python -u -m diagnostics.backward_phase_search \
    --validate "$CONTRACT" --corrector-onnx "$INNER/corrector.onnx" \
    --output "$ROOT/outputs/path_r13_k${PATH_K}_60s" \
    --seed-start 60 --seed-count 5 --duration 60 --workers 4 \
    > "$ROOT/tmp/path_r13_k${PATH_K}_60s.log" 2>&1
  .venv/bin/python -m diagnostics.summarize_phase_result \
    "$ROOT/outputs/path_r13_k${PATH_K}_60s/results.json"
done
