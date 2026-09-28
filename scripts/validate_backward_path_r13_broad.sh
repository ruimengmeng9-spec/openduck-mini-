#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/path_r13_k0.6_broad_60s" --validate "$root/training/backward_path_r13/k0.6.json" --corrector-onnx "$root/training/backward_contact_balance_r11/export_probe/corrector.onnx" --duration 60 --seed-start 400 --seed-count 50 --workers 8
.venv/bin/python -m diagnostics.summarize_phase_result "$root/outputs/path_r13_k0.6_broad_60s/results.json"
