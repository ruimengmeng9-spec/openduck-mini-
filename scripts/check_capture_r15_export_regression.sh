#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
model="$root/training/backward_capture_native_r15/export_final"
.venv/bin/python -m unittest diagnostics.test_backward_phase_search diagnostics.test_capture_point_observer -v
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/capture_r15_export_training_regression" --validate "$model/controller_contract.json" --corrector-onnx "$model/corrector.onnx" --duration 60 --validation-seeds 0,21,31,38,57,117 --workers 8
.venv/bin/python -m diagnostics.compare_phase_results "$root/training/backward_capture_native_r15/best.json" "$root/outputs/capture_r15_export_training_regression/results.json"
