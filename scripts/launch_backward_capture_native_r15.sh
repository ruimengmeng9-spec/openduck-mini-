#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
out="$root/training/backward_capture_native_r15"
test ! -e "$out/search.json"
.venv/bin/python -m diagnostics.prepare_capture_native_base "$root/training/backward_capture_r14/capture_base.json" "$out/zero_base.json"
.venv/bin/python -m diagnostics.export_phase_controller --controller "$out/zero_base.json" --output "$out/export_zero"
# Matched zero test must reproduce the existing R11 failure, not hide it.
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/capture_r15_zero_regression" --validate "$out/export_zero/controller_contract.json" --corrector-onnx "$out/export_zero/corrector.onnx" --duration 60 --seed-start 117 --seed-count 1
.venv/bin/python -m diagnostics.compare_phase_results "$root/outputs/contact_balance_r11_final_heldout_60s/results.json" "$root/outputs/capture_r15_zero_regression/results.json"
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$out" --capture-base "$out/zero_base.json" --native-controller-template "$out/export_zero/corrector.onnx" --duration 60 --seeds 0,21,31,38,57,117 --generations 6 --population 16 --workers 8 --initial-std .4
.venv/bin/python -m diagnostics.export_phase_controller --controller "$out/best.json" --output "$out/export_final"
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/capture_r15_final_heldout_60s" --validate "$out/export_final/controller_contract.json" --corrector-onnx "$out/export_final/corrector.onnx" --duration 60 --seed-start 300 --seed-count 30 --workers 8
.venv/bin/python -m diagnostics.summarize_phase_result "$root/outputs/capture_r15_final_heldout_60s/results.json"
