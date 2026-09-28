#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
out="$root/training/backward_filtered_heading_r20"
base="$root/training/backward_target_smoothing_r18/tau0.01.json"
model="$root/training/backward_contact_balance_r11/export_final/corrector.onnx"
test ! -e "$out/search.json"
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$out" --feedback-base "$base" --resume "$base" --native-controller-template "$model" --duration 60 --seeds 0,21,31,117,800,825,847 --generations 6 --population 16 --workers 8 --initial-std .12
.venv/bin/python -m diagnostics.export_phase_controller --controller "$out/best.json" --output "$out/export_final"
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/filtered_heading_r20_final_heldout_60s" --validate "$out/export_final/controller_contract.json" --corrector-onnx "$out/export_final/corrector.onnx" --duration 60 --seed-start 1200 --seed-count 50 --workers 8
.venv/bin/python -m diagnostics.summarize_phase_result "$root/outputs/filtered_heading_r20_final_heldout_60s/results.json"
