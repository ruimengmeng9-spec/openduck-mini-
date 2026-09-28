#!/usr/bin/env bash
# Long uninterrupted motion and complete skill sequences are distinct gates.
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
model="$root/training/backward_filtered_heading_r20/export_final"
test -f "$model/controller_contract.json"
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/filtered_combined_r22_120s" --validate "$model/controller_contract.json" --corrector-onnx "$model/corrector.onnx" --duration 120 --seed-start 1500 --seed-count 20 --workers 8 > "$root/tmp/filtered_combined_r22_long.log" 2>&1 &
long_pid=$!
.venv/bin/python -u -m diagnostics.backward_skill_sequence --contract "$model/controller_contract.json" --model "$model/corrector.onnx" --output "$root/outputs/filtered_combined_r22_sequence" --seed-start 1600 --seed-count 50 --workers 8 --stop-blend-s 1 > "$root/tmp/filtered_combined_r22_sequence.log" 2>&1 &
sequence_pid=$!
wait "$long_pid"
wait "$sequence_pid"
.venv/bin/python -m diagnostics.summarize_phase_result "$root/outputs/filtered_combined_r22_120s/results.json"
.venv/bin/python -m diagnostics.summarize_backward_sequence "$root/outputs/filtered_combined_r22_sequence/results.json"
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/filtered_heading_r20_training_regression" --validate "$model/controller_contract.json" --corrector-onnx "$model/corrector.onnx" --duration 60 --validation-seeds 0,21,31,117,800,825,847 --workers 8
.venv/bin/python -m diagnostics.compare_phase_results "$root/training/backward_filtered_heading_r20/best.json" "$root/outputs/filtered_heading_r20_training_regression/results.json"
