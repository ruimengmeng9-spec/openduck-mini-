#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
inner="$root/training/backward_contact_balance_r11/export_final"
contract="$root/training/backward_target_smoothing_r18/tau0.01.json"
.venv/bin/python -m diagnostics.make_target_smoothing_contract "$inner/controller_contract.json" "$contract" --tau .01
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/target_smoothing_r18_tau0.01_broad_60s" --validate "$contract" --corrector-onnx "$inner/corrector.onnx" --duration 60 --seed-start 800 --seed-count 50 --workers 8
.venv/bin/python -m diagnostics.summarize_phase_result "$root/outputs/target_smoothing_r18_tau0.01_broad_60s/results.json"
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/target_smoothing_r18_tau0.01_stand_60s" --validate "$contract" --corrector-onnx "$inner/corrector.onnx" --duration 60 --seed-start 900 --seed-count 20 --workers 8 --stand-warmup-s 3
.venv/bin/python -m diagnostics.summarize_phase_result "$root/outputs/target_smoothing_r18_tau0.01_stand_60s/results.json"
