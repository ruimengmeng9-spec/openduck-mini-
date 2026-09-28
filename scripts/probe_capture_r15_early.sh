#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
root=/data/shijinsheng/open_duck
out="$root/training/backward_capture_native_r15"
.venv/bin/python -m diagnostics.snapshot_controller "$out/best.json" "$out/snapshot_probe.json"
.venv/bin/python -m diagnostics.export_phase_controller --controller "$out/snapshot_probe.json" --output "$out/export_probe"
.venv/bin/python -u -m diagnostics.backward_phase_search --output "$root/outputs/capture_r15_probe_heldout_60s" --validate "$out/export_probe/controller_contract.json" --corrector-onnx "$out/export_probe/corrector.onnx" --duration 60 --seed-start 500 --seed-count 20 --workers 8
.venv/bin/python -m diagnostics.summarize_phase_result "$root/outputs/capture_r15_probe_heldout_60s/results.json"
