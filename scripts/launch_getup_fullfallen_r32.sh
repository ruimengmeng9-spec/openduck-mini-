#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
stamp=$(date +%Y%m%d_%H%M%S)
out=/data/shijinsheng/open_duck/training/getup_fullfallen_r32_$stamp
[[ ! -e "$out" ]] || { echo 'Refusing experiment overwrite'; exit 1; }
# Read-only resource check. Do not evict any existing GPU user.
free=$(nvidia-smi -i 6 --query-gpu=memory.free --format=csv,noheader,nounits)
util=$(nvidia-smi -i 6 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
[[ "$free" -ge 8000 && "$util" -lt 20 ]] || { echo 'GPU 6 no longer available; choose a free resource explicitly'; exit 1; }
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -m unittest diagnostics.test_getup_fullfallen_r32 -v
initialize=/data/shijinsheng/open_duck/training/getup_native_ppo_r11_stage1/checkpoint_0192.msgpack
[[ -f "$initialize" ]] || { echo 'Missing declared initialization actor'; exit 1; }
echo "R32_OUTPUT $out"
nice -n 10 .venv/bin/python -u -m diagnostics.train_getup_fullfallen_ppo_r32 \
  --output "${out}_smoke" --iterations 2 --envs 4 --workers 2 --horizon 16 \
  --initialize-actor "$initialize" --seed 1232
nice -n 10 .venv/bin/python -u -m diagnostics.train_getup_fullfallen_ppo_r32 \
  --output "$out" --iterations 256 --envs 16 --workers 4 --horizon 128 \
  --initialize-actor "$initialize" --seed 1232
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu nice -n 15 .venv/bin/python -u \
  -m diagnostics.validate_getup_fullfallen_r32 --actor "$out/final.onnx" \
  --output "${out}_discovery" --discovery --workers 2
# A failed four-pose discovery run is analyzed, not mislabeled as final success.
if .venv/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if all(x["short_discovery_pass"] for x in r["results"]) else 1)' "${out}_discovery/results.json"; then
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu nice -n 15 .venv/bin/python -u \
    -m diagnostics.validate_getup_fullfallen_r32 --actor "$out/final.onnx" \
    --output "${out}_acceptance" --workers 2
else
  echo 'R32_DISCOVERY_FAILED: full-fallen task remains incomplete; preserve failures for next iteration'
fi
echo 'R32_TRAINING_AND_DISCOVERY_TERMINAL'
