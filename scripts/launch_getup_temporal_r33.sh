#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground

export CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Independent experiment: refuse to overlap the previous GPU6 learner.
if pgrep -f 'python -u -m diagnostics.train_getup_fullfallen_ppo_r32' >/dev/null; then
  echo 'R32 still running; do not share its learner GPU'
  exit 1
fi

available=$(nvidia-smi -i 6 --query-gpu=memory.free --format=csv,noheader,nounits)
busy=$(nvidia-smi -i 6 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
[[ "$available" -ge 8000 && "$busy" -lt 20 ]] || { echo 'GPU6 busy; do not evict another user'; exit 1; }

CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -m unittest \
  diagnostics.test_getup_temporal_noise_r33 diagnostics.test_getup_fullfallen_r32 -v

initialize=/data/shijinsheng/open_duck/training/getup_native_ppo_r11_stage1/checkpoint_0192.msgpack
[[ -f "$initialize" ]] || { echo 'Missing declared initialization actor'; exit 1; }

stamp=$(date +%Y%m%d_%H%M%S)
out=/data/shijinsheng/open_duck/training/getup_temporal_r33_$stamp
[[ ! -e "$out" && ! -e "${out}_smoke" ]] || { echo 'Refusing output overwrite'; exit 1; }
echo "R33_OUTPUT $out"

nice -n 10 .venv/bin/python -u -m diagnostics.train_getup_temporal_ppo_r33 \
  --output "${out}_smoke" --iterations 2 --envs 4 --workers 2 --horizon 16 \
  --initialize-actor "$initialize" --seed 1333

nice -n 10 .venv/bin/python -u -m diagnostics.train_getup_temporal_ppo_r33 \
  --output "$out" --iterations 256 --envs 16 --workers 4 --horizon 128 \
  --initialize-actor "$initialize" --seed 1333

# Four short, disjoint development starts. Only a full 80-trial test can pass.
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu nice -n 15 .venv/bin/python -u \
  -m diagnostics.validate_getup_fullfallen_r32 --actor "$out/final.onnx" \
  --output "${out}_discovery" --discovery --workers 2

if .venv/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if all(x["short_discovery_pass"] for x in r["results"]) else 1)' "${out}_discovery/results.json"; then
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu nice -n 15 .venv/bin/python -u \
    -m diagnostics.validate_getup_fullfallen_r32 --actor "$out/final.onnx" \
    --output "${out}_acceptance" --workers 2
else
  echo 'R33_DISCOVERY_FAILED: preserve all four failed trajectories for further iteration'
fi
echo 'R33_TRAINING_AND_DISCOVERY_TERMINAL'
