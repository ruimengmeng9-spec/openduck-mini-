#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
ROOT=/data/shijinsheng/open_duck/training/getup_fullpath_r27
mkdir -p "$ROOT"
for pose in left_side right_side prone supine; do
  if test -f "$ROOT/$pose/results.json"; then
    echo "Completed pose preserved: $pose"
    continue
  fi
  if test -d "$ROOT/$pose"; then
    echo "Incomplete experiment exists; refusing overwrite: $pose" >&2
    exit 1
  fi
  nice -n 10 .venv/bin/python -u -m diagnostics.train_getup_fullpath_r27 \
    --pose "$pose" --output "$ROOT/$pose" --generations 64 \
    --population 32 --workers 4 --seed 527
done
echo 'All four experiments finished; inspect independent validation before adoption.'
