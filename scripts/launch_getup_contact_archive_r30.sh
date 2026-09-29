#!/usr/bin/env bash
set -euo pipefail
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=/data/shijinsheng/open_duck/training/getup_contact_archive_r30
[[ ! -e "$out" ]] || { echo 'Refusing to overwrite experiment'; exit 1; }
mkdir -p "$out"
for pose in prone supine left_side right_side; do
  nice -n 15 .venv/bin/python -u -m diagnostics.search_getup_contact_archive_r30 \
    --pose "$pose" --output "$out/$pose" --generations 48 --population 32 --workers 2
done
echo 'R30_ALL_ORIENTATIONS_TERMINAL'
