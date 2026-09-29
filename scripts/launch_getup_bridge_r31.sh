#!/usr/bin/env bash
set -euo pipefail
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
archive=/data/shijinsheng/open_duck/training/getup_contact_archive_r30
out=/data/shijinsheng/open_duck/training/getup_bridge_r31
[[ ! -e "$out" ]] || { echo 'Refusing to overwrite bridge experiment'; exit 1; }
mkdir -p "$out"
for pose in prone supine left_side right_side; do
  while [[ ! -f "$archive/$pose/results.json" ]]; do
    if ! kill -0 1040260 2>/dev/null; then
      echo "Archive producer terminated without results for $pose" >&2
      exit 1
    fi
    sleep 10
  done
  nice -n 15 .venv/bin/python -u -m diagnostics.bridge_getup_contact_prefixes_r31 \
    --archive "$archive/$pose" --pose "$pose" --output "$out/$pose" --prefixes 24 --workers 2
done
echo 'R31_BRIDGE_ALL_ORIENTATIONS_TERMINAL'
