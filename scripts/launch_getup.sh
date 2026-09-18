#!/usr/bin/env bash
# Recovery ("getup") training on the collision-enabled model.
#   $1 = GPU id, $2 = output dir name, $3 = "scratch" or a checkpoint path
set -euo pipefail

GPU="$1"
NAME="$2"
RESTORE="$3"

P=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
OUT="/data/shijinsheng/open_duck/training/${NAME}"
LOG="$OUT/train.log"
mkdir -p "$OUT"
cd "$P"

export CUDA_VISIBLE_DEVICES="$GPU"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export TF_FORCE_GPU_ALLOW_GROWTH=true

ARGS=(--skill getup --task flat_terrain_getup --output_dir "$OUT"
      --num_timesteps 12000000 --num_envs 1024 --num_evals 7 --num_eval_envs 64
      --seed 60 --learning_rate 3e-5)
if [ "$RESTORE" != "scratch" ]; then
  ARGS+=(--restore_checkpoint_path "$RESTORE")
fi

{
  echo "=== getup ${NAME} start $(date -Is) gpu=$GPU restore=$RESTORE ==="
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner "${ARGS[@]}"
  echo "=== getup ${NAME} exited rc=$? $(date -Is) ==="
} >> "$LOG" 2>&1
