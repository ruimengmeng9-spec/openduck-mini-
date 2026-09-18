#!/usr/bin/env bash
# Backward v14: restore the reverse gait that v13 flattened into standing.
set -euo pipefail

P=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
OUT=/data/shijinsheng/open_duck/training/backward_v14_progress_balance_20260918
RESTORE=/data/shijinsheng/open_duck/training/backward_v12_ratio_curriculum_20260915/2026_09_15_233406_12779520
LOG="$OUT/train.log"

mkdir -p "$OUT"
cd "$P"

export CUDA_VISIBLE_DEVICES=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export TF_FORCE_GPU_ALLOW_GROWTH=true

{
  echo "=== backward v14 start $(date -Is) ==="
  echo "restore=$RESTORE"
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
    --skill backward \
    --output_dir "$OUT" \
    --num_timesteps 12000000 \
    --num_envs 1024 \
    --num_evals 7 \
    --num_eval_envs 64 \
    --seed 58 \
    --learning_rate 3e-5 \
    --vx-min -0.05 \
    --vx-max -0.03 \
    --restore_checkpoint_path "$RESTORE"
  echo "=== backward v14 exited rc=$? $(date -Is) ==="
} >> "$LOG" 2>&1
