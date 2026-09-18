#!/usr/bin/env bash
# Backward v15: imitation is now 0 for reverse commands and the overspeed band
# was widened, so "walk backwards" finally out-scores "stand still".
#   $1=gpu $2=name $3=restore checkpoint
set -euo pipefail
GPU="$1"; NAME="$2"; RESTORE="$3"
P=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
OUT="/data/shijinsheng/open_duck/training/${NAME}"
mkdir -p "$OUT"
cd "$P"
export CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export TF_FORCE_GPU_ALLOW_GROWTH=true
{
  echo "=== ${NAME} start $(date -Is) gpu=$GPU restore=$RESTORE ==="
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
    --skill backward --output_dir "$OUT" \
    --num_timesteps 12000000 --num_envs 1024 --num_evals 7 --num_eval_envs 64 \
    --seed 63 --learning_rate 3e-5 \
    --vx-min -0.05 --vx-max -0.03 \
    --restore_checkpoint_path "$RESTORE"
  echo "=== ${NAME} exited rc=$? $(date -Is) ==="
} >> "$OUT/train.log" 2>&1
