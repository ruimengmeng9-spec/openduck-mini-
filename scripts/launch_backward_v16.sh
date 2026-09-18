#!/usr/bin/env bash
# Backward v16: v15 finally walks backwards at the commanded speed (measured
# -0.072 / -0.095 m/s) but still falls after 1.8-2.5 s, which was the original
# v12 failure.  Now that "move" beats "stand still", the posture costs can be
# raised again to buy stability without collapsing back onto standing.
#   $1=gpu $2=name $3=restore
set -euo pipefail
GPU="$1"; NAME="$2"; RESTORE="$3"
P=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
OUT="/data/shijinsheng/open_duck/training/${NAME}"
mkdir -p "$OUT"; cd "$P"
export CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export TF_FORCE_GPU_ALLOW_GROWTH=true
# Re-tighten stability (was relaxed in v15 to break the standing optimum).
export BACKWARD_TILT_MARGIN="${BACKWARD_TILT_MARGIN:--1500}"
export BACKWARD_HEIGHT_MARGIN="${BACKWARD_HEIGHT_MARGIN:--1500}"
export BACKWARD_TERMINATION="${BACKWARD_TERMINATION:--2500}"
export BACKWARD_TILT="${BACKWARD_TILT:--400}"
{
  echo "=== ${NAME} start $(date -Is) gpu=$GPU restore=$RESTORE ==="
  echo "stability: tilt_margin=$BACKWARD_TILT_MARGIN height_margin=$BACKWARD_HEIGHT_MARGIN termination=$BACKWARD_TERMINATION tilt=$BACKWARD_TILT"
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
    --skill backward --output_dir "$OUT" \
    --num_timesteps 12000000 --num_envs 1024 --num_evals 7 --num_eval_envs 64 \
    --seed 64 --learning_rate 2e-5 \
    --vx-min -0.06 --vx-max -0.04 \
    --restore_checkpoint_path "$RESTORE"
  echo "=== ${NAME} exited rc=$? $(date -Is) ==="
} >> "$OUT/train.log" 2>&1
