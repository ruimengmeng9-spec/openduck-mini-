#!/usr/bin/env bash
# Backward v17: target the reference grid value.
# The reference grid is {-0.148, -0.074, 0.0, +0.074, ...}; the v13/v15 curriculum
# of -0.03..-0.05 quantised onto dx=0.000 (the standing reference), which is why
# imitation had to be switched off to get any reverse motion at all.  Centring the
# command on -0.074 lands on a real reverse gait, so imitation can be restored and
# supply the balance prior that v15/v16 lacked (they fall after ~2 s).
#   $1=gpu $2=name $3=restore $4=imitation weight
set -euo pipefail
GPU="$1"; NAME="$2"; RESTORE="$3"; IMIT="$4"
P=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
OUT="/data/shijinsheng/open_duck/training/${NAME}"
mkdir -p "$OUT"; cd "$P"
export CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export TF_FORCE_GPU_ALLOW_GROWTH=true
export BACKWARD_IMITATION="$IMIT"
{
  echo "=== ${NAME} start $(date -Is) gpu=$GPU restore=$RESTORE imitation=$IMIT ==="
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
    --skill backward --output_dir "$OUT" \
    --num_timesteps 12000000 --num_envs 1024 --num_evals 7 --num_eval_envs 64 \
    --seed 65 --learning_rate 2e-5 \
    --vx-min -0.090 --vx-max -0.060 \
    --restore_checkpoint_path "$RESTORE"
  echo "=== ${NAME} exited rc=$? $(date -Is) ==="
} >> "$OUT/train.log" 2>&1
