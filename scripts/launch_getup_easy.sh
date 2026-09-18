#!/usr/bin/env bash
# Easy-mode recovery: mild tilt only, home joints.  Verifies the task is
# learnable before scaling to the full fallen distribution.
set -euo pipefail
P=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
OUT=/data/shijinsheng/open_duck/training/getup_easy_lean_20260918
mkdir -p "$OUT"
cd "$P"
export CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 TF_FORCE_GPU_ALLOW_GROWTH=true
# Mild-only: never fully fallen, keep the home joint posture.
export GETUP_FALLEN_PROB=0.0
export GETUP_LEAN_TILT=0.35,0.70
export GETUP_RANDOM_JOINTS=0
{
  echo "=== getup easy lean start $(date -Is) ==="
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
    --skill getup --task flat_terrain_getup --output_dir "$OUT" \
    --num_timesteps 8000000 --num_envs 1024 --num_evals 8 --num_eval_envs 64 \
    --seed 61 --learning_rate 3e-5
  echo "=== getup easy exited rc=$? $(date -Is) ==="
} >> "$OUT/train.log" 2>&1
