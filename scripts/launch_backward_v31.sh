#!/usr/bin/env bash
# Heading-constrained continuation of contact-aware reverse stepping.
set -euo pipefail

ROOT=/data/shijinsheng/open_duck
REPO="$ROOT/projects/Open_Duck_Playground"
OUT="$ROOT/training/backward_heading_v31"
RESTORE="$ROOT/training/backward_contact_step_v30/2026_09_27_182859_4259840"

mkdir -p "$OUT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export BACKWARD_REAR_SUPPORT_SHORTFALL=60
export BACKWARD_SINGLE_SUPPORT=40
export BACKWARD_SWING_REAR=30
export BACKWARD_TERMINATION=10000
export BACKWARD_HEADING_ERROR=60
export BACKWARD_YAW_ERROR=20
export BACKWARD_LIN_VEL_XY_ERROR=25

.venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
  --skill backward --output_dir "$OUT" \
  --num_timesteps 4096000 --num_envs 1024 \
  --num_evals 3 --num_eval_envs 64 \
  --seed 78 --learning_rate 1e-5 \
  --vx-min -0.074 --vx-max -0.074 \
  --restore_checkpoint_path "$RESTORE"
