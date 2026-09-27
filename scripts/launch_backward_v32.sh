#!/usr/bin/env bash
# Backward v32: speed up the first policy that survives the sustained gate.
#
# v30 (contact-aware stepping) is 5/5 upright for the full 10 s, the first reverse
# policy ever to pass that part, but only reaches -0.028 m/s against a -0.074
# command.  v31 added heading/yaw/lin-vel constraints on top and broke it (1/5).
# This run keeps v30's contact terms, DROPS the v31 heading group, and pushes the
# two speed terms instead.
#
#   $1=gpu  $2=name  $3=restore  $4=normalized_progress  $5=progress_shortfall
set -euo pipefail
ROOT=/data/shijinsheng/open_duck
REPO="$ROOT/projects/Open_Duck_Playground"
GPU="$1"; NAME="$2"; RESTORE="$3"; NP="$4"; PS="$5"
OUT="$ROOT/training/$NAME"
mkdir -p "$OUT"; cd "$REPO"

export CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
export TF_FORCE_GPU_ALLOW_GROWTH=true

# v30's contact-aware stability group (the part that made it work).
export BACKWARD_REAR_SUPPORT_SHORTFALL="${BACKWARD_REAR_SUPPORT_SHORTFALL:-60}"
export BACKWARD_SINGLE_SUPPORT="${BACKWARD_SINGLE_SUPPORT:-40}"
export BACKWARD_SWING_REAR="${BACKWARD_SWING_REAR:-30}"
export BACKWARD_TERMINATION="${BACKWARD_TERMINATION:-10000}"
# v31's heading group is deliberately NOT set: it dropped completion to 1/5.
export BACKWARD_NORMALIZED_PROGRESS="$NP"
export BACKWARD_PROGRESS_SHORTFALL="$PS"

{
  echo "=== ${NAME} start $(date -Is) gpu=$GPU restore=$RESTORE np=$NP ps=$PS ==="
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
    --skill backward --output_dir "$OUT" \
    --num_timesteps 12000000 --num_envs 1024 \
    --num_evals 6 --num_eval_envs 64 \
    --seed "${SEED:-80}" --learning_rate 1e-5 \
    --vx-min -0.074 --vx-max -0.074 \
    --restore_checkpoint_path "$RESTORE"
  echo "=== ${NAME} exited rc=$? $(date -Is) ==="
} >> "$OUT/train.log" 2>&1
