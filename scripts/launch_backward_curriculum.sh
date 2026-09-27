#!/usr/bin/env bash
# Staged heading-safe continuation from a stable reverse checkpoint.
set -euo pipefail
ROOT=/data/shijinsheng/open_duck
REPO="$ROOT/projects/Open_Duck_Playground"
GPU="${1:-6}"
NAME="${2:-backward_curriculum_s1}"
VX_MIN="${3:--0.035}"
VX_MAX="${4:--0.035}"
STEPS="${5:-4096000}"
RESTORE="${6:-$ROOT/training/backward_contact_step_v30/2026_09_27_182859_4259840}"
OUT="$ROOT/training/$NAME"
mkdir -p "$OUT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 TF_FORCE_GPU_ALLOW_GROWTH=true
export BACKWARD_REAR_SUPPORT_SHORTFALL="${BACKWARD_REAR_SUPPORT_SHORTFALL:-60}"
export BACKWARD_SINGLE_SUPPORT="${BACKWARD_SINGLE_SUPPORT:-40}"
export BACKWARD_SWING_REAR="${BACKWARD_SWING_REAR:-30}"
export BACKWARD_TERMINATION="${BACKWARD_TERMINATION:-10000}"
export BACKWARD_NORMALIZED_PROGRESS="${BACKWARD_NORMALIZED_PROGRESS:-160}"
export BACKWARD_PROGRESS_SHORTFALL="${BACKWARD_PROGRESS_SHORTFALL:-160}"
export BACKWARD_HEADING_ERROR="${BACKWARD_HEADING_ERROR:-4}"
export BACKWARD_HEADING_WARMUP_STEPS="${BACKWARD_HEADING_WARMUP_STEPS:-250}"
export BACKWARD_HEADING_RAMP_STEPS="${BACKWARD_HEADING_RAMP_STEPS:-250}"
{
  echo "=== $NAME start $(date -Is) gpu=$GPU vx=[$VX_MIN,$VX_MAX] steps=$STEPS restore=$RESTORE ==="
  env | grep '^BACKWARD_' | sort
  git rev-parse HEAD
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
    --skill backward --output_dir "$OUT" --num_timesteps "$STEPS" \
    --num_envs 1024 --num_evals 4 --num_eval_envs 64 --seed "${SEED:-81}" \
    --learning_rate 1e-5 --vx-min "$VX_MIN" --vx-max "$VX_MAX" \
    --restore_checkpoint_path "$RESTORE"
  echo "=== $NAME exited rc=$? $(date -Is) ==="
} >> "$OUT/train.log" 2>&1
