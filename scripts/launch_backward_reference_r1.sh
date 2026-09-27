#!/usr/bin/env bash
# R1: stable v30 policy with a continuous longer-stride reverse reference.
set -euo pipefail
ROOT=/data/shijinsheng/open_duck
REPO="$ROOT/projects/Open_Duck_Playground"
GPU="${1:-6}"
NAME="${2:-backward_reference_r1}"
STEPS="${3:-2097152}"
RESTORE="$ROOT/training/backward_contact_step_v30/2026_09_27_182859_4259840"
OUT="$ROOT/training/$NAME"
mkdir -p "$OUT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 TF_FORCE_GPU_ALLOW_GROWTH=true
export REFERENCE_DX_INTERPOLATION=1
export REFERENCE_DX="${REFERENCE_DX:--0.0925}"
export BACKWARD_ACTION_IMITATION="${BACKWARD_ACTION_IMITATION:-25}"
export BACKWARD_ACTION_IMITATION_SIGMA="${BACKWARD_ACTION_IMITATION_SIGMA:-0.25}"
export BACKWARD_REAR_SUPPORT_SHORTFALL="${BACKWARD_REAR_SUPPORT_SHORTFALL:-60}"
export BACKWARD_SINGLE_SUPPORT="${BACKWARD_SINGLE_SUPPORT:-40}"
export BACKWARD_SWING_REAR="${BACKWARD_SWING_REAR:-30}"
export BACKWARD_TERMINATION="${BACKWARD_TERMINATION:-10000}"
export BACKWARD_NORMALIZED_PROGRESS="${BACKWARD_NORMALIZED_PROGRESS:-160}"
export BACKWARD_PROGRESS_SHORTFALL="${BACKWARD_PROGRESS_SHORTFALL:-160}"
export BACKWARD_HEADING_ERROR="${BACKWARD_HEADING_ERROR:-0}"
{
  echo "=== $NAME start $(date -Is) gpu=$GPU reference_dx=$REFERENCE_DX steps=$STEPS restore=$RESTORE ==="
  env | grep -E '^(REFERENCE_|BACKWARD_)' | sort
  git rev-parse HEAD
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
    --skill backward --output_dir "$OUT" --num_timesteps "$STEPS" \
    --num_envs 1024 --num_evals 4 --num_eval_envs 64 --seed "${SEED:-82}" \
    --learning_rate 1e-5 --vx-min -0.074 --vx-max -0.074 \
    --restore_checkpoint_path "$RESTORE"
  echo "=== $NAME exited rc=$? $(date -Is) ==="
} >> "$OUT/train.log" 2>&1
