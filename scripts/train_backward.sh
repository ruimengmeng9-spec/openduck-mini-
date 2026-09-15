#!/usr/bin/env bash
set -euo pipefail

OPEN_DUCK_ROOT="${OPEN_DUCK_ROOT:-$HOME/open_duck}"
PLAYGROUND_ROOT="${PLAYGROUND_ROOT:-$OPEN_DUCK_ROOT/projects/Open_Duck_Playground}"
OUTPUT_DIR="${OUTPUT_DIR:-$OPEN_DUCK_ROOT/training/backward_v13}"
GPU_ID="${GPU_ID:-0}"
RESTORE_CHECKPOINT="${RESTORE_CHECKPOINT:-}"

if [[ -z "$RESTORE_CHECKPOINT" ]]; then
  echo "Set RESTORE_CHECKPOINT to an Orbax checkpoint directory." >&2
  exit 2
fi

cd "$PLAYGROUND_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

.venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
  --skill backward \
  --output_dir "$OUTPUT_DIR" \
  --num_timesteps "${NUM_TIMESTEPS:-12000000}" \
  --num_envs "${NUM_ENVS:-1024}" \
  --num_evals "${NUM_EVALS:-7}" \
  --num_eval_envs "${NUM_EVAL_ENVS:-64}" \
  --seed "${SEED:-56}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --restore_checkpoint_path "$RESTORE_CHECKPOINT"
