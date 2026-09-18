#!/usr/bin/env bash
# Staged recovery run: difficulty is set entirely by environment variables so a
# single script covers "mild lean fine-tune" through "full fall from scratch".
#   $1=gpu  $2=name  $3=restore|scratch  $4=fallen_prob  $5=fallen_tilt
#   $6=lean_tilt  $7=random_joints(0/1)  $8=timesteps
set -euo pipefail
GPU="$1"; NAME="$2"; RESTORE="$3"; FALLEN_PROB="$4"; FALLEN_TILT="$5"
LEAN_TILT="$6"; RAND_JOINTS="$7"; STEPS="$8"

P=/data/shijinsheng/open_duck/projects/Open_Duck_Playground
OUT="/data/shijinsheng/open_duck/training/${NAME}"
mkdir -p "$OUT"
cd "$P"

export CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export TF_FORCE_GPU_ALLOW_GROWTH=true
export GETUP_FALLEN_PROB="$FALLEN_PROB"
export GETUP_FALLEN_TILT="$FALLEN_TILT"
export GETUP_LEAN_TILT="$LEAN_TILT"
export GETUP_RANDOM_JOINTS="$RAND_JOINTS"

ARGS=(--skill getup --task flat_terrain_getup --output_dir "$OUT"
      --num_timesteps "$STEPS" --num_envs 1024 --num_evals 8 --num_eval_envs 64
      --seed 62 --learning_rate 1e-5)
if [ "$RESTORE" != "scratch" ]; then ARGS+=(--restore_checkpoint_path "$RESTORE"); fi

{
  echo "=== ${NAME} start $(date -Is) gpu=$GPU restore=$RESTORE ==="
  echo "difficulty: fallen_prob=$FALLEN_PROB fallen_tilt=$FALLEN_TILT lean_tilt=$LEAN_TILT random_joints=$RAND_JOINTS"
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner "${ARGS[@]}"
  echo "=== ${NAME} exited rc=$? $(date -Is) ==="
} >> "$OUT/train.log" 2>&1
