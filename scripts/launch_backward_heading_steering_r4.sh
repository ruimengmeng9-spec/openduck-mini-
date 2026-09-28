#!/usr/bin/env bash
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
ROOT=/data/shijinsheng/open_duck
OUT="$ROOT/training/backward_heading_steering_r4"
mkdir -p "$OUT"
cd "$ROOT/projects/Open_Duck_Playground"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
export REFERENCE_DX_INTERPOLATION=1 REFERENCE_DX=-0.0925
unset BACKWARD_TEACHER_STAGE REFERENCE_DTHETA_INTERPOLATION
export BACKWARD_YAW_RANGE=0,0 BACKWARD_NOISE_LEVEL=0
export BACKWARD_ACTION_MAX_DELAY=1 BACKWARD_IMU_MAX_DELAY=1
export BACKWARD_ACTION_IMITATION=25 BACKWARD_ACTION_IMITATION_ERROR=5
export BACKWARD_REAR_SUPPORT_SHORTFALL=30 BACKWARD_SINGLE_SUPPORT=20 BACKWARD_SWING_REAR=15
export BACKWARD_TERMINATION=10000 BACKWARD_HEADING_ERROR=20
export BACKWARD_NORMALIZED_PROGRESS=80 BACKWARD_PROGRESS_SHORTFALL=80
{
  echo "Frozen R2 heading steering R4: $(date -Is)"
  env | grep -E '^(REFERENCE_|BACKWARD_)' | sort
  .venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
    --skill backward --output_dir "$OUT" --num_timesteps 4096000 \
    --num_envs 1024 --num_evals 5 --num_eval_envs 64 --seed 86 \
    --learning_rate 3e-4 --vx-min -0.074 --vx-max -0.074 \
    --reference-residual-gain 0.12 --reference-ramp-s 1.0 \
    --steering-baseline-model "$ROOT/training/backward_reference_residual_r2/final.onnx" \
    --steering-yaw-gain .05 --steering-roll-gain .025 --steering-initial-error .15
} > "$OUT/train.log" 2>&1
