#!/usr/bin/env bash
set -euo pipefail

source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

# Preserve the stable v2/v4 curriculum while continuing from the final v4
# checkpoint.  v5 adds a symmetric yaw-rate error so positive overspeed is no
# longer free, and increases the double-support cost to help the weak reverse
# turn break static contact at high friction.
export COMMAND_MODE_WEIGHTS="0.15,0.45,0.35,0.05"
export WALK_FORWARD_ONLY=1
export WALK_MIN_SPEED=0.08
export WALK_MAX_SPEED=0.12
export TURN_MIN_YAW=0.10
export TURN_MAX_YAW=0.20
export TURN_POSITIVE_PROBABILITY=0.5

export TRACKING_ANG_VEL_SCALE=12
export TURN_PROGRESS_SCALE=20
export TURN_SHORTFALL_SCALE=-20
export TURN_WRONG_WAY_SCALE=-20
export YAW_TRACKING_ERROR_SCALE=-8
export TURN_DOUBLE_SUPPORT_SCALE=-5
export REWARD_FLOOR=-5

export CONTACT_FRICTION_MIN=0.3
export CONTACT_FRICTION_MAX=1.4

RESTORE_CHECKPOINT="/data/shijinsheng/open_duck/training/official_seed_turn_balance_v4_friction/2026_09_23_232511_2621440"
SAVE_DIR="/data/shijinsheng/open_duck/training/official_seed_turn_balance_v5_yaw_error"

mkdir -p "$SAVE_DIR"

exec .venv/bin/python -u -m playground.open_duck_mini_v2.walk_turn_stop_runner \
  --num_timesteps 2621440 \
  --num_envs 4096 \
  --num_evals 5 \
  --num_eval_envs 128 \
  --learning_rate 1e-7 \
  --seed 42 \
  --restore_checkpoint_path "$RESTORE_CHECKPOINT" \
  --output_dir "$SAVE_DIR"
