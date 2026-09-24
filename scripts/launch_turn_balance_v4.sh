#!/usr/bin/env bash
set -euo pipefail

source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

# Keep the accepted v2 curriculum and add only the diagnosed corrections.
export COMMAND_MODE_WEIGHTS="0.15,0.45,0.35,0.05"
export WALK_FORWARD_ONLY=1
export WALK_MIN_SPEED=0.08
export WALK_MAX_SPEED=0.12
export TURN_MIN_YAW=0.10
export TURN_MAX_YAW=0.20
export TURN_POSITIVE_PROBABILITY=0.50

export TRACKING_ANG_VEL_SCALE=12
export TURN_PROGRESS_SCALE=20
export TURN_SHORTFALL_SCALE=-20
export TURN_WRONG_WAY_SCALE=-20

# Preserve negative feedback and discourage high-friction double-foot locking.
export REWARD_FLOOR=-5.0
export TURN_DOUBLE_SUPPORT_SCALE=-2.0
export CONTACT_FRICTION_MIN=0.3
export CONTACT_FRICTION_MAX=1.4

checkpoint="/data/shijinsheng/open_duck/training/official_seed_turn_balance_v2/2026_09_23_122903_5079040"
output="/data/shijinsheng/open_duck/training/official_seed_turn_balance_v4_friction"

exec .venv/bin/python -m playground.open_duck_mini_v2.walk_turn_stop_runner \
  --output_dir "$output" \
  --num_timesteps 2621440 \
  --num_envs 4096 \
  --num_evals 5 \
  --num_eval_envs 128 \
  --seed 42 \
  --learning_rate 2e-7 \
  --restore_checkpoint_path "$checkpoint"
