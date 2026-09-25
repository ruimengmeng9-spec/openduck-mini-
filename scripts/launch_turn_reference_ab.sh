#!/usr/bin/env bash
set -euo pipefail

variant="${1:?use control, reduced, matched_control, matched_reduced, joint_only, ref_speedup, negative_curriculum, targeted_negative, targeted_support, or negative_no_imitation}"
case "$variant" in
  control) imitation_factor=1.0 ;;
  reduced) imitation_factor=0.25 ;;
  matched_control) imitation_factor=1.0 ;;
  matched_reduced) imitation_factor=0.25 ;;
  joint_only) imitation_factor=1.0 ;;
  ref_speedup) imitation_factor=1.0 ;;
  negative_curriculum) imitation_factor=1.0 ;;
  targeted_negative) imitation_factor=1.0 ;;
  targeted_support) imitation_factor=1.0 ;;
  negative_no_imitation) imitation_factor=0.0 ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac

source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

# The physics XML has floor friction 0.6 and foot friction 1.0.
# Training randomization differs by variant; all evaluations use the XML values.
export COMMAND_MODE_WEIGHTS=0.15,0.45,0.35,0.05
export WALK_FORWARD_ONLY=1
export WALK_MIN_SPEED=0.08
export WALK_MAX_SPEED=0.12
export TURN_MIN_YAW=0.10
export TURN_MAX_YAW=0.20
export TURN_POSITIVE_PROBABILITY=0.5
export TURN_MIN_YAW_NEG=0.10
export TURN_MAX_YAW_NEG=0.20

export TRACKING_ANG_VEL_SCALE=12
export TURN_PROGRESS_SCALE=20
export TURN_SHORTFALL_SCALE=-20
export TURN_WRONG_WAY_SCALE=-20
export YAW_TRACKING_ERROR_SCALE=-8
export TURN_DOUBLE_SUPPORT_SCALE=-5
export REWARD_FLOOR=-5
export NEGATIVE_TURN_IMITATION_FACTOR="$imitation_factor"
if [[ "$variant" == joint_only ]]; then
  export NEGATIVE_TURN_JOINT_IMITATION_FACTOR=0.25
else
  export NEGATIVE_TURN_JOINT_IMITATION_FACTOR=1.0
fi
if [[ "$variant" == ref_speedup || "$variant" == targeted_negative || "$variant" == targeted_support ]]; then
  export NEGATIVE_TURN_REFERENCE_SPEEDUP=1.6
else
  export NEGATIVE_TURN_REFERENCE_SPEEDUP=1.0
fi
if [[ "$variant" == targeted_support ]]; then
  export NEGATIVE_TURN_DOUBLE_SUPPORT_MULTIPLIER=5.0
else
  export NEGATIVE_TURN_DOUBLE_SUPPORT_MULTIPLIER=1.0
fi

# The first two variants disabled all domain randomization.  Their shared
# regression means that comparison cannot isolate reference weighting.  The
# matched variants retain v5's complete training setup and differ only in the
# negative-turn imitation factor.  Evaluation always uses untouched XML
# friction (floor 0.6, soles 1.0).
num_timesteps=2621440
learning_rate=1e-7
if [[ "$variant" == negative_curriculum ]]; then
  export TURN_POSITIVE_PROBABILITY=0.25
  export TURN_MIN_YAW_NEG=0.13
  export TURN_MAX_YAW_NEG=0.30
  num_timesteps=5242880
  learning_rate=2e-7
fi
if [[ "$variant" == targeted_negative || "$variant" == targeted_support || "$variant" == negative_no_imitation ]]; then
  export COMMAND_MODE_WEIGHTS=0.10,0.25,0.60,0.05
  export TURN_POSITIVE_PROBABILITY=0.20
  export TURN_MIN_YAW_NEG=0.145
  export TURN_MAX_YAW_NEG=0.155
  export RANDOMIZE_CONTACT_FRICTION=0
  num_timesteps=5242880
  learning_rate=2e-7
fi
if [[ "$variant" == negative_no_imitation ]]; then
  export NEGATIVE_TURN_REFERENCE_SPEEDUP=1.0
fi

if [[ "$variant" == matched_* || "$variant" == joint_only || "$variant" == ref_speedup || "$variant" == negative_curriculum || "$variant" == targeted_negative || "$variant" == targeted_support || "$variant" == negative_no_imitation ]]; then
  export CONTACT_FRICTION_MIN=0.3
  export CONTACT_FRICTION_MAX=1.4
  randomization_args=()
else
  export TURN_POSITIVE_PROBABILITY=0.25
  randomization_args=(--no_domain_randomization)
fi

checkpoint=/data/shijinsheng/open_duck/training/official_seed_turn_balance_v5_yaw_error/2026_09_24_094346_2621440
output="/data/shijinsheng/open_duck/training/turn_reference_ab_${variant}"

exec .venv/bin/python -u -m playground.open_duck_mini_v2.walk_turn_stop_runner \
  --output_dir "$output" \
  --num_timesteps "$num_timesteps" \
  --num_envs 4096 \
  --num_evals 5 \
  --num_eval_envs 128 \
  --seed 42 \
  --learning_rate "$learning_rate" \
  --restore_checkpoint_path "$checkpoint" \
  "${randomization_args[@]}"
