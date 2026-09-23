"""Focused stand/walk/turn/stop training task for Open Duck Mini V2."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import joystick


def default_config() -> config_dict.ConfigDict:
    """Training configuration constrained to the deployed high-level API."""
    config = joystick.default_config()

    # Only train commands that the deployed controller exposes.
    config.lin_vel_x = [-0.15, 0.15]
    config.lin_vel_y = [0.0, 0.0]
    config.ang_vel_yaw = [-0.30, 0.30]
    config.neck_pitch_range = [0.0, 0.0]
    config.head_pitch_range = [0.0, 0.0]
    config.head_yaw_range = [0.0, 0.0]
    config.head_roll_range = [0.0, 0.0]
    config.head_range_factor = 0.0

    # Stage 1 learns command tracking before adding external pushes.
    config.push_config.enable = False
    config.noise_config.level = 0.5

    # Strong, symmetric tracking rewards.  Keep some reference-motion guidance
    # so a useful gait is discoverable from scratch, but do not let it dominate.
    config.reward_config.scales.tracking_lin_vel = 20.0
    config.reward_config.scales.lin_vel_error = -50.0
    config.reward_config.scales.tracking_ang_vel = float(
        os.environ.get("TRACKING_ANG_VEL_SCALE", "8.0")
    )
    config.reward_config.scales.imitation = 1.0
    config.reward_config.scales.stand_still = -0.2
    config.reward_config.scales.action_rate = -0.5
    # A stationary policy still receives substantial tracking reward at modest
    # commands.  Signed, normalized progress makes "do not move" distinctly
    # worse than following the requested direction without rewarding overspeed.
    config.reward_config.scales.normalized_progress = 30.0
    config.reward_config.scales.progress_shortfall = -30.0
    config.reward_config.scales.wrong_way = -30.0
    config.reward_config.scales.normalized_yaw_progress = float(
        os.environ.get("TURN_PROGRESS_SCALE", "0.0")
    )
    config.reward_config.scales.yaw_shortfall = float(
        os.environ.get("TURN_SHORTFALL_SCALE", "0.0")
    )
    config.reward_config.scales.wrong_yaw = float(
        os.environ.get("TURN_WRONG_WAY_SCALE", "0.0")
    )
    config.reward_config.tracking_sigma = 0.02
    return config


class WalkTurnStop(joystick.Joystick):
    """Joystick task with safe starts and balanced motion-command sampling."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(
            task=task,
            config=config if config is not None else default_config(),
            config_overrides=config_overrides,
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Keep random heading/position but always start joints at home.

        The upstream task multiplies every home joint by U(0.5, 1.5).  That
        produced the measured seed-1 immediate fall.  Dynamics/noise remain
        randomized; only the unsafe initial joint and base-velocity perturbation
        is removed in this first curriculum stage.
        """
        state = super().reset(rng)
        info = dict(state.info)
        qpos = self.set_actuator_joints_qpos(
            self._default_actuator,
            state.data.qpos,
        )
        qvel = jp.zeros_like(state.data.qvel)
        data = mjx_env.init(
            self.mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=self._default_actuator,
        )

        zeros = jp.zeros(self.mjx_model.nu)
        info["last_act"] = zeros
        info["last_last_act"] = zeros
        info["last_last_last_act"] = zeros
        info["motor_targets"] = self._default_actuator
        info["action_history"] = jp.zeros(
            self._config.noise_config.action_max_delay * self._actuators
        )
        info["imu_history"] = jp.zeros(
            self._config.noise_config.imu_max_delay * 3
        )

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        obs = self._get_obs(data, info, contact)
        return state.replace(data=data, obs=obs, info=info)

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample stand, walk, turn, and gentle arc commands symmetrically."""
        mode_rng, x_mag_rng, x_sign_rng, yaw_mag_rng, yaw_sign_rng = (
            jax.random.split(rng, 5)
        )
        # 0=stand, 1=walk, 2=turn, 3=walk+turn.
        mode_weights = [
            float(value)
            for value in os.environ.get(
                "COMMAND_MODE_WEIGHTS", "0.10,0.75,0.10,0.05"
            ).split(",")
        ]
        if len(mode_weights) != 4 or any(value <= 0.0 for value in mode_weights):
            raise ValueError("COMMAND_MODE_WEIGHTS must contain four positive values")
        mode_probabilities = jp.asarray(mode_weights) / sum(mode_weights)
        mode = jax.random.categorical(mode_rng, jp.log(mode_probabilities))
        min_speed = float(os.environ.get("WALK_MIN_SPEED", "0.10"))
        max_speed = float(os.environ.get("WALK_MAX_SPEED", "0.15"))
        x_magnitude = jax.random.uniform(
            x_mag_rng, minval=min_speed, maxval=max_speed
        )
        forward_only = os.environ.get("WALK_FORWARD_ONLY", "0") == "1"
        x_sign = jp.where(
            forward_only,
            1.0,
            jp.where(jax.random.bernoulli(x_sign_rng), 1.0, -1.0),
        )
        min_yaw = float(os.environ.get("TURN_MIN_YAW", "0.10"))
        max_yaw = float(os.environ.get("TURN_MAX_YAW", "0.30"))
        yaw_magnitude = jax.random.uniform(
            yaw_mag_rng, minval=min_yaw, maxval=max_yaw
        )
        positive_yaw_probability = float(
            os.environ.get("TURN_POSITIVE_PROBABILITY", "0.5")
        )
        if not 0.0 <= positive_yaw_probability <= 1.0:
            raise ValueError("TURN_POSITIVE_PROBABILITY must be between 0 and 1")
        yaw_sign = jp.where(
            jax.random.bernoulli(yaw_sign_rng, positive_yaw_probability),
            1.0,
            -1.0,
        )

        use_x = (mode == 1) | (mode == 3)
        use_yaw = (mode == 2) | (mode == 3)
        vx = jp.where(use_x, x_sign * x_magnitude, 0.0)
        yaw = jp.where(use_yaw, yaw_sign * yaw_magnitude, 0.0)
        return jp.array([vx, 0.0, yaw, 0.0, 0.0, 0.0, 0.0])

    def _get_reward(
        self,
        data: Any,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        local_velocity = self.get_local_linvel(data)
        command_x = info["command"][0]
        command_speed = jp.abs(command_x)
        direction = jp.sign(command_x)
        progress_ratio = direction * local_velocity[0] / jp.maximum(
            command_speed, 0.02
        )
        moving = command_speed > 0.01
        rewards["lin_vel_error"] = jp.abs(
            command_x - local_velocity[0]
        )
        rewards["normalized_progress"] = jp.where(
            moving, jp.clip(progress_ratio, -1.0, 1.0), 0.0
        )
        rewards["progress_shortfall"] = jp.where(
            moving, jp.maximum(0.8 - progress_ratio, 0.0), 0.0
        )
        rewards["wrong_way"] = jp.where(
            moving, jp.maximum(-progress_ratio, 0.0), 0.0
        )
        command_yaw = info["command"][2]
        yaw_speed = jp.abs(command_yaw)
        yaw_direction = jp.sign(command_yaw)
        yaw_ratio = yaw_direction * self.get_gyro(data)[2] / jp.maximum(
            yaw_speed, 0.05
        )
        turning = yaw_speed > 0.01
        rewards["normalized_yaw_progress"] = jp.where(
            turning, jp.clip(yaw_ratio, -1.0, 1.0), 0.0
        )
        rewards["yaw_shortfall"] = jp.where(
            turning, jp.maximum(0.8 - yaw_ratio, 0.0), 0.0
        )
        rewards["wrong_yaw"] = jp.where(
            turning, jp.maximum(-yaw_ratio, 0.0), 0.0
        )
        return rewards
