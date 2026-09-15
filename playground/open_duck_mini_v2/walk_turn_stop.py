"""Focused stand/walk/turn/stop training task for Open Duck Mini V2."""

from __future__ import annotations

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
    config.reward_config.scales.tracking_ang_vel = 8.0
    config.reward_config.scales.imitation = 1.0
    config.reward_config.scales.stand_still = -0.2
    config.reward_config.scales.action_rate = -0.5
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
        mode = jax.random.categorical(
            mode_rng,
            jp.log(jp.array([0.10, 0.75, 0.10, 0.05])),
        )
        x_magnitude = jax.random.uniform(x_mag_rng, minval=0.10, maxval=0.15)
        x_sign = jp.where(jax.random.bernoulli(x_sign_rng), 1.0, -1.0)
        yaw_magnitude = jax.random.uniform(
            yaw_mag_rng, minval=0.10, maxval=0.30
        )
        yaw_sign = jp.where(jax.random.bernoulli(yaw_sign_rng), 1.0, -1.0)

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
        rewards["lin_vel_error"] = jp.abs(
            info["command"][0] - local_velocity[0]
        )
        return rewards
