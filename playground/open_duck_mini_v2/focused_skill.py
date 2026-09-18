"""Focused locomotion curricula for extending Open Duck Mini skills."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict

from . import walk_turn_stop


SUPPORTED_SKILLS = ("backward", "lateral", "arc", "turn", "unified")


def focused_config(skill: str) -> config_dict.ConfigDict:
    """Return rewards and command bounds for one focused curriculum."""
    if skill not in SUPPORTED_SKILLS:
        raise ValueError(f"Unsupported skill {skill!r}; choose from {SUPPORTED_SKILLS}")

    config = walk_turn_stop.default_config()
    # Backward uses a staged curriculum.  Keep perturbations gentle until the
    # policy learns a repeatable reverse gait, then add robustness separately.
    config.noise_config.level = 0.05 if skill == "backward" else 0.35
    config.push_config.enable = False

    # Keep every term positive enough that the upstream reward clipping does not
    # erase the velocity gradient.  Direct squared errors remain small shaping
    # costs, while the exponential terms give a clear improvement near target.
    scales = config.reward_config.scales
    scales.tracking_lin_vel = 0.0
    scales.tracking_ang_vel = 14.0
    scales.lin_vel_error = -3.0
    scales.tracking_xy = 60.0 if skill == "backward" else 18.0
    scales.lin_vel_xy_error = -10.0 if skill == "backward" else -8.0
    scales.yaw_error = -3.0
    scales.upright = 20.0 if skill == "backward" else 10.0
    scales.vertical_velocity = -5.0 if skill == "backward" else -2.0
    scales.angular_xy = -2.0 if skill == "backward" else -0.5
    scales.command_progress = {
        "backward": 0.0,
        "lateral": 80.0,
        "arc": 40.0,
        "turn": 0.0,
        "unified": 40.0,
    }[skill]
    scales.yaw_progress = {
        "backward": 0.0,
        "lateral": 0.0,
        "arc": 20.0,
        "turn": 35.0,
        "unified": 20.0,
    }[skill]
    scales.speed_limit = -300.0
    scales.yaw_limit = -20.0
    # The upstream environment clips the summed reward at zero.  These linear
    # backward-only terms prevent the positive upright/alive reward from making
    # stationary behaviour a profitable local optimum.
    scales.normalized_progress = 40.0 if skill == "backward" else 0.0
    scales.progress_shortfall = -40.0 if skill == "backward" else 0.0
    scales.wrong_way = -80.0 if skill == "backward" else 0.0
    scales.overspeed = -4000.0 if skill == "backward" else 0.0
    scales.tilt_margin = -2000.0 if skill == "backward" else 0.0
    scales.height_margin = -2000.0 if skill == "backward" else 0.0
    scales.tilt = -500.0 if skill == "backward" else -30.0
    scales.termination = -3000.0 if skill == "backward" else -50.0
    scales.alive = 0.0 if skill == "backward" else 8.0
    scales.imitation = {
        # The official imitation target is excellent for balance but tends to
        # anchor a backward command near standing.  Keep a small stabilizing
        # prior while allowing the learned gait to depart from it.
        "backward": 0.25,
        "lateral": 0.5,
        "arc": 1.5,
        "turn": 1.0,
        "unified": 1.5,
    }[skill]
    scales.stand_still = -0.5
    scales.action_rate = -0.3
    scales.torques = -5.0e-4
    config.reward_config.tracking_sigma = 0.0025 if skill == "backward" else 0.015
    # Focused backward learning needs signed feedback: clipping every negative
    # total to zero creates a flat region around the stationary policy.
    config.reward_config.reward_floor = -10000.0 if skill == "backward" else 0.0
    return config


class FocusedSkill(walk_turn_stop.WalkTurnStop):
    """Home-start environment with a skill-specific command distribution."""

    def __init__(
        self,
        skill: str,
        task: str = "flat_terrain",
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        if skill not in SUPPORTED_SKILLS:
            raise ValueError(f"Unsupported skill {skill!r}")
        self.skill = skill
        super().__init__(
            task=task,
            config=config if config is not None else focused_config(skill),
            config_overrides=config_overrides,
        )

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample balanced commands without a zero-gradient dead zone."""
        mode_rng, x_rng, y_rng, yaw_rng, sign_x_rng, sign_y_rng, sign_yaw_rng = (
            jax.random.split(rng, 7)
        )
        backward_min = 0.025 if self.skill == "backward" else 0.06
        backward_max = 0.035 if self.skill == "backward" else 0.15
        x_mag = jax.random.uniform(
            x_rng, minval=backward_min, maxval=backward_max
        )
        y_mag = jax.random.uniform(y_rng, minval=0.05, maxval=0.12)
        yaw_mag = jax.random.uniform(yaw_rng, minval=0.10, maxval=0.30)
        x_sign = jp.where(jax.random.bernoulli(sign_x_rng), 1.0, -1.0)
        y_sign = jp.where(jax.random.bernoulli(sign_y_rng), 1.0, -1.0)
        yaw_sign = jp.where(jax.random.bernoulli(sign_yaw_rng), 1.0, -1.0)

        if self.skill == "backward":
            moving = jax.random.bernoulli(mode_rng, p=0.95)
            vx = jp.where(moving, -x_mag, 0.0)
            return jp.array([vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if self.skill == "lateral":
            moving = jax.random.bernoulli(mode_rng, p=0.85)
            vy = jp.where(moving, y_sign * y_mag, 0.0)
            return jp.array([0.0, vy, 0.0, 0.0, 0.0, 0.0, 0.0])

        if self.skill == "turn":
            moving = jax.random.bernoulli(mode_rng, p=0.85)
            yaw = jp.where(moving, yaw_sign * yaw_mag, 0.0)
            return jp.array([0.0, 0.0, yaw, 0.0, 0.0, 0.0, 0.0])

        if self.skill == "arc":
            # 0=stand, 1=forward, 2=turn, 3=forward arc.
            mode = jax.random.categorical(
                mode_rng, jp.log(jp.array([0.10, 0.15, 0.15, 0.60]))
            )
            vx = jp.where((mode == 1) | (mode == 3), x_mag, 0.0)
            yaw = jp.where((mode == 2) | (mode == 3), yaw_sign * yaw_mag, 0.0)
            return jp.array([vx, 0.0, yaw, 0.0, 0.0, 0.0, 0.0])

        # Unified deployment curriculum: stand, forward, backward, turn, arc,
        # and lateral motion.  Focused policies should be learned first.
        mode = jax.random.categorical(
            mode_rng,
            jp.log(jp.array([0.10, 0.20, 0.15, 0.25, 0.20, 0.10])),
        )
        vx = jp.where(mode == 1, x_mag, 0.0)
        vx = jp.where(mode == 2, -x_mag, vx)
        vx = jp.where(mode == 4, x_mag, vx)
        vy = jp.where(mode == 5, y_sign * y_mag, 0.0)
        yaw = jp.where((mode == 3) | (mode == 4), yaw_sign * yaw_mag, 0.0)
        return jp.array([vx, vy, yaw, 0.0, 0.0, 0.0, 0.0])

    def _get_termination(self, data: Any) -> jax.Array:
        """Match backward training failures to the runtime safety envelope."""
        done = super()._get_termination(data)
        if self.skill != "backward":
            return done
        gravity_z = self.get_gravity(data)[-1]
        base_height = self.get_floating_base_qpos(data.qpos)[2]
        # Runtime stops at up_z < 0.50 or base height < 0.08 m.  Terminating
        # training somewhat earlier gives PPO a useful margin instead of
        # rewarding a fast backward lunge until the robot is almost inverted.
        return done | (gravity_z < 0.75) | (base_height < 0.10)

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
        command = info["command"]
        local_velocity = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        xy_error = jp.sum(jp.square(command[:2] - local_velocity[:2]))
        yaw_error = jp.square(command[2] - gyro[2])
        command_speed = jp.linalg.norm(command[:2])
        command_direction = command[:2] / jp.maximum(command_speed, 0.05)
        command_progress = jp.dot(command_direction, local_velocity[:2])
        moving_command = command_speed > 0.01
        progress_ratio = command_progress / jp.maximum(command_speed, 0.02)
        # The absolute progress term is tiny for low-speed commands and allowed
        # a near-stationary local optimum.  A clipped dimensionless ratio makes
        # the target equally visible across the 0.025--0.04 m/s curriculum,
        # without rewarding a dangerous high-speed reverse lunge.
        normalized_progress = jp.where(
            moving_command, jp.clip(progress_ratio, -1.0, 1.0), 0.0
        )
        progress_shortfall = jp.where(
            moving_command, jp.maximum(0.80 - progress_ratio, 0.0), 0.0
        )
        wrong_way = jp.where(moving_command, jp.maximum(-progress_ratio, 0.0), 0.0)
        allowed_speed = 1.10 * command_speed + 0.005
        overspeed = jp.maximum(jp.linalg.norm(local_velocity[:2]) - allowed_speed, 0.0)
        tilt_margin = jp.maximum(0.985 - gravity[2], 0.0)
        base_height = self.get_floating_base_qpos(data.qpos)[2]
        height_margin = jp.maximum(0.145 - base_height, 0.0)
        yaw_progress = jp.sign(command[2]) * gyro[2]
        rewards.update(
            tracking_xy=jp.nan_to_num(
                jp.exp(-xy_error / self._config.reward_config.tracking_sigma)
            ),
            lin_vel_xy_error=jp.nan_to_num(xy_error),
            yaw_error=jp.nan_to_num(yaw_error),
            upright=jp.nan_to_num(jp.square(jp.clip(gravity[2], 0.0, 1.0))),
            vertical_velocity=jp.nan_to_num(jp.square(local_velocity[2])),
            angular_xy=jp.nan_to_num(jp.sum(jp.square(gyro[:2]))),
            command_progress=jp.nan_to_num(command_progress),
            normalized_progress=jp.nan_to_num(normalized_progress),
            progress_shortfall=jp.nan_to_num(progress_shortfall),
            wrong_way=jp.nan_to_num(wrong_way),
            overspeed=jp.nan_to_num(overspeed),
            tilt_margin=jp.nan_to_num(tilt_margin),
            height_margin=jp.nan_to_num(height_margin),
            yaw_progress=jp.nan_to_num(yaw_progress),
            speed_limit=jp.nan_to_num(
                jp.square(jp.maximum(jp.linalg.norm(local_velocity[:2]) - 0.22, 0.0))
            ),
            yaw_limit=jp.nan_to_num(
                jp.square(jp.maximum(jp.abs(gyro[2]) - 0.50, 0.0))
            ),
            tilt=jp.nan_to_num(jp.square(1.0 - jp.clip(gravity[2], -1.0, 1.0))),
            termination=done,
        )
        return rewards
