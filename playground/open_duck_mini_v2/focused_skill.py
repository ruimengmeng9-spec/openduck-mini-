"""Focused locomotion curricula for extending Open Duck Mini skills."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import walk_turn_stop


def _env_pair(name: str, default_lo: float, default_hi: float) -> tuple[float, float]:
    """Read a two-value range such as "0.50,0.90" from the environment."""
    raw = os.environ.get(name)
    if not raw:
        return default_lo, default_hi
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be 'low,high', got {raw!r}")
    return float(parts[0]), float(parts[1])


def _env_scale(name: str, default: float) -> float:
    """Read a cost magnitude from the environment.

    Values are magnitudes: the skill definition applies the negative sign.  A
    negative input is rejected rather than silently accepted, because
    ``-(-1500)`` turns a stability *cost* into a stability *reward* -- a mistake
    that is invisible in the training log until the return jumps upward.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    value = float(raw)
    if value < 0:
        raise ValueError(
            f"{name} must be a non-negative magnitude (got {value}); "
            f"the sign is applied internally, so a negative value would invert "
            f"this cost into a reward."
        )
    return value


SUPPORTED_SKILLS = ("backward", "lateral", "arc", "turn", "unified", "getup")

# Base height of the robot in the `home` keyframe.  Everything that has to reason
# about "how tall is the robot when it is standing" derives from this value.
STANDING_BASE_HEIGHT = 0.15


def focused_config(skill: str) -> config_dict.ConfigDict:
    """Return rewards and command bounds for one focused curriculum."""
    if skill not in SUPPORTED_SKILLS:
        raise ValueError(f"Unsupported skill {skill!r}; choose from {SUPPORTED_SKILLS}")

    is_getup = skill == "getup"
    config = walk_turn_stop.default_config()
    # Backward uses a staged curriculum.  Keep perturbations gentle until the
    # policy learns a repeatable reverse gait, then add robustness separately.
    # Recovery is also learned with reduced sensor noise: the task itself is
    # already hard enough without a noisy gravity vector.
    if skill == "backward":
        config.noise_config.level = 0.05
    elif is_getup:
        config.noise_config.level = 0.10
    else:
        config.noise_config.level = 0.35
    config.push_config.enable = False
    if skill == "backward":
        # V14 widens the reverse-speed curriculum slightly beyond V13's narrow
        # -0.025..-0.035 window: the target was so small that the progress signal
        # could not outbid the posture costs.  Bounds are stored as a negative
        # [min, max] interval so the runner can override them from the CLI.
        config.lin_vel_x = [-0.05, -0.03]

    # Keep every term positive enough that the upstream reward clipping does not
    # erase the velocity gradient.  Direct squared errors remain small shaping
    # costs, while the exponential terms give a clear improvement near target.
    scales = config.reward_config.scales
    scales.tracking_lin_vel = 0.0
    scales.tracking_ang_vel = 0.0 if is_getup else 14.0
    scales.lin_vel_error = 0.0 if is_getup else -3.0
    scales.tracking_xy = (
        0.0 if is_getup else (60.0 if skill == "backward" else 18.0)
    )
    # Recovery has no velocity command, so this term degenerates into a plain
    # penalty on dragging/sliding the chassis across the floor.
    scales.lin_vel_xy_error = (
        -5.0 if is_getup else (-10.0 if skill == "backward" else -8.0)
    )
    scales.yaw_error = 0.0 if is_getup else -3.0
    # Recovery needs a strong, always-present gradient towards "upright": the
    # squared projection gives 0 while lying and 1 while standing.
    scales.upright = 40.0 if is_getup else (20.0 if skill == "backward" else 10.0)
    scales.vertical_velocity = (
        -1.0 if is_getup else (-5.0 if skill == "backward" else -2.0)
    )
    # Tumbling generates very large gyro values, so a mild cost is enough to
    # discourage spinning without forbidding the vigorous motion recovery needs.
    scales.angular_xy = (
        -0.5 if is_getup else (-2.0 if skill == "backward" else -0.5)
    )
    scales.command_progress = {
        "backward": 0.0,
        "lateral": 80.0,
        "arc": 40.0,
        "turn": 0.0,
        "unified": 40.0,
        "getup": 0.0,
    }[skill]
    scales.yaw_progress = {
        "backward": 0.0,
        "lateral": 0.0,
        "arc": 20.0,
        "turn": 35.0,
        "unified": 20.0,
        "getup": 0.0,
    }[skill]
    scales.speed_limit = -50.0 if is_getup else -300.0
    scales.yaw_limit = 0.0 if is_getup else -20.0
    # V13 stabilised the reverse gait by making the posture costs overwhelming,
    # and the policy responded by collapsing back onto the near-stationary local
    # optimum (measured -0.004 m/s for a -0.04 m/s command).  V14 keeps the
    # stability terms meaningful but no longer dominant, and doubles the progress
    # incentive so that "stand still" is clearly worse than "walk backwards".
    scales.normalized_progress = 80.0 if skill == "backward" else 0.0
    scales.progress_shortfall = -80.0 if skill == "backward" else 0.0
    scales.wrong_way = -80.0 if skill == "backward" else 0.0
    # V12--V14 kept tightening this term and every run collapsed onto the
    # near-stationary optimum.  The reason is arithmetic, not tuning: with a
    # -0.04 m/s target the old band (1.10*|v_cmd| + 0.005 = 0.049 m/s) sits below
    # the natural velocity oscillation of a walking gait, so *moving at all* cost
    # 4000 * overspeed * dt ~ -8 per step while the progress reward was worth
    # about +2.5.  Standing still (overspeed = 0) was therefore strictly better.
    # Weight is cut and the band widened so this only catches a genuine lunge;
    # tracking_xy and lin_vel_xy_error already punish deviation from the command,
    # so nothing here needs to carry the whole burden.
    scales.overspeed = -600.0 if skill == "backward" else 0.0
    # Backward stability group.  V15 relaxed these to break the "stand still"
    # optimum and succeeded at producing a real reverse gait (-0.072 m/s), but
    # the policy then fell after 1.8-2.5 s.  With the incentive direction now
    # correct these can be raised again to buy stability, and they are exposed as
    # environment knobs so a sweep does not require editing this file.
    scales.tilt_margin = -_env_scale("BACKWARD_TILT_MARGIN", 600.0) if skill == "backward" else 0.0
    scales.height_margin = -_env_scale("BACKWARD_HEIGHT_MARGIN", 600.0) if skill == "backward" else 0.0
    scales.tilt = (
        -200.0
        if is_getup
        else (-_env_scale("BACKWARD_TILT", 500.0) if skill == "backward" else -30.0)
    )
    # Recovery never terminates on a fall, so there is nothing to penalise here;
    # an inverted robot is instead pushed back by the `tilt` cost.
    scales.termination = (
        0.0
        if is_getup
        else (-_env_scale("BACKWARD_TERMINATION", 1200.0) if skill == "backward" else -50.0)
    )
    # No free reward for lying still: `alive` must stay 0 for recovery or the
    # optimal policy is to never move.  The `stand_up` term replaces it.
    scales.alive = 0.0 if (skill == "backward" or is_getup) else 8.0
    scales.imitation = {
        # The official imitation target is excellent for balance but tends to
        # anchor a backward command near standing.  Keep a small stabilizing
        # prior while allowing the learned gait to depart from it.  The standing
        # reference is simply wrong for a robot that is on the floor.
        #
        # Backward defaults to 0 because of command quantisation: the reference
        # grid is {-0.148, -0.074, 0.0, +0.074, ...}, so the v13 curriculum of
        # -0.025..-0.035 resolves to dx = 0.000, i.e. the *standing* reference
        # (measured: max|low_cmd - standing| = 0.0000).  Weighting that at 0.25
        # actively pinned the policy to standing.  The dataset does contain real
        # reverse gaits at -0.074 (which differ from +0.074 by 8471 in max
        # absolute coefficient), so a reverse command that lands on the grid can
        # safely use a positive weight -- see BACKWARD_IMITATION.
        "backward": _env_scale("BACKWARD_IMITATION", 0.0),
        "lateral": 0.5,
        "arc": 1.5,
        "turn": 1.0,
        "unified": 1.5,
        "getup": 0.0,
    }[skill]
    scales.stand_still = 0.0 if is_getup else -0.5
    scales.action_rate = -0.4 if is_getup else -0.3
    scales.torques = -5.0e-4
    # Recovery-only terms.  `stand_up` is the product of "is upright" and "is
    # tall", so it is 0 while lying and 1 while standing: it cannot be farmed by
    # balancing on the head or by pressing the chassis against the floor.
    scales.stand_up = 80.0 if is_getup else 0.0
    scales.height_progress = 20.0 if is_getup else 0.0
    scales.joint_velocity = -0.002 if is_getup else 0.0
    config.reward_config.tracking_sigma = 0.0025 if skill == "backward" else 0.015
    # Focused backward learning needs signed feedback: clipping every negative
    # total to zero creates a flat region around the stationary policy.  The same
    # applies to recovery, where the lying-down penalty must survive.
    config.reward_config.reward_floor = (
        -10000.0 if (skill == "backward" or is_getup) else 0.0
    )
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
        # Difficulty knobs for the recovery curriculum.  Exposed through the
        # environment so a sweep can stage the task (easy lean -> full fall)
        # without editing this file.  Defaults are the hard, deployment-relevant
        # setting; a mild value is used to verify the task is learnable at all.
        self._getup_fallen_prob = float(os.environ.get("GETUP_FALLEN_PROB", "0.7"))
        fallen_lo, fallen_hi = _env_pair("GETUP_FALLEN_TILT", 1.40, 1.83)
        lean_lo, lean_hi = _env_pair("GETUP_LEAN_TILT", 0.50, 0.90)
        self._getup_fallen_tilt = (fallen_lo, fallen_hi)
        self._getup_lean_tilt = (lean_lo, lean_hi)
        self._getup_random_joints = os.environ.get("GETUP_RANDOM_JOINTS", "1") != "0"
        super().__init__(
            task=task,
            config=config if config is not None else focused_config(skill),
            config_overrides=config_overrides,
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Start recovery episodes from a randomised fallen pose."""
        state = super().reset(rng)
        if self.skill != "getup":
            return state

        info = dict(state.info)
        # Start with randomised joint angles rather than the standing pose.  A
        # robot dropped on its back with the home (straight-leg) posture has to
        # discover leg folding on its own, and from-scratch PPO never found it:
        # both runs converged to a fixed lying pose.  Sampling inside the joint
        # limits supplies the tucked/lopsided configurations that make pushing
        # off the floor reachable by gradient.
        rng, joint_rng = jax.random.split(rng)
        lowers, uppers = self._soft_lowers, self._soft_uppers
        sampled_joints = jax.random.uniform(
            joint_rng,
            (self._actuators,),
            minval=lowers.astype(jp.float32),
            maxval=uppers.astype(jp.float32),
        )
        joints = jp.where(self._getup_random_joints, sampled_joints, self._default_actuator)
        qpos = self.set_actuator_joints_qpos(joints, state.data.qpos)

        rng, tilt_rng, side_rng, sign_rng, yaw_rng, height_rng = jax.random.split(
            rng, 6
        )
        # Tilt the base and randomise the heading.  Fully fallen attitudes
        # (80--105 deg, covering face down / on the back / either side) cannot be
        # recovered by from-scratch exploration: measured returns sat at exactly
        # one episode's worth of the lying-still tilt penalty (-2900 for 1000
        # steps), i.e. the policy never attempted to rise.  Mixing in milder
        # tilts (29--51 deg) keeps a reachable path to standing so the same
        # network can first learn to push up from a near-standing lean and then
        # extend that skill to the fully fallen poses.
        tilt_rng, mode_rng = jax.random.split(tilt_rng)
        fallen = jax.random.bernoulli(mode_rng, p=self._getup_fallen_prob)
        tilt = jp.where(
            fallen,
            jax.random.uniform(
                tilt_rng, (), minval=self._getup_fallen_tilt[0], maxval=self._getup_fallen_tilt[1]
            ),
            jax.random.uniform(
                tilt_rng, (), minval=self._getup_lean_tilt[0], maxval=self._getup_lean_tilt[1]
            ),
        )
        sign = jp.where(jax.random.bernoulli(sign_rng), 1.0, -1.0)
        side = jax.random.bernoulli(side_rng)
        axis = jp.where(
            side, jp.array([0.0, 1.0, 0.0]), jp.array([1.0, 0.0, 0.0])
        )
        tilt_quat = math.axis_angle_to_quat(axis, sign * tilt)
        yaw = jax.random.uniform(yaw_rng, (), minval=-jp.pi, maxval=jp.pi)
        yaw_quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        # Applying yaw in the world frame last keeps the fallen direction
        # independent of the heading.
        base_quat = math.quat_mul(yaw_quat, tilt_quat)

        # The settled height depends on how far the robot is tilted: a fully
        # fallen robot rests a few centimetres off the floor while a mildly
        # leaning one still stands on extended legs.  Starting slightly above the
        # resting height lets contact resolve gently; starting below it drives the
        # trunk collision box through the floor and ejects the robot.
        fallen_height_rng, lean_height_rng = jax.random.split(height_rng)
        height = jp.where(
            fallen,
            jax.random.uniform(fallen_height_rng, (), minval=0.088, maxval=0.120),
            jax.random.uniform(lean_height_rng, (), minval=0.125, maxval=0.155),
        )
        base_qpos = jp.concatenate([jp.zeros(3).at[2].set(height), base_quat])
        qpos = self.set_floating_base_qpos(base_qpos, qpos)

        qvel = jp.zeros(self.mjx_model.nv)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=joints)

        zeros = jp.zeros(self.mjx_model.nu)
        info["command"] = jp.zeros(7)
        info["last_act"] = zeros
        info["last_last_act"] = zeros
        info["last_last_last_act"] = zeros
        info["motor_targets"] = joints
        info["action_history"] = jp.zeros(
            self._config.noise_config.action_max_delay * self._actuators
        )
        info["imu_history"] = jp.zeros(
            self._config.noise_config.imu_max_delay * 3
        )
        info["feet_air_time"] = jp.zeros(2)
        info["last_contact"] = jp.zeros(2, dtype=bool)
        info["swing_peak"] = jp.zeros(2)
        info["step"] = 0

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        obs = self._get_obs(data, info, contact)
        return state.replace(data=data, obs=obs, info=info)

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample balanced commands without a zero-gradient dead zone."""
        if self.skill == "getup":
            # Recovery is not a tracking task: hold the standing command so the
            # observation layout stays identical to the deployed controller.
            return jp.zeros(7)

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
            # Magnitude comes from the configurable curriculum band so a run can
            # be retargeted without touching this file.
            lo, hi = (abs(float(v)) for v in self._config.lin_vel_x)
            mag = jax.random.uniform(
                x_rng, minval=min(lo, hi), maxval=max(lo, hi)
            )
            moving = jax.random.bernoulli(mode_rng, p=0.95)
            vx = jp.where(moving, -mag, 0.0)
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
        if self.skill == "getup":
            # The whole point of recovery is that the robot starts on the floor,
            # so the upstream "upside down" termination must be disabled.  Only
            # genuinely unusable (non-finite) states end an episode.
            return jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
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
        # Allow the natural velocity oscillation of a gait.  The previous band
        # (1.10 * command + 0.005) was narrower than that oscillation at the
        # low-speed backward curriculum, which made any movement unprofitable.
        allowed_speed = jp.maximum(1.5 * command_speed, 0.15)
        overspeed = jp.maximum(jp.linalg.norm(local_velocity[:2]) - allowed_speed, 0.0)
        tilt_margin = jp.maximum(0.985 - gravity[2], 0.0)
        base_height = self.get_floating_base_qpos(data.qpos)[2]
        height_margin = jp.maximum(0.145 - base_height, 0.0)
        yaw_progress = jp.sign(command[2]) * gyro[2]
        # Recovery shaping: "upright" alone can be satisfied by balancing on the
        # head or the back, so the height ratio is multiplied in.  The product is
        # 0 on the floor and 1 in the home stance.
        upright_signal = jp.square(jp.clip(gravity[2], 0.0, 1.0))
        height_signal = jp.clip(base_height / STANDING_BASE_HEIGHT, 0.0, 1.0)
        rewards.update(
            tracking_xy=jp.nan_to_num(
                jp.exp(-xy_error / self._config.reward_config.tracking_sigma)
            ),
            lin_vel_xy_error=jp.nan_to_num(xy_error),
            yaw_error=jp.nan_to_num(yaw_error),
            upright=jp.nan_to_num(upright_signal),
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
            stand_up=jp.nan_to_num(upright_signal * height_signal),
            height_progress=jp.nan_to_num(height_signal),
            joint_velocity=jp.nan_to_num(
                jp.sum(jp.square(self.get_actuator_joints_qvel(data.qvel)))
            ),
        )
        return rewards
