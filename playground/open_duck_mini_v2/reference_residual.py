"""Simulation-only residual stabilization around an untruncated gait reference.

The ONNX actor from this task emits residuals, not legacy motor actions. It
must be evaluated with the matching reference decoder; never deploy it through
the ordinary home + 0.25 * action interface.
"""

import jax.numpy as jp

from .focused_skill import FocusedSkill


class ReferenceResidualBackward(FocusedSkill):
    def __init__(self, *, residual_gain: float, ramp_s: float, **kwargs):
        if not 0.0 < residual_gain <= 0.15:
            raise ValueError("residual_gain must be in (0, 0.15] rad")
        if ramp_s < 0.0:
            raise ValueError("ramp_s must be non-negative")
        self.residual_gain = residual_gain
        self.reference_ramp_s = ramp_s
        super().__init__(skill="backward", **kwargs)
        joint_ids = self.mj_model.actuator_trnid[:, 0]
        self.target_lower = jp.asarray(self.mj_model.jnt_range[joint_ids, 0])
        self.target_upper = jp.asarray(self.mj_model.jnt_range[joint_ids, 1])

    def reference_target(self, command, phase_i, time_s):
        reference = self.PRM.get_reference_motion(*command[:3], phase_i)
        joints = jp.concatenate([reference[:9], reference[11:16]])
        ramp = (
            jp.clip(time_s / self.reference_ramp_s, 0.0, 1.0)
            if self.reference_ramp_s > 0.0 else 1.0
        )
        # Project the reference BEFORE adding feedback. Otherwise a target
        # far beyond a knee limit would consume the whole residual authority,
        # making every actor output map to the same saturated joint target.
        return jp.clip(
            self._default_actuator + ramp * (joints - self._default_actuator),
            self.target_lower, self.target_upper,
        )

    def step(self, state, residual):
        # Use the phase visible to the actor. Joystick.step advances the phase
        # before rewards, but the action selected now belongs to the old phase.
        target = self.reference_target(
            state.info["command"], state.info["imitation_i"], state.data.time
        )
        target = jp.clip(
            target + self.residual_gain * residual,
            self.target_lower, self.target_upper,
        )
        motor_action = (target - self._default_actuator) / self._config.action_scale
        return super().step(state, motor_action)

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        reference_i = (info["imitation_i"] - 1) % self.PRM.nb_steps_in_period
        target = self.reference_target(info["command"], reference_i, data.time - self.dt)
        full_reference_action = (target - self._default_actuator) / self._config.action_scale
        action_mse = jp.mean(jp.square(action - full_reference_action))
        rewards["action_imitation_error"] = action_mse
        rewards["action_imitation"] = jp.exp(-action_mse / 0.25)
        return rewards
