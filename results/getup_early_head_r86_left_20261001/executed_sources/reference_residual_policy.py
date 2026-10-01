"""Matching native-MuJoCo decoder for a reference-residual ONNX actor."""

import os

import numpy as np


class ContinuousDxReference:
    """NumPy counterpart of the opt-in JAX continuous-dx reference path."""

    def __init__(self, reference, dx=None, interpolate=None):
        self.reference = reference
        self.nb_steps_in_period = reference.nb_steps_in_period
        self.dx = float(os.environ["REFERENCE_DX"]) if dx is None and "REFERENCE_DX" in os.environ else dx
        self.interpolate = (
            os.environ.get("REFERENCE_DX_INTERPOLATION", "0") == "1"
            if interpolate is None else interpolate
        )

    def get_reference_motion(self, dx, dy, yaw, phase_i):
        dx = float(dx if self.dx is None else self.dx)
        if not self.interpolate:
            return self.reference.get_reference_motion(dx, dy, yaw, phase_i)
        base = self.reference
        _, iy, itheta = base.vel_to_index(dx, dy, yaw)
        grid = np.asarray(base.dxs, dtype=float)
        clipped = np.clip(dx, base.dx_range[0], base.dx_range[1])
        upper = int(np.clip(np.searchsorted(grid, clipped, side="right"), 1, len(grid) - 1))
        lower = upper - 1
        alpha = np.clip((clipped - grid[lower]) / max(grid[upper] - grid[lower], 1e-8), 0.0, 1.0)
        t = np.clip((phase_i % self.nb_steps_in_period) / self.nb_steps_in_period, 0.0, 1.0)
        low = np.asarray(base.sample_polynomial(t, base.data_array[lower][iy][itheta]))
        high = np.asarray(base.sample_polynomial(t, base.data_array[upper][iy][itheta]))
        out = (1.0 - alpha) * low + alpha * high
        out[32:34] = low[32:34]
        return out


class ZeroResidualActor:
    def infer(self, obs):
        return np.zeros(14, dtype=np.float32)


class ReferenceResidualPolicy:
    def __init__(self, actor, sim, residual_gain: float, ramp_s: float):
        if not 0.0 < residual_gain <= 0.15 or ramp_s < 0.0:
            raise ValueError("invalid simulation reference-residual parameters")
        self.actor = actor
        self.sim = sim
        self.reference = ContinuousDxReference(sim.PRM)
        self.gain = residual_gain
        self.ramp_s = ramp_s
        self.start_s = float(sim.data.time)
        joint_ids = sim.model.actuator_trnid[:, 0]
        self.lower = sim.model.jnt_range[joint_ids, 0]
        self.upper = sim.model.jnt_range[joint_ids, 1]

    def infer(self, obs):
        phase = np.arctan2(float(obs[100]), float(obs[99])) % (2.0 * np.pi)
        phase_i = phase / (2.0 * np.pi) * self.sim.PRM.nb_steps_in_period
        reference = np.asarray(self.reference.get_reference_motion(
            *obs[6:9], phase_i
        ), dtype=np.float32)
        joints = np.concatenate([reference[:9], reference[11:16]])
        ramp = (
            np.clip((float(self.sim.data.time) - self.start_s) / self.ramp_s, 0.0, 1.0)
            if self.ramp_s > 0.0 else 1.0
        )
        target = np.clip(
            self.sim.default_actuator + ramp * (joints - self.sim.default_actuator),
            self.lower, self.upper,
        )
        residual = np.asarray(self.actor.infer(obs), dtype=np.float32)
        target = np.clip(target + self.gain * residual, self.lower, self.upper)
        return ((target - self.sim.default_actuator) / self.sim.action_scale).astype(np.float32)
