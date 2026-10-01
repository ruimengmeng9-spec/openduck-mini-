"""Simulation-only per-skill target execution, with unchanged physical limits."""
import numpy as np


class SkillTargetExecutor:
    def __init__(self, sim, dt, lower, upper, backward_tau):
        if not 0 < dt <= .1 or not 0 < backward_tau <= .1:
            raise ValueError('invalid control dt or backward filter tau')
        self.sim, self.dt = sim, dt
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.tau = backward_tau
        self.filtered = sim.prev_motor_targets.copy()
        self.previous_backward = False

    def apply(self, requested_target, raw_action, backward_contract):
        s = self.sim
        requested = np.asarray(requested_target)
        if requested.shape != s.default_actuator.shape or not np.isfinite(requested).all():
            raise ValueError('invalid joint target')
        if not backward_contract:
            raw_action = np.asarray(raw_action, dtype=np.float32)
            if raw_action.shape != requested.shape or not np.isfinite(raw_action).all():
                raise ValueError('legacy policy requires its original normalized action')
        target = np.clip(requested, self.lower, self.upper)
        if backward_contract:
            if not self.previous_backward:
                # Entering backward initializes the filter from the last actual
                # motor command, not an unexecuted legacy policy request.
                self.filtered = s.prev_motor_targets.copy()
            self.filtered += self.dt/(self.tau+self.dt)*(target-self.filtered)
            history_action = ((self.filtered-s.default_actuator)/s.action_scale).astype(np.float32)
        else:
            self.filtered = target.copy()
            history_action = raw_action.copy()
        applied = np.clip(self.filtered, s.prev_motor_targets-s.max_motor_velocity*self.dt,
                          s.prev_motor_targets+s.max_motor_velocity*self.dt)
        s.last_last_last_action = s.last_last_action.copy()
        s.last_last_action = s.last_action.copy()
        s.last_action = history_action
        s.data.ctrl[:] = applied
        s.motor_targets = applied.copy()
        s.prev_motor_targets = applied.copy()
        self.previous_backward = bool(backward_contract)
        return applied.copy()
