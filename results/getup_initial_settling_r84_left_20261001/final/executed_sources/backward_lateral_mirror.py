"""Diagnostic left/right policy averaging for straight reverse commands.

This is only a closed-loop simulation probe, not a deployable controller.
"""

from __future__ import annotations

import numpy as np

from diagnostics.test_turn_mirror_policy import mirror_joint, mirror_obs


class LateralMirrorBlendPolicy:
    def __init__(self, policy, default: np.ndarray, weight: float,
                 flip_phase: bool = False):
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        self.policy = policy
        self.default = np.asarray(default, dtype=np.float32)
        self.weight = weight
        self.flip_phase = flip_phase

    def infer(self, obs: np.ndarray) -> np.ndarray:
        original = np.asarray(self.policy.infer(obs), dtype=np.float32)
        if self.weight == 0.0:
            return original
        mirrored_obs = mirror_obs(
            np.asarray(obs, dtype=np.float32), self.default, self.flip_phase
        )
        reflected = mirror_joint(
            np.asarray(self.policy.infer(mirrored_obs), dtype=np.float32)
        )
        return ((1.0 - self.weight) * original + self.weight * reflected).astype(
            np.float32
        )


class AlternatingMirrorPolicy(LateralMirrorBlendPolicy):
    """Switch between original and reflected gait, with brief soft transitions."""

    def __init__(self, policy, default: np.ndarray, time_fn, period_s: float,
                 original_fraction: float = 0.45, transition_s: float = 0.08,
                 flip_phase: bool = False):
        super().__init__(policy, default, 0.0, flip_phase)
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        if not 0.0 < original_fraction < 1.0:
            raise ValueError("original_fraction must be between 0 and 1")
        if not 0.0 <= transition_s < min(
            period_s * original_fraction, period_s * (1.0 - original_fraction)
        ):
            raise ValueError("transition_s must fit inside each gait interval")
        self.time_fn = time_fn
        self.period_s = period_s
        self.original_fraction = original_fraction
        self.transition_s = transition_s
        self.start_s = float(time_fn())

    def infer(self, obs: np.ndarray) -> np.ndarray:
        phase_s = (float(self.time_fn()) - self.start_s) % self.period_s
        switch_s = self.period_s * self.original_fraction
        if self.transition_s == 0.0:
            self.weight = float(phase_s >= switch_s)
        elif phase_s < switch_s - self.transition_s:
            self.weight = 0.0
        elif phase_s < switch_s:
            self.weight = (phase_s - (switch_s - self.transition_s)) / self.transition_s
        elif phase_s < self.period_s - self.transition_s:
            self.weight = 1.0
        else:
            self.weight = 1.0 - (phase_s - (self.period_s - self.transition_s)) / self.transition_s
        return super().infer(obs)
