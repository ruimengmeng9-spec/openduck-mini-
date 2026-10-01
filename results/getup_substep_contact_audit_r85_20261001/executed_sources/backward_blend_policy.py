"""Simulation-only blend between a balanced base and a reverse-gait candidate."""

from __future__ import annotations

import numpy as np


class BlendedBackwardPolicy:
    def __init__(self, base: object, reverse: object, reverse_weight: float):
        if not 0.0 <= reverse_weight <= 1.0:
            raise ValueError("reverse_weight must be in [0, 1]")
        self.base = base
        self.reverse = reverse
        self.reverse_weight = reverse_weight

    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (101,) or not np.isfinite(obs).all():
            raise ValueError("Expected 101 finite observation values")
        base = np.asarray(self.base.infer(obs), dtype=np.float32)
        reverse = np.asarray(self.reverse.infer(obs), dtype=np.float32)
        if base.shape != (14,) or reverse.shape != (14,):
            raise ValueError("Policies must return 14 joint actions")
        if not np.isfinite(base).all() or not np.isfinite(reverse).all():
            raise ValueError("Policy returned non-finite joint actions")
        if float(obs[6]) >= 0.0:
            return base
        return np.asarray(
            (1.0 - self.reverse_weight) * base + self.reverse_weight * reverse,
            dtype=np.float32,
        )


class PitchGuardBackwardPolicy(BlendedBackwardPolicy):
    """Diagnostic only: fade a reverse policy when measured pitch is negative."""

    def __init__(self, base, reverse, pitch_degrees, full_reverse_deg,
                 full_guard_deg, guard_reverse_weight):
        super().__init__(base, reverse, guard_reverse_weight)
        if not full_guard_deg < full_reverse_deg:
            raise ValueError("full_guard_deg must be below full_reverse_deg")
        self.pitch_degrees = pitch_degrees
        self.full_reverse_deg = full_reverse_deg
        self.full_guard_deg = full_guard_deg
        self.guard_reverse_weight = guard_reverse_weight

    def infer(self, obs: np.ndarray) -> np.ndarray:
        pitch = float(self.pitch_degrees())
        if not np.isfinite(pitch):
            raise ValueError("Pitch must be finite")
        fraction = float(np.clip(
            (pitch - self.full_guard_deg)
            / (self.full_reverse_deg - self.full_guard_deg), 0.0, 1.0
        ))
        self.reverse_weight = self.guard_reverse_weight + (1.0 - self.guard_reverse_weight) * fraction
        # Apply a temporary weight; never let a previous step's weight become
        # the next step's guard floor.
        try:
            return super().infer(obs)
        finally:
            self.reverse_weight = self.guard_reverse_weight
