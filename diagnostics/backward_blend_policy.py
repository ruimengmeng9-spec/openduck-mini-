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
