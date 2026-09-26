"""Simulation-only adapter for a distilled negative-turn action residual."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort


class ActionPolicy(Protocol):
    def infer(self, obs: np.ndarray) -> np.ndarray: ...


class ResidualNegativePolicy:
    def __init__(
        self, base: ActionPolicy, residual_path: Path, command_band: str = "narrow"
    ):
        if command_band not in ("narrow", "wide", "extended"):
            raise ValueError("command_band must be narrow, wide, or extended")
        self.base = base
        self.command_band = command_band
        self.residual = ort.InferenceSession(
            str(residual_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.residual.get_inputs()[0].name

    def _weight(self, yaw: float) -> float:
        if self.command_band == "narrow":
            return float(-0.16 <= yaw <= -0.14)
        speed = -yaw
        if self.command_band == "extended":
            return float(
                np.clip((speed - 0.08) / 0.02, 0.0, 1.0)
                * np.clip((0.20 - speed) / 0.03, 0.0, 1.0)
            )
        return float(
            np.clip((speed - 0.10) / 0.03, 0.0, 1.0)
            * np.clip((0.20 - speed) / 0.03, 0.0, 1.0)
        )

    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (101,) or not np.isfinite(obs).all():
            raise ValueError("Expected 101 finite observation values")
        base_action = np.asarray(self.base.infer(obs), dtype=np.float32)
        if base_action.shape != (14,) or not np.isfinite(base_action).all():
            raise ValueError("Base policy returned an invalid 14-joint action")
        weight = self._weight(float(obs[8]))
        if weight <= 0.0:
            return base_action
        features = np.concatenate([obs, base_action])[None, :]
        delta = np.asarray(
            self.residual.run(None, {self.input_name: features})[0][0],
            dtype=np.float32,
        )
        if delta.shape != (14,) or not np.isfinite(delta).all():
            raise ValueError("Residual policy returned an invalid 14-joint correction")
        return np.clip(base_action + weight * delta, -1.0, 1.0).astype(np.float32)
