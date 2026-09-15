"""Headless Open Duck Mini simulator with bounded high-level actions.

This module deliberately reuses the repository's official ``MjInfer`` class so
the 101-value observation layout and the ONNX policy stay identical to the
validated native-MuJoCo rollout.  The language model never sees joint-level
controls; it can only issue short, bounded velocity commands.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer
from playground.common.onnx_infer import OnnxInfer


class DuckSimulation:
    """Persistent, headless simulation controlled at the policy's 50 Hz rate."""

    CONTROL_DT = 0.02
    MAX_DURATION_S = 5.0
    MAX_WALK_SPEED_MPS = 0.15
    MAX_TURN_RATE_RAD_S = 0.30
    FALL_UP_Z = 0.50
    FALL_BASE_HEIGHT_M = 0.08

    def __init__(
        self,
        repo_root: str | Path,
        onnx_model: str | Path,
        output_root: str | Path | None = None,
        warmup_s: float = 3.0,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.onnx_model = Path(onnx_model).expanduser().resolve()
        turn_model_value = os.environ.get("OPEN_DUCK_TURN_ONNX_MODEL", "").strip()
        self.turn_onnx_model = (
            Path(turn_model_value).expanduser().resolve()
            if turn_model_value
            else None
        )
        model_path = (
            self.repo_root
            / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        )
        reference_path = (
            self.repo_root
            / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
        )
        required_paths = [model_path, reference_path, self.onnx_model]
        if self.turn_onnx_model is not None:
            required_paths.append(self.turn_onnx_model)
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Required Open Duck file not found: {path}")

        self.sim = MjInfer(
            str(model_path),
            str(reference_path),
            str(self.onnx_model),
            standing=False,
        )
        self._walk_policy = self.sim.policy
        self._turn_policy = (
            OnnxInfer(str(self.turn_onnx_model), awd=True)
            if self.turn_onnx_model is not None
            else self._walk_policy
        )
        self._active_policy_name = "walk"
        self._physics_steps = 0
        self._last_action_name = "reset"
        self._last_command = np.zeros(7, dtype=np.float32)

        root = (
            Path(output_root).expanduser().resolve()
            if output_root
            else self.repo_root.parent.parent / "outputs"
        )
        stamp = datetime.now().strftime("agent_%Y%m%d_%H%M%S_%f")
        self.output_dir = root / stamp
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.log_path = self.output_dir / "actions.jsonl"

        self.reset()
        if warmup_s > 0:
            self.stand(warmup_s, source="startup_warmup")

    def reset(self) -> dict[str, Any]:
        """Restore the exact home keyframe and clear policy history."""
        sim = self.sim
        sim.data.qpos[:] = sim.model.keyframe("home").qpos
        sim.data.qvel[:] = 0.0
        sim.data.ctrl[:] = sim.default_actuator
        sim.motor_targets = np.array(sim.default_actuator, dtype=np.float64).copy()
        sim.prev_motor_targets = sim.motor_targets.copy()
        sim.last_action = np.zeros(sim.num_dofs, dtype=np.float32)
        sim.last_last_action = np.zeros(sim.num_dofs, dtype=np.float32)
        sim.last_last_last_action = np.zeros(sim.num_dofs, dtype=np.float32)
        sim.commands = [0.0] * 7
        sim.imitation_i = 0
        sim.imitation_phase = np.zeros(2, dtype=np.float32)
        sim.phase_frequency_factor = 1.0
        self._physics_steps = 0
        self._last_action_name = "reset"
        self._last_command[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        result = self.status()
        self._log({"event": "reset", **result})
        return result

    def stand(self, duration_s: float = 1.0, source: str = "tool") -> dict[str, Any]:
        """Maintain balance at zero commanded velocity for a bounded duration."""
        self._select_policy("stand")
        return self._run_command(
            "stand", [0.0] * 7, duration_s=duration_s, source=source
        )

    def walk(self, speed_mps: float, duration_s: float) -> dict[str, Any]:
        """Walk forward/backward; speed is clipped to the validated range."""
        self._select_policy("walk")
        speed = self._clip_finite(
            speed_mps,
            -self.MAX_WALK_SPEED_MPS,
            self.MAX_WALK_SPEED_MPS,
            "speed_mps",
        )
        return self._run_command(
            "walk", [speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration_s
        )

    def turn(self, yaw_rate_rad_s: float, duration_s: float) -> dict[str, Any]:
        """Turn in place; rate is clipped to the conservative tested range."""
        self._select_policy("turn")
        rate = self._clip_finite(
            yaw_rate_rad_s,
            -self.MAX_TURN_RATE_RAD_S,
            self.MAX_TURN_RATE_RAD_S,
            "yaw_rate_rad_s",
        )
        result = self._run_command(
            "turn", [0.0, 0.0, rate, 0.0, 0.0, 0.0, 0.0], duration_s
        )
        if rate < 0 and self._turn_policy is self._walk_policy:
            result["calibration_note"] = (
                "Negative turning is weaker than positive turning in the current "
                "policy (measured about -0.04 to -0.06 rad/s for a -0.3 command)."
            )
        return result

    def stop(self, settle_s: float = 0.5, source: str = "tool") -> dict[str, Any]:
        """Zero the velocity command and let the balance policy settle."""
        settle = self._clip_finite(settle_s, 0.2, 2.0, "settle_s")
        return self._run_command(
            "stop", [0.0] * 7, duration_s=settle, source=source
        )

    def status(self) -> dict[str, Any]:
        sim = self.sim
        base = np.asarray(sim.get_floating_base_qpos(sim.data.qpos), dtype=float)
        up = self._sensor("upvector")
        linvel = self._sensor("local_linvel")
        up_z = float(up[2])
        height = float(base[2])
        finite = bool(
            np.all(np.isfinite(base))
            and np.all(np.isfinite(up))
            and np.all(np.isfinite(linvel))
        )
        fallen = (not finite) or up_z < self.FALL_UP_Z or height < self.FALL_BASE_HEIGHT_M
        return {
            "simulation_time_s": round(float(sim.data.time), 6),
            "base_position_m": np.round(base[:3], 6).tolist(),
            "local_velocity_mps": np.round(linvel, 6).tolist(),
            "up_vector_z": round(up_z, 6),
            "fallen": bool(fallen),
            "last_action": self._last_action_name,
            "last_command": np.round(self._last_command, 6).tolist(),
            "active_policy": self._active_policy_name,
        }

    def _select_policy(self, action_name: str) -> None:
        """Use the stable locomotion policy for walking and the trained turn policy."""
        if action_name == "turn":
            policy = self._turn_policy
            policy_name = "turn" if policy is not self._walk_policy else "walk"
        elif action_name in {"stand", "walk"}:
            policy = self._walk_policy
            policy_name = "walk"
        else:
            # Stop settles with whichever policy produced the preceding motion.
            return
        self.sim.policy = policy
        self._active_policy_name = policy_name

    def _run_command(
        self,
        action_name: str,
        command: list[float],
        duration_s: float,
        source: str = "tool",
    ) -> dict[str, Any]:
        duration = self._clip_finite(
            duration_s, self.CONTROL_DT, self.MAX_DURATION_S, "duration_s"
        )
        control_steps = max(1, int(round(duration / self.CONTROL_DT)))
        sim = self.sim
        sim.commands = [float(x) for x in command]
        self._last_command = np.asarray(command, dtype=np.float32)
        self._last_action_name = action_name
        start_time = float(sim.data.time)
        start_position = np.asarray(
            sim.get_floating_base_qpos(sim.data.qpos)[:3], dtype=float
        ).copy()
        interrupted = False

        for _ in range(control_steps):
            for _ in range(sim.decimation):
                mujoco.mj_step(sim.model, sim.data)
                self._physics_steps += 1

            sim.imitation_i = (
                sim.imitation_i + sim.phase_frequency_factor
            ) % sim.PRM.nb_steps_in_period
            angle = sim.imitation_i / sim.PRM.nb_steps_in_period * 2.0 * np.pi
            sim.imitation_phase = np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float32
            )
            observation = sim.get_obs(sim.data, sim.commands)
            action = np.asarray(sim.policy.infer(observation), dtype=np.float32)

            sim.last_last_last_action = sim.last_last_action.copy()
            sim.last_last_action = sim.last_action.copy()
            sim.last_action = action.copy()

            target = sim.default_actuator + action * sim.action_scale
            max_change = sim.max_motor_velocity * (sim.sim_dt * sim.decimation)
            sim.motor_targets = np.clip(
                target,
                sim.prev_motor_targets - max_change,
                sim.prev_motor_targets + max_change,
            )
            sim.prev_motor_targets = sim.motor_targets.copy()
            sim.data.ctrl[:] = sim.motor_targets

            if self.status()["fallen"]:
                interrupted = True
                sim.commands = [0.0] * 7
                self._last_command[:] = 0.0
                break

        end = self.status()
        end_position = np.asarray(end["base_position_m"], dtype=float)
        result = {
            "ok": not interrupted,
            "action": action_name,
            "requested_duration_s": round(duration, 3),
            "executed_duration_s": round(float(sim.data.time) - start_time, 6),
            "command": [round(float(x), 6) for x in command],
            "horizontal_displacement_m": round(
                float(np.linalg.norm(end_position[:2] - start_position[:2])), 6
            ),
            **end,
        }
        if interrupted:
            result["error"] = "Safety stop: the robot is fallen or state is not finite."
        self._log({"event": "action", "source": source, **result})
        return result

    def _sensor(self, name: str) -> np.ndarray:
        sensor_id = mujoco.mj_name2id(
            self.sim.model, mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        if sensor_id < 0:
            raise RuntimeError(f"MuJoCo sensor not found: {name}")
        address = int(self.sim.model.sensor_adr[sensor_id])
        dimension = int(self.sim.model.sensor_dim[sensor_id])
        return np.asarray(
            self.sim.data.sensordata[address : address + dimension], dtype=float
        ).copy()

    @staticmethod
    def _clip_finite(value: float, low: float, high: float, name: str) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        return float(np.clip(numeric, low, high))

    def _log(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
