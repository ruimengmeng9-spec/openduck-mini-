"""Probe whether mirroring the policy's positive turn repairs negative yaw.

This is a diagnostic, not a deployable controller. The joint parity follows
the home-pose world joint axes in the v2 MJCF and uses offsets around the
keyframe actuator pose. All runs use untouched XML contact friction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
OFFICIAL = ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
DEFAULT_MODEL = ROOT / "training/official_seed_turn_balance_v5_yaw_error/final.onnx"

JOINT_SWAP = np.array([9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4])
JOINT_SIGN = np.array([-1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1])
COMMAND_SIGN = np.array([1, -1, -1, 1, 1, -1, -1])
GYRO_SIGN = np.array([-1, 1, -1])
ACCEL_SIGN = np.array([1, -1, 1])


def mirror_joint(values: np.ndarray) -> np.ndarray:
    return JOINT_SIGN * values[JOINT_SWAP]


def mirror_obs(obs: np.ndarray, default: np.ndarray, flip_phase: bool) -> np.ndarray:
    out = obs.copy()
    out[0:3] = GYRO_SIGN * obs[0:3]
    out[3:6] = ACCEL_SIGN * obs[3:6]
    out[6:13] = COMMAND_SIGN * obs[6:13]
    for start in (13, 27, 41, 55, 69):
        out[start : start + 14] = mirror_joint(obs[start : start + 14])
    out[83:97] = default + mirror_joint(obs[83:97] - default)
    out[97:99] = obs[97:99][::-1]
    out[99:101] = -obs[99:101] if flip_phase else obs[99:101]
    return out


class MirrorNegativePolicy:
    """Diagnostic adapter: reflect only observations with negative yaw commands."""

    def __init__(
        self, policy: OnnxInfer, default: np.ndarray, flip_phase: bool, blend: float = 1.0
    ):
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        self.policy = policy
        self.default = np.asarray(default, dtype=np.float32)
        self.flip_phase = flip_phase
        self.blend = blend

    def infer(self, obs: np.ndarray) -> np.ndarray:
        if float(obs[8]) < -0.01:
            mirrored = mirror_obs(obs, self.default, self.flip_phase)
            mirrored_action = mirror_joint(
                np.asarray(self.policy.infer(mirrored), dtype=np.float32)
            )
            if self.blend == 1.0:
                return mirrored_action
            original_action = np.asarray(self.policy.infer(obs), dtype=np.float32)
            return (1.0 - self.blend) * original_action + self.blend * mirrored_action
        return self.policy.infer(obs)


def yaw(qpos: np.ndarray) -> float:
    qw, qx, qy, qz = (float(x) for x in qpos[3:7])
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def run(
    model_path: Path, out: Path, mode: str, command_yaw: float,
    blend: float = 0.87,
) -> dict:
    simulation = DuckSimulation(REPO, OFFICIAL, output_root=out, warmup_s=0.0)
    sim = simulation.sim
    sim.policy = OnnxInfer(str(model_path), awd=True)
    dt = sim.sim_dt * sim.decimation
    default = np.asarray(sim.default_actuator, dtype=np.float32)
    flip_phase = mode == "mirror_flip_phase"
    probe = np.arange(101, dtype=np.float32) / 100
    assert np.allclose(mirror_joint(mirror_joint(default)), default)
    assert np.allclose(
        mirror_obs(mirror_obs(probe, default, flip_phase), default, flip_phase), probe
    )

    def step(command: list[float], mirror: bool) -> None:
        for _ in range(sim.decimation):
            mujoco.mj_step(sim.model, sim.data)
        sim.imitation_i = (
            sim.imitation_i + sim.phase_frequency_factor
        ) % sim.PRM.nb_steps_in_period
        angle = float(sim.imitation_i / sim.PRM.nb_steps_in_period) * 2 * np.pi
        sim.imitation_phase = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        sim.commands = command
        obs = np.asarray(sim.get_obs(sim.data, command), dtype=np.float32)
        policy_obs = mirror_obs(obs, default, flip_phase) if mirror else obs
        action = np.asarray(sim.policy.infer(policy_obs), dtype=np.float32)
        if mirror:
            action = mirror_joint(action)
        if mode == "mirror_blend":
            base_action = np.asarray(sim.policy.infer(obs), dtype=np.float32)
            action = (1.0 - blend) * base_action + blend * action
        requested = default + action * sim.action_scale
        max_change = sim.max_motor_velocity * dt
        applied = np.clip(
            requested,
            sim.prev_motor_targets - max_change,
            sim.prev_motor_targets + max_change,
        )
        sim.last_last_last_action = sim.last_last_action.copy()
        sim.last_last_action = sim.last_action.copy()
        sim.last_action = action.copy()
        sim.motor_targets = applied
        sim.prev_motor_targets = applied.copy()
        sim.data.ctrl[:] = applied

    for _ in range(round(3.0 / dt)):
        step([0.0] * 7, mirror=False)
    start_yaw = yaw(sim.get_floating_base_qpos(sim.data.qpos))
    up_min = 1.0
    completed = 0
    for _ in range(round(5.0 / dt)):
        step([0.0, 0.0, command_yaw, 0.0, 0.0, 0.0, 0.0], mirror=mode != "normal")
        base_qpos = sim.get_floating_base_qpos(sim.data.qpos)
        up = float(1.0 - 2.0 * (base_qpos[4] ** 2 + base_qpos[5] ** 2))
        up_min = min(up_min, up)
        completed += 1
        if up < 0.5:
            break
    end_yaw = yaw(sim.get_floating_base_qpos(sim.data.qpos))
    angle = math.atan2(math.sin(end_yaw - start_yaw), math.cos(end_yaw - start_yaw))
    return {
        "mode": mode,
        "blend": blend if mode == "mirror_blend" else None,
        "command_yaw_rad_s": command_yaw,
        "completed_s": round(completed * dt, 4),
        "heading_change_deg": round(math.degrees(angle), 4),
        "minimum_up_z": round(up_min, 5),
        "fallen": up_min < 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commands", type=float, nargs="+", default=[-0.15, 0.15])
    parser.add_argument(
        "--modes", nargs="+",
        default=["normal", "mirror_same_phase", "mirror_flip_phase"],
        choices=("normal", "mirror_same_phase", "mirror_flip_phase", "mirror_blend"),
    )
    parser.add_argument("--blend", type=float, default=0.87)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for command in args.commands:
        for mode in args.modes:
            row = run(args.model, args.output_dir, mode, command, args.blend)
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.output_dir / "results.json").write_text(
        json.dumps({"model": str(args.model), "rows": rows}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
