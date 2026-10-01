"""Measure how closely an ONNX policy reproduces the clipped reference action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
OFFICIAL = ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
DEFAULT_MODEL = ROOT / "training/backward_teacher_v20/final.onnx"
REF_TO_ACT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=-0.074)
    args = parser.parse_args()

    simulation = DuckSimulation(REPO, OFFICIAL, warmup_s=3.0)
    sim = simulation.sim
    sim.policy = OnnxInfer(str(args.model), awd=True)
    sim.commands = [args.speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    default = np.asarray(sim.default_actuator, dtype=float)
    actions: list[np.ndarray] = []
    targets_by_shift: dict[int, list[np.ndarray]] = {i: [] for i in range(-3, 4)}
    local_velocities: list[np.ndarray] = []
    up_values: list[float] = []

    for _ in range(round(args.duration_s / 0.02)):
        for _ in range(sim.decimation):
            mujoco.mj_step(sim.model, sim.data)
        sim.imitation_i = (sim.imitation_i + 1) % sim.PRM.nb_steps_in_period
        angle = sim.imitation_i / sim.PRM.nb_steps_in_period * 2.0 * np.pi
        sim.imitation_phase = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        obs = sim.get_obs(sim.data, sim.commands)
        action = np.asarray(sim.policy.infer(obs), dtype=float)
        actions.append(action)
        for shift in targets_by_shift:
            ref = np.asarray(
                sim.PRM.get_reference_motion(
                    args.speed, 0.0, 0.0, sim.imitation_i + shift
                ),
                dtype=float,
            )
            ref_action = np.clip(
                (ref[REF_TO_ACT] - default) / sim.action_scale, -1.0, 1.0
            )
            targets_by_shift[shift].append(ref_action)

        sim.last_last_last_action = sim.last_last_action.copy()
        sim.last_last_action = sim.last_action.copy()
        sim.last_action = action.copy()
        target = default + action * sim.action_scale
        max_change = sim.max_motor_velocity * 0.02
        sim.motor_targets = np.clip(
            target,
            sim.prev_motor_targets - max_change,
            sim.prev_motor_targets + max_change,
        )
        sim.prev_motor_targets = sim.motor_targets.copy()
        sim.data.ctrl[:] = sim.motor_targets
        local_velocities.append(np.asarray(sim.get_linvel(sim.data), dtype=float))
        up_z = float(simulation.status()["up_vector_z"])
        up_values.append(up_z)
        if up_z < 0.5 or float(sim.data.qpos[2]) < 0.08:
            break

    action_array = np.asarray(actions)
    velocity_array = np.asarray(local_velocities)
    result = {
        "model": str(args.model),
        "steps": len(actions),
        "duration_s": round(len(actions) * 0.02, 3),
        "fell": bool(up_values[-1] < 0.5 or float(sim.data.qpos[2]) < 0.08),
        "final_up_z": round(up_values[-1], 6),
        "minimum_up_z": round(min(up_values), 6),
        "mean_local_velocity_mps": np.mean(velocity_array, axis=0).round(6).tolist(),
        "action_abs_mean": round(float(np.mean(np.abs(action_array))), 6),
        "action_saturation_fraction": round(
            float(np.mean(np.abs(action_array) >= 0.99)), 6
        ),
        "teacher_mse_by_phase_shift": {
            str(shift): round(
                float(np.mean(np.square(action_array - np.asarray(targets)))), 6
            )
            for shift, targets in targets_by_shift.items()
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
