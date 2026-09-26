"""Record a reverse-policy failure without changing the robot or the policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from diagnostics.backward_blend_policy import BlendedBackwardPolicy
from diagnostics.diagnose_turn_four_hypotheses import foot_contact_sample
from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
OFFICIAL = ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"


def sensor(sim, name: str) -> np.ndarray:
    index = sim.model.sensor(name).id
    address = int(sim.model.sensor_adr[index])
    size = int(sim.model.sensor_dim[index])
    return np.asarray(sim.data.sensordata[address:address + size], dtype=float).copy()


def orientation(qpos: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in qpos[3:7])
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--reverse-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=-0.074)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--no-motor-slew-limit", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not -0.15 <= args.speed < 0:
        parser.error("--speed must be negative and at least -0.15 m/s")
    if not 0 <= args.reverse_weight <= 1:
        parser.error("--reverse-weight must be in [0, 1]")
    if args.reverse_weight != 1 and not args.base_model:
        parser.error("--base-model is needed for blending")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    simulation = DuckSimulation(REPO, OFFICIAL, output_root=args.output_dir, warmup_s=3.0)
    sim = simulation.sim
    candidate = OnnxInfer(str(args.model), awd=True)
    sim.policy = (
        BlendedBackwardPolicy(OnnxInfer(str(args.base_model), awd=True), candidate,
                              args.reverse_weight)
        if args.base_model else candidate
    )
    if args.no_motor_slew_limit:
        sim.max_motor_velocity = float("inf")
    rng = np.random.default_rng(args.seed)
    sim.data.qvel[:] += rng.uniform(-0.02, 0.02, size=sim.model.nv)
    mujoco.mj_forward(sim.model, sim.data)
    start = np.asarray(sim.get_floating_base_qpos(sim.data.qpos), dtype=float).copy()
    start_yaw = orientation(start)[2]
    command = [args.speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sim.commands = command
    dt = sim.sim_dt * sim.decimation
    joint_ids = np.asarray(sim.model.actuator_trnid[:, 0], dtype=int)
    joint_addresses = np.asarray(sim.model.jnt_qposadr[joint_ids], dtype=int)
    # Resolve actual foot geom names via the same constant as contact sampling.
    from playground.open_duck_mini_v2 import constants
    foot_ids = [sim.model.geom(constants.LEFT_FEET_GEOMS[0]).id,
                sim.model.geom(constants.RIGHT_FEET_GEOMS[0]).id]
    floor_id = sim.model.geom("floor").id

    rows: list[dict] = []
    for step in range(round(args.duration_s / dt)):
        for _ in range(sim.decimation):
            mujoco.mj_step(sim.model, sim.data)
        sim.imitation_i = (sim.imitation_i + sim.phase_frequency_factor) % sim.PRM.nb_steps_in_period
        angle = sim.imitation_i / sim.PRM.nb_steps_in_period * 2 * np.pi
        sim.imitation_phase = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        obs = np.asarray(sim.get_obs(sim.data, command), dtype=np.float32)
        action = np.asarray(sim.policy.infer(obs), dtype=np.float32)
        requested = sim.default_actuator + action * sim.action_scale
        max_change = sim.max_motor_velocity * dt
        applied = np.clip(requested, sim.prev_motor_targets - max_change,
                          sim.prev_motor_targets + max_change)
        base = np.asarray(sim.get_floating_base_qpos(sim.data.qpos), dtype=float).copy()
        roll, pitch, yaw = orientation(base)
        contacts = foot_contact_sample(sim)
        feet = np.asarray(sim.data.geom_xpos[foot_ids], dtype=float)
        com = np.asarray(sim.data.subtree_com[sim.model.body("base").id], dtype=float)
        support_x = [
            float(np.dot(np.asarray(sim.data.contact[i].pos[:2]) - com[:2],
                         [math.cos(start_yaw), math.sin(start_yaw)]))
            for i in range(sim.data.ncon)
            if floor_id in (int(sim.data.contact[i].geom1), int(sim.data.contact[i].geom2))
            and any(foot_id in (int(sim.data.contact[i].geom1),
                                int(sim.data.contact[i].geom2)) for foot_id in foot_ids)
        ]
        actual_joints = np.asarray(sim.data.qpos[joint_addresses], dtype=float)
        dx, dy = base[0] - start[0], base[1] - start[1]
        forward = math.cos(start_yaw) * dx + math.sin(start_yaw) * dy
        row = {
            "time_s": round((step + 1) * dt, 4),
            "phase": float(sim.imitation_i / sim.PRM.nb_steps_in_period),
            "base_height_m": float(base[2]),
            "forward_displacement_m": float(forward),
            "up_z": float(sensor(sim, "upvector")[2]),
            "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw - start_yaw),
            "local_vx_mps": float(sensor(sim, "local_linvel")[0]),
            "pitch_rate_rad_s": float(sensor(sim, "gyro")[1]),
            "com_rear_support_margin_m": -min(support_x) if support_x else None,
            "com_front_support_margin_m": max(support_x) if support_x else None,
            "left_contact_n": contacts["left"]["normal_n"],
            "right_contact_n": contacts["right"]["normal_n"],
            "left_foot_x_rel_m": float(feet[0, 0] - base[0]),
            "right_foot_x_rel_m": float(feet[1, 0] - base[0]),
            "max_slew_clip_rad": float(np.max(np.abs(requested - applied))),
            "slew_clipped_joints": int(np.sum(np.abs(requested - applied) > 1e-6)),
            "joint_tracking_mae_rad": float(np.mean(np.abs(actual_joints - applied))),
            "joint_tracking_max_rad": float(np.max(np.abs(actual_joints - applied))),
            "action_max_abs": float(np.max(np.abs(action))),
        }
        rows.append(row)
        sim.last_last_last_action = sim.last_last_action.copy()
        sim.last_last_action = sim.last_action.copy()
        sim.last_action = action.copy()
        sim.motor_targets = applied
        sim.prev_motor_targets = applied.copy()
        sim.data.ctrl[:] = applied
        if row["up_z"] < 0.5 or row["base_height_m"] < 0.08:
            break

    trajectory = args.output_dir / "trajectory.json"
    trajectory.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    windows = {}
    for label, selected in (
        ("first_second", [r for r in rows if r["time_s"] <= 1.0]),
        ("prefall_half_second", [r for r in rows if r["time_s"] >= rows[-1]["time_s"] - 0.5]),
    ):
        windows[label] = {
            key: round(float(np.mean([r[key] for r in selected])), 6)
            for key in ("up_z", "pitch_deg", "roll_deg", "local_vx_mps",
                        "left_contact_n", "right_contact_n", "max_slew_clip_rad",
                        "joint_tracking_mae_rad")
        }
    summary = {
        "model": str(args.model), "base_model": str(args.base_model) if args.base_model else None,
        "reverse_weight": args.reverse_weight, "seed": args.seed,
        "motor_slew_limit_enabled": not args.no_motor_slew_limit,
        "command_mps": args.speed, "duration_s": rows[-1]["time_s"],
        "fallen": rows[-1]["up_z"] < 0.5 or rows[-1]["base_height_m"] < 0.08,
        "forward_displacement_m": rows[-1]["forward_displacement_m"],
        "min_up_z": min(r["up_z"] for r in rows),
        "max_abs_pitch_deg": max(abs(r["pitch_deg"]) for r in rows),
        "max_abs_roll_deg": max(abs(r["roll_deg"]) for r in rows),
        "max_slew_clip_rad": max(r["max_slew_clip_rad"] for r in rows),
        "max_joint_tracking_error_rad": max(r["joint_tracking_max_rad"] for r in rows),
        "windows": windows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
