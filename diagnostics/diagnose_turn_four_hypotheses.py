"""Test four suspected causes of asymmetric in-place turning.

This native-MuJoCo diagnostic measures motor target slew clipping, foot contact
forces/yaw moments, left-right load balance in the settled policy stance, and
the sensitivity of heading response to sliding friction. Reward clipping is
tested separately by ``diagnose_turn_reward_clipping.py`` in the JAX task.
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
from playground.open_duck_mini_v2 import constants


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
OFFICIAL = ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
DEFAULT_MODEL = ROOT / "training/official_seed_turn_balance_v2/final.onnx"
DEFAULT_OUTPUT = ROOT / "outputs/turn_four_hypotheses_20260923"


def yaw_of(qpos: np.ndarray) -> float:
    qw, qx, qy, qz = (float(value) for value in qpos[3:7])
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def foot_contact_sample(sim) -> dict[str, dict[str, float]]:
    model, data = sim.model, sim.data
    floor_id = model.geom("floor").id
    foot_ids = {
        "left": model.geom(constants.LEFT_FEET_GEOMS[0]).id,
        "right": model.geom(constants.RIGHT_FEET_GEOMS[0]).id,
    }
    base = np.asarray(sim.get_floating_base_qpos(data.qpos)[:3], dtype=float)
    values = {
        side: {"normal_n": 0.0, "tangent_n": 0.0, "yaw_nm": 0.0,
               "cop_x_num": 0.0, "cop_y_num": 0.0, "contacts": 0.0}
        for side in foot_ids
    }
    for index in range(data.ncon):
        contact = data.contact[index]
        geoms = {int(contact.geom1), int(contact.geom2)}
        if floor_id not in geoms:
            continue
        side = next(
            (name for name, geom_id in foot_ids.items() if geom_id in geoms),
            None,
        )
        if side is None:
            continue
        local = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, local)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        force = frame.T @ local[:3]
        torque = frame.T @ local[3:]
        if force[2] < 0.0:
            force *= -1.0
            torque *= -1.0
        position = np.asarray(contact.pos, dtype=float)
        yaw_moment = float(np.cross(position - base, force)[2] + torque[2])
        normal = float(max(force[2], 0.0))
        tangent = float(np.linalg.norm(force[:2]))
        item = values[side]
        item["normal_n"] += normal
        item["tangent_n"] += tangent
        item["yaw_nm"] += yaw_moment
        item["cop_x_num"] += normal * float(position[0])
        item["cop_y_num"] += normal * float(position[1])
        item["contacts"] += 1.0
    for item in values.values():
        normal = item["normal_n"]
        item["cop_x_m"] = item.pop("cop_x_num") / normal if normal > 1e-8 else 0.0
        item["cop_y_m"] = item.pop("cop_y_num") / normal if normal > 1e-8 else 0.0
        item["friction_ratio"] = item["tangent_n"] / normal if normal > 1e-8 else 0.0
    return values


def run(model_path: Path, output: Path, yaw_command: float, friction: float | None) -> dict:
    simulation = DuckSimulation(REPO, OFFICIAL, output_root=output, warmup_s=0.0)
    sim = simulation.sim
    sim.policy = OnnxInfer(str(model_path), awd=True)
    geom_ids = [
        sim.model.geom("floor").id,
        sim.model.geom(constants.LEFT_FEET_GEOMS[0]).id,
        sim.model.geom(constants.RIGHT_FEET_GEOMS[0]).id,
    ]
    if friction is not None:
        sim.model.geom_friction[geom_ids, 0] = friction
        mujoco.mj_forward(sim.model, sim.data)

    dt = sim.sim_dt * sim.decimation
    joint_names = [
        mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        or f"actuator_{index}"
        for index in range(sim.model.nu)
    ]
    stand_loads: list[dict[str, dict[str, float]]] = []
    turn_rows = []

    def step(command: list[float], collect: bool) -> None:
        for _ in range(sim.decimation):
            mujoco.mj_step(sim.model, sim.data)
        sim.imitation_i = (
            sim.imitation_i + sim.phase_frequency_factor
        ) % sim.PRM.nb_steps_in_period
        phase = float(sim.imitation_i / sim.PRM.nb_steps_in_period)
        angle = phase * 2.0 * np.pi
        sim.imitation_phase = np.asarray(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        sim.commands = command
        observation = np.asarray(sim.get_obs(sim.data, command), dtype=np.float32)
        action = np.asarray(sim.policy.infer(observation), dtype=np.float32)
        requested = sim.default_actuator + action * sim.action_scale
        max_change = sim.max_motor_velocity * dt
        applied = np.clip(
            requested,
            sim.prev_motor_targets - max_change,
            sim.prev_motor_targets + max_change,
        )
        clipping = np.abs(requested - applied)
        contacts = foot_contact_sample(sim)
        if collect:
            turn_rows.append(
                {
                    "phase": phase,
                    "gyro_z": float(observation[2]),
                    "clip": clipping,
                    "contacts": contacts,
                }
            )
        else:
            stand_loads.append(contacts)
        sim.last_last_last_action = sim.last_last_action.copy()
        sim.last_last_action = sim.last_action.copy()
        sim.last_action = action.copy()
        sim.motor_targets = applied
        sim.prev_motor_targets = applied.copy()
        sim.data.ctrl[:] = applied

    zero = [0.0] * 7
    for step_index in range(round(3.0 / dt)):
        step(zero, collect=False)
    start_yaw = yaw_of(sim.get_floating_base_qpos(sim.data.qpos))
    command = [0.0, 0.0, yaw_command, 0.0, 0.0, 0.0, 0.0]
    for _ in range(round(5.0 / dt)):
        step(command, collect=True)
    end_yaw = yaw_of(sim.get_floating_base_qpos(sim.data.qpos))

    clips = np.stack([row["clip"] for row in turn_rows])
    phases = np.asarray([row["phase"] for row in turn_rows])
    gyros = np.asarray([row["gyro_z"] for row in turn_rows])
    burst = (phases >= 0.4) & (phases < 0.7)

    def contact_series(side: str, field: str, rows=turn_rows) -> np.ndarray:
        return np.asarray([row["contacts"][side][field] for row in rows])

    stand_tail = stand_loads[-round(2.0 / dt) :]
    left_stand = np.asarray([row["left"]["normal_n"] for row in stand_tail])
    right_stand = np.asarray([row["right"]["normal_n"] for row in stand_tail])
    total_stand = left_stand + right_stand
    left_share = np.divide(
        left_stand,
        total_stand,
        out=np.full_like(left_stand, 0.5),
        where=total_stand > 1e-8,
    )

    per_joint = {}
    for index, name in enumerate(joint_names):
        per_joint[name] = {
            "clipped_fraction": round(float(np.mean(clips[:, index] > 1e-7)), 6),
            "mean_clip_rad": round(float(np.mean(clips[:, index])), 8),
            "max_clip_rad": round(float(np.max(clips[:, index])), 8),
            "burst_clipped_fraction": round(
                float(np.mean(clips[burst, index] > 1e-7)), 6
            ),
        }

    left_normal = contact_series("left", "normal_n")
    right_normal = contact_series("right", "normal_n")
    both = (left_normal > 1e-6) & (right_normal > 1e-6)
    wrong = gyros * yaw_command < 0.0
    result = {
        "yaw_command_rad_s": yaw_command,
        "friction": friction,
        "heading_change_deg": round(math.degrees(wrap(end_yaw - start_yaw)), 6),
        "mean_yaw_rate_rad_s": round(float(np.mean(gyros[50:])), 6),
        "wrong_way_fraction_after_1s": round(float(np.mean(wrong[50:])), 6),
        "motor_slew": {
            "any_joint_clipped_frame_fraction": round(
                float(np.mean(np.any(clips > 1e-7, axis=1))), 6
            ),
            "phase_0p4_to_0p7_any_clipped_fraction": round(
                float(np.mean(np.any(clips[burst] > 1e-7, axis=1))), 6
            ),
            "per_joint": per_joint,
        },
        "settled_stance_load": {
            "left_normal_mean_n": round(float(np.mean(left_stand)), 6),
            "right_normal_mean_n": round(float(np.mean(right_stand)), 6),
            "left_load_share_mean": round(float(np.mean(left_share)), 6),
            "left_load_share_std": round(float(np.std(left_share)), 6),
        },
        "turn_contact": {
            "both_feet_contact_fraction": round(float(np.mean(both)), 6),
            "left_normal_mean_n": round(float(np.mean(left_normal)), 6),
            "right_normal_mean_n": round(float(np.mean(right_normal)), 6),
            "left_friction_ratio_p95": round(
                float(np.percentile(contact_series("left", "friction_ratio"), 95)), 6
            ),
            "right_friction_ratio_p95": round(
                float(np.percentile(contact_series("right", "friction_ratio"), 95)), 6
            ),
            "net_yaw_moment_mean_nm": round(
                float(
                    np.mean(
                        contact_series("left", "yaw_nm")
                        + contact_series("right", "yaw_nm")
                    )
                ),
                8,
            ),
            "burst_net_yaw_moment_mean_nm": round(
                float(
                    np.mean(
                        (
                            contact_series("left", "yaw_nm")
                            + contact_series("right", "yaw_nm")
                        )[burst]
                    )
                ),
                8,
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frictions", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.4]
    )
    parser.add_argument(
        "--native-friction", action="store_true",
        help="Keep the XML's floor and sole friction values unchanged",
    )
    parser.add_argument(
        "--commands", type=float, nargs="+", default=[0.15, -0.15],
        help="Yaw commands in rad/s; each run starts from the same settled pose",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    frictions = [None] if args.native_friction else args.frictions
    for friction in frictions:
        for command in args.commands:
            row = run(args.model, args.output_dir, command, friction)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    payload = {"model": str(args.model), "rows": rows}
    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"SAVED {result_path}")


if __name__ == "__main__":
    main()
