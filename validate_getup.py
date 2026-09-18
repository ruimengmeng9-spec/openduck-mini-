"""Validate a recovery policy in native MuJoCo from randomised fallen poses.

The upstream model cannot represent "lying on the floor" (only the foot soles
have collision geometry), so this uses the collision-enabled scene produced by
``make_getup_model.py``.  A recovery attempt is scored by whether the robot
reaches a standing attitude within the time budget: ``up_vector_z`` close to 1
and the base back near the home height.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer
from playground.common.onnx_infer import OnnxInfer

REPO = Path("/data/shijinsheng/open_duck/projects/Open_Duck_Playground")
DEFAULT_OUTPUT = Path("/data/shijinsheng/open_duck/outputs/getup_validation_20260918")
REFERENCE = REPO / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
GETUP_SCENE = REPO / "playground/open_duck_mini_v2/xmls/scene_flat_terrain_getup.xml"

# Standing test thresholds, matching the runtime safety envelope in duck_sim.py
# (which declares a fall at up_z < 0.50 or base height < 0.08 m).
STAND_UP_Z = 0.85
STAND_HEIGHT_M = 0.12

# (label, tilt about x in rad) -- 0 is face-down, pi is on the back.
POSES = [
    ("face_down", 1.57),
    ("on_back", -1.57),
    ("left_side", 1.57),
    ("right_side", -1.57),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget-s", type=float, default=8.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--side-axis",
        action="store_true",
        help="Tilt about y (roll) instead of x (pitch) for the side poses.",
    )
    args = parser.parse_args()
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX model not found: {args.onnx}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, tilt in POSES:
        for seed in args.seeds:
            sim = MjInfer(str(GETUP_SCENE), str(REFERENCE), str(args.onnx), standing=False)
            sim.policy = OnnxInfer(str(args.onnx), awd=True)

            model, data = sim.model, sim.data
            home = model.keyframe("home").qpos
            data.qpos[:] = home
            # Place the robot on the floor in the requested attitude.  Lying on
            # its side is a roll (about y); face-down/on-back is a pitch (about x).
            axis = np.array([0.0, 1.0, 0.0]) if (args.side_axis or "side" in label) else np.array([1.0, 0.0, 0.0])
            angle = tilt if "back" not in label and "right" not in label else -tilt
            half = angle / 2.0
            s = math.sin(half)
            data.qpos[3:7] = np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])
            data.qpos[2] = 0.075
            data.qvel[:] = 0.0
            data.ctrl[:] = sim.default_actuator
            sim.motor_targets = np.array(sim.default_actuator, dtype=np.float64).copy()
            sim.prev_motor_targets = sim.motor_targets.copy()
            sim.commands = [0.0] * 7
            mujoco.mj_forward(model, data)

            imu = model.site("imu").id
            steps = int(round(args.budget_s / (sim.sim_dt * sim.decimation)))
            peak_up_z = -1.0
            peak_height = -1.0
            stood_at = None
            for step in range(steps):
                for _ in range(sim.decimation):
                    mujoco.mj_step(model, data)
                sim.imitation_i = (sim.imitation_i + sim.phase_frequency_factor) % sim.PRM.nb_steps_in_period
                ang = sim.imitation_i / sim.PRM.nb_steps_in_period * 2.0 * np.pi
                sim.imitation_phase = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
                obs = sim.get_obs(data, sim.commands)
                action = np.asarray(sim.policy.infer(obs), dtype=np.float32)
                sim.last_last_last_action = sim.last_action.copy()
                sim.last_last_action = sim.last_action.copy()
                sim.last_action = action.copy()
                target = sim.default_actuator + action * sim.action_scale
                max_change = sim.max_motor_velocity * (sim.sim_dt * sim.decimation)
                sim.motor_targets = np.clip(
                    target, sim.prev_motor_targets - max_change, sim.prev_motor_targets + max_change
                )
                sim.prev_motor_targets = sim.motor_targets.copy()
                data.ctrl[:] = sim.motor_targets

                up_z = float(data.site_xmat[imu].reshape(3, 3)[2, 2])
                height = float(data.qpos[2])
                peak_up_z = max(peak_up_z, up_z)
                peak_height = max(peak_height, height)
                if stood_at is None and up_z > STAND_UP_Z and height > STAND_HEIGHT_M:
                    stood_at = (step + 1) * sim.sim_dt * sim.decimation

            final_up_z = float(data.site_xmat[imu].reshape(3, 3)[2, 2])
            row = {
                "pose": label,
                "seed": seed,
                "model": str(args.onnx),
                "stood_up": stood_at is not None,
                "time_to_stand_s": round(stood_at, 3) if stood_at else None,
                "final_up_z": round(final_up_z, 4),
                "final_base_height_m": round(float(data.qpos[2]), 4),
                "peak_up_z": round(peak_up_z, 4),
                "peak_base_height_m": round(peak_height, 4),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    ok = sum(1 for r in rows if r["stood_up"])
    summary = {"attempts": len(rows), "stood_up": ok, "success_rate": round(ok / max(len(rows), 1), 3)}
    (args.output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    print("SAVED", args.output_dir / "results.json", flush=True)


if __name__ == "__main__":
    main()
