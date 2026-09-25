"""Replay signed reference yaw gaits through the native 50 Hz motor pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.open_duck_mini_v2.mujoco_infer import MjInfer


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
SCENE = REPO / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
REFERENCE = REPO / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
WALK = ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
REF_TO_ACT = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15])


def heading(qpos: np.ndarray) -> float:
    w, x, y, z = qpos[3:7]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--control-mode", choices=("clipped", "slew", "direct"),
        default="clipped",
    )
    args = parser.parse_args()

    prm = PolyReferenceMotion(str(REFERENCE))
    sim = MjInfer(str(SCENE), str(REFERENCE), str(WALK), standing=False)
    model, data = sim.model, sim.data
    imu_id = model.site("imu").id
    default = np.asarray(sim.default_actuator, dtype=float)
    control_dt = sim.sim_dt * sim.decimation
    rows = []

    for command_yaw in (0.15, -0.15, -0.25):
        # No learned policy is used.  Start every reference at the same settled
        # home pose to isolate whether the reference itself turns both ways.
        data.qpos[:] = model.keyframe("home").qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = default
        sim.prev_motor_targets = default.copy()
        mujoco.mj_forward(model, data)
        for _ in range(round(3.0 / control_dt)):
            for _ in range(sim.decimation):
                mujoco.mj_step(model, data)

        yaw_samples = [heading(data.qpos)]
        min_up_z = 1.0
        fall_time = None
        selected_grid_yaw = float(prm.dthetas[int(np.argmin(np.abs(np.asarray(prm.dthetas) - command_yaw)))])
        for step in range(round(args.seconds / control_dt)):
            ref = np.asarray(
                prm.get_reference_motion(0.0, 0.0, command_yaw, step),
                dtype=float,
            )
            requested_action = (ref[REF_TO_ACT] - default) / sim.action_scale
            if args.control_mode == "direct":
                applied = ref[REF_TO_ACT]
            else:
                action = (
                    np.clip(requested_action, -1.0, 1.0)
                    if args.control_mode == "clipped"
                    else requested_action
                )
                requested = default + action * sim.action_scale
                max_change = sim.max_motor_velocity * control_dt
                applied = np.clip(
                    requested,
                    sim.prev_motor_targets - max_change,
                    sim.prev_motor_targets + max_change,
                )
            sim.prev_motor_targets = applied.copy()
            data.ctrl[:] = applied
            for _ in range(sim.decimation):
                mujoco.mj_step(model, data)
            yaw_samples.append(heading(data.qpos))
            up_z = float(data.site_xmat[imu_id].reshape(3, 3)[2, 2])
            min_up_z = min(min_up_z, up_z)
            if fall_time is None and (up_z < 0.5 or float(data.qpos[2]) < 0.08):
                fall_time = round((step + 1) * control_dt, 3)

        unwrapped = np.unwrap(np.asarray(yaw_samples))
        row = {
            "command_yaw_rad_s": command_yaw,
            "control_mode": args.control_mode,
            "selected_reference_yaw_rad_s": selected_grid_yaw,
            "heading_change_deg": round(float(np.degrees(unwrapped[-1] - unwrapped[0])), 4),
            "min_up_z": round(min_up_z, 4),
            "fall_time_s": fall_time,
            "final_height_m": round(float(data.qpos[2]), 4),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"SAVED {args.output}")


if __name__ == "__main__":
    main()
