"""Quantify which signed turn-reference joints exceed executable policy actions."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prm = PolyReferenceMotion(str(REFERENCE))
    sim = MjInfer(str(SCENE), str(REFERENCE), str(WALK), standing=False)
    default = np.asarray(sim.default_actuator, dtype=float)
    names = [
        mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(sim.model.nu)
    ]
    rows = []

    for command_yaw in (0.15, -0.15, -0.25):
        actions = np.asarray(
            [
                (
                    np.asarray(prm.get_reference_motion(0.0, 0.0, command_yaw, i))[REF_TO_ACT]
                    - default
                )
                / sim.action_scale
                for i in range(prm.nb_steps_in_period)
            ],
            dtype=float,
        )
        excess = np.maximum(np.abs(actions) - 1.0, 0.0)
        joints = [
            {
                "joint": name,
                "max_abs_action": round(float(np.max(np.abs(actions[:, j]))), 4),
                "clipped_phase_fraction": round(float(np.mean(excess[:, j] > 0)), 4),
                "mean_excess": round(float(np.mean(excess[:, j])), 4),
            }
            for j, name in enumerate(names)
        ]
        row = {
            "command_yaw_rad_s": command_yaw,
            "selected_reference_yaw_rad_s": float(
                prm.dthetas[int(np.argmin(np.abs(np.asarray(prm.dthetas) - command_yaw)))]
            ),
            "any_joint_clipped_phase_fraction": round(
                float(np.mean(np.any(excess > 0.0, axis=1))), 4
            ),
            "mean_excess_all_actions": round(float(np.mean(excess)), 4),
            "joint_bounds": joints,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"SAVED {args.output}")


if __name__ == "__main__":
    main()
