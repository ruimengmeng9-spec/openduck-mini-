"""Compare direct reverse reference replay with the actor-reachable target.

The actor emits normalized actions in [-1, 1] and the runtime maps them to
home + 0.25 * action. A physically stable direct joint reference is not a
usable teacher if that mapping cannot express its targets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer
from diagnostics.reference_residual_policy import ContinuousDxReference


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
REF_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dx", type=float, default=-0.0925)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    args = parser.parse_args()
    os.environ["REFERENCE_DX_INTERPOLATION"] = "1"
    os.environ["REFERENCE_DX"] = str(args.reference_dx)
    scene = REPO / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
    reference = REPO / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
    walk = ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
    sim = MjInfer(str(scene), str(reference), str(walk), standing=False)
    sim.PRM = ContinuousDxReference(sim.PRM)
    model, data = sim.model, sim.data
    home = np.asarray(model.keyframe("home").ctrl, dtype=float)
    dt = sim.sim_dt * sim.decimation
    period = sim.PRM.nb_steps_in_period
    reference_actions = []
    for phase in range(round(period)):
        ref = np.asarray(
            sim.PRM.get_reference_motion(args.reference_dx, 0.0, 0.0, phase),
            dtype=float,
        )[REF_IDS]
        reference_actions.append((ref - home) / sim.action_scale)
    actions = np.asarray(reference_actions)
    joint_ids = model.actuator_trnid[:, 0]
    targets = home + actions * sim.action_scale
    joint_lower = model.jnt_range[joint_ids, 0]
    joint_upper = model.jnt_range[joint_ids, 1]
    beyond_joint_limit = (targets < joint_lower) | (targets > joint_upper)
    unreachable = np.abs(actions) > 1.0
    reachability = {
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "fraction_of_action_entries_outside_limit": float(unreachable.mean()),
        "fraction_of_phases_with_any_unreachable_joint": float(
            unreachable.any(axis=1).mean()
        ),
        "fraction_of_phases_with_any_unreachable_leg_joint": float(
            unreachable[:, list(range(5)) + list(range(9, 14))].any(axis=1).mean()
        ),
        "per_joint_fraction_outside_limit": unreachable.mean(axis=0).tolist(),
        "max_abs_action_per_joint": np.abs(actions).max(axis=0).tolist(),
        "joint_names": [model.actuator(i).name for i in range(model.nu)],
        "per_joint_fraction_reference_beyond_physical_limit": beyond_joint_limit.mean(axis=0).tolist(),
        "reference_joint_min_rad": targets.min(axis=0).tolist(),
        "reference_joint_max_rad": targets.max(axis=0).tolist(),
        "joint_lower_rad": joint_lower.tolist(),
        "joint_upper_rad": joint_upper.tolist(),
    }
    rows = []
    for mode in ("direct", "actor_clipped"):
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            data.qpos[:] = model.keyframe("home").qpos
            data.qvel[:] = rng.uniform(-0.02, 0.02, size=model.nv)
            data.ctrl[:] = home
            mujoco.mj_forward(model, data)
            for _ in range(round(args.warmup_s / sim.sim_dt)):
                mujoco.mj_step(model, data)
            start = data.qpos[:3].copy()
            previous = home.copy()
            min_up = 1.0
            fallen = False
            completed = 0
            for step in range(round(args.duration_s / dt)):
                ref = np.asarray(
                    sim.PRM.get_reference_motion(
                        args.reference_dx, 0.0, 0.0, step % period
                    ), dtype=float,
                )[REF_IDS]
                target = (
                    ref if mode == "direct"
                    else home + sim.action_scale * np.clip(
                        (ref - home) / sim.action_scale, -1.0, 1.0
                    )
                )
                limit = sim.max_motor_velocity * dt
                target = np.clip(target, previous - limit, previous + limit)
                previous = target.copy()
                data.ctrl[:] = target
                for _ in range(sim.decimation):
                    mujoco.mj_step(model, data)
                completed = step + 1
                up = float(data.site_xmat[model.site("imu").id].reshape(3, 3)[2, 2])
                min_up = min(min_up, up)
                if up < 0.5 or data.qpos[2] < 0.08:
                    fallen = True
                    break
            rows.append({
                "mode": mode,
                "seed": seed,
                "duration_s": completed * dt,
                "fallen": fallen,
                "forward_displacement_m": float(data.qpos[0] - start[0]),
                "lateral_displacement_m": float(data.qpos[1] - start[1]),
                "mean_forward_speed_mps": float(
                    (data.qpos[0] - start[0]) / max(completed * dt, 1e-6)
                ),
                "min_up_z": min_up,
            })
    result = {
        "reference_dx": args.reference_dx,
        "action_scale": sim.action_scale,
        "motor_velocity_limit_rad_s": sim.max_motor_velocity,
        "control_dt_s": dt,
        "warmup_s": args.warmup_s,
        "reachability": reachability,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
