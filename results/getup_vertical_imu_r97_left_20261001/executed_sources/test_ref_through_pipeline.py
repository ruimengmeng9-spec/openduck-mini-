"""Replay the -0.074 reference through the real control-rate pipeline.

The motor speed limit is applied once per 50 Hz control tick, matching the JAX
environment and native policy runner. Applying it inside the 500 Hz physics
loop accidentally makes the actuator target move ten times faster.
"""

import numpy as np
import mujoco

from playground.open_duck_mini_v2.mujoco_infer import MjInfer
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion import PolyReferenceMotion

REPO = "/data/shijinsheng/open_duck/projects/Open_Duck_Playground"
WALK = "/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
SCENE = f"{REPO}/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
REF = f"{REPO}/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
REF_TO_ACT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]

prm = PolyReferenceMotion(REF)
sim = MjInfer(SCENE, REF, WALK, standing=False)
sim.policy = OnnxInfer(WALK, awd=True)
m, d = sim.model, sim.data
imu = m.site("imu").id
default = np.array(sim.default_actuator, dtype=float)


def run(
    seed: int, mode: str, seconds: float = 10.0
) -> tuple[bool, float, float, float, float]:
    rng = np.random.default_rng(seed)
    d.qpos[:] = m.keyframe("home").qpos
    d.qvel[:] = rng.uniform(-0.05, 0.05, size=m.nv)
    d.ctrl[:] = default
    sim.motor_targets = default.copy()
    sim.prev_motor_targets = default.copy()
    mujoco.mj_forward(m, d)
    start_xy = np.asarray(d.qpos[:2], dtype=float).copy()
    ups = []
    nb = prm.nb_steps_in_period
    for step in range(round(seconds / 0.02)):
        ref = np.asarray(
            prm.get_reference_motion(-0.074, 0.0, 0.0, step % nb), dtype=float
        )
        joints = ref[REF_TO_ACT]
        if mode == "direct":
            target = joints
        else:
            action = (joints - default) / sim.action_scale
            if mode == "clipped":
                action = np.clip(action, -1.0, 1.0)
            target = default + action * sim.action_scale
            max_change = sim.max_motor_velocity * 0.02
            target = np.clip(
                target,
                sim.prev_motor_targets - max_change,
                sim.prev_motor_targets + max_change,
            )
            sim.prev_motor_targets = target.copy()
        d.ctrl[:] = target
        for _ in range(sim.decimation):
            mujoco.mj_step(m, d)
        up_z = float(d.site_xmat[imu].reshape(3, 3)[2, 2])
        ups.append(up_z)
        if up_z < 0.5 or float(d.qpos[2]) < 0.08:
            delta = np.asarray(d.qpos[:2], dtype=float) - start_xy
            return True, (step + 1) * 0.02, min(ups), float(delta[0]), float(delta[1])
    delta = np.asarray(d.qpos[:2], dtype=float) - start_xy
    return False, seconds, min(ups), float(delta[0]), float(delta[1])


for mode in ("direct", "unclipped", "clipped"):
    results = [run(seed, mode) for seed in range(5)]
    print(
        f"{mode:10s} falls={sum(item[0] for item in results)}/5 "
        f"durations={[round(item[1], 2) for item in results]} "
        f"min_up_z={min(item[2] for item in results):.3f} "
        f"mean_dx={np.mean([item[3] for item in results]):.3f} "
        f"mean_dy={np.mean([item[4] for item in results]):.3f}"
    )
