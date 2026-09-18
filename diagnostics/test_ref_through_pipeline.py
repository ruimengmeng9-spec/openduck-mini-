"""Replay the -0.074 reference through the REAL action pipeline.

The plain open-loop replay sets ctrl = reference joint angles directly and
survives 5 s, whereas every trained policy falls after ~2 s.  The difference is
the interface: a policy output is scaled by action_scale and rate-limited by
max_motor_velocity before becoming a position target.  If the reference still
survives when pushed through that pipeline, the interface can express the gait
and the failure is purely one of learning/tracking.
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

for label, use_pipeline in [("direct ctrl (reference)", False),
                            ("through action pipeline", True)]:
    d.qpos[:] = m.keyframe("home").qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = default
    sim.motor_targets = default.copy()
    sim.prev_motor_targets = default.copy()
    sim.commands = [0.0] * 7
    mujoco.mj_forward(m, d)
    ups = []
    nb = prm.nb_steps_in_period
    for step in range(250):  # 5 s
        ref = np.asarray(prm.get_reference_motion(-0.074, 0.0, 0.0, step % nb), dtype=float)
        joints = ref[REF_TO_ACT]
        for _ in range(sim.decimation):
            if use_pipeline:
                # This is what a perfect tracker must emit as its normalised action.
                action = (joints - default) / sim.action_scale
                target = default + action * sim.action_scale
                max_change = sim.max_motor_velocity * (sim.sim_dt * sim.decimation)
                sim.motor_targets = np.clip(target,
                                            sim.prev_motor_targets - max_change,
                                            sim.prev_motor_targets + max_change)
                sim.prev_motor_targets = sim.motor_targets.copy()
                d.ctrl[:] = sim.motor_targets
            else:
                d.ctrl[:] = joints
            mujoco.mj_step(m, d)
        ups.append(float(d.site_xmat[imu].reshape(3, 3)[2, 2]))
    ups = np.array(ups)
    fell = next((i * 0.02 for i, u in enumerate(ups) if u < 0.5), None)
    print("%-28s final_up_z=%.3f min=%.3f height=%.4f fell_at=%s"
          % (label, ups[-1], ups.min(), float(d.qpos[2]),
             ("%.2f s" % fell) if fell else "never"))
