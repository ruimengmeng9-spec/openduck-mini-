"""Is the -0.074 reference gait itself physically stable? Open-loop replay test.

If tracking the reference joint trajectory open-loop keeps the robot upright, the
reference is a viable gait and the failure is a tracking/learning problem.  If it
falls, imitation can never supply stability and the reference entries are the
limit -- which would explain why v12..v17 all fall after ~2 s.
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

prm = PolyReferenceMotion(REF)
sim = MjInfer(SCENE, REF, WALK, standing=False)
sim.policy = OnnxInfer(WALK, awd=True)
m, d = sim.model, sim.data
imu = m.site("imu").id

# Reference joint order (16 dims): 5 left leg, 4 head, 2 antenna, 5 right leg.
# The robot has 14 actuators: left_hip_yaw..left_ankle, neck_pitch, head_pitch,
# head_yaw, head_roll, right_hip_yaw..right_ankle -- i.e. reference dims 0-4 and
# 11-15, skipping the two antennas at 9-10.
REF_TO_ACT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]

for label, vx in [("forward +0.074", 0.074), ("reverse -0.074", -0.074),
                  ("standing 0.000", 0.0)]:
    d.qpos[:] = m.keyframe("home").qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = sim.default_actuator
    mujoco.mj_forward(m, d)
    ups = []
    nb = prm.nb_steps_in_period
    for step in range(250):  # 5 s
        ref = np.asarray(prm.get_reference_motion(vx, 0.0, 0.0, step % nb), dtype=float)
        joints = ref[REF_TO_ACT]
        for _ in range(sim.decimation):
            d.ctrl[:] = joints
            mujoco.mj_step(m, d)
        ups.append(float(d.site_xmat[imu].reshape(3, 3)[2, 2]))
    ups = np.array(ups)
    fell_at = next((i * 0.02 for i, u in enumerate(ups) if u < 0.5), None)
    print("%-18s final_up_z=%.3f  min_up_z=%.3f  final_height=%.4f  fell_at=%s"
          % (label, ups[-1], ups.min(), float(d.qpos[2]),
             ("%.2f s" % fell_at) if fell_at else "never"))
