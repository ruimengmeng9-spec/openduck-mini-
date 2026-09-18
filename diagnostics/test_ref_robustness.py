"""Is the -0.074 reference gait robustly stable, or only marginally/chaotically so?

Direct replay and pipeline replay are numerically almost identical yet one
survives 5 s and the other falls at 2 s.  That points to a marginally stable
(chaotic) gait rather than a pipeline artefact.  Replay the same reference with
escalating initial perturbations; if small perturbations already topple it, the
reference cannot be expected to confer robust balance.
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
nb = prm.nb_steps_in_period


def replay(vx, qvel_scale, seed):
    rng = np.random.default_rng(seed)
    d.qpos[:] = m.keyframe("home").qpos
    d.qvel[:] = rng.normal(scale=qvel_scale, size=m.nv)
    d.ctrl[:] = sim.default_actuator
    mujoco.mj_forward(m, d)
    ups = []
    for step in range(250):
        joints = np.asarray(prm.get_reference_motion(vx, 0.0, 0.0, step % nb), dtype=float)[REF_TO_ACT]
        for _ in range(sim.decimation):
            d.ctrl[:] = joints
            mujoco.mj_step(m, d)
        ups.append(float(d.site_xmat[imu].reshape(3, 3)[2, 2]))
    return next((i * 0.02 for i, u in enumerate(ups) if u < 0.5), None)


for vx in (-0.074, 0.074):
    print("reference vx = %+0.3f" % vx)
    for scale in (0.0, 1e-6, 1e-4, 1e-2, 5e-2):
        falls = [replay(vx, scale, s) for s in range(5)]
        n_fell = sum(f is not None for f in falls)
        sample = [("%.2f" % f) if f else "ok" for f in falls]
        print("   qvel noise %-8s -> fell %d/5   %s" % (scale, n_fell, sample))
    print()
