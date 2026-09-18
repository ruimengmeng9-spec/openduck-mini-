"""Is the reverse reference a walking GAIT or effectively a standing pose?

An open-loop replay of -0.074 survived 5 s while +0.074 fell at 1.7 s, which is
backwards from what a walking gait would do (a stepping trajectory needs balance
feedback; a near-static pose does not).  Measure how much each reference actually
moves over its cycle, and how far it sits from the home pose.
"""

import numpy as np
import mujoco

from playground.common.poly_reference_motion import PolyReferenceMotion

REPO = "/data/shijinsheng/open_duck/projects/Open_Duck_Playground"
REF = f"{REPO}/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SCENE = f"{REPO}/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
REF_TO_ACT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]

prm = PolyReferenceMotion(REF)
m = mujoco.MjModel.from_xml_path(SCENE)
home = np.array(m.keyframe("home").qpos[7:], dtype=float)  # 14 actuator joints
nb = prm.nb_steps_in_period

print("home joint angles (14):", np.round(home, 3))
print()
print("%-10s %-10s %-12s %-14s %s" % ("commanded", "grid dx", "cyc span", "vs home", "verdict"))
for v in (0.222, 0.148, 0.074, 0.0, -0.074, -0.148):
    ix, iy, it = prm.vel_to_index(v, 0.0, 0.0)
    traj = np.stack([
        np.asarray(prm.get_reference_motion(v, 0.0, 0.0, k), dtype=float)[REF_TO_ACT]
        for k in range(nb)
    ])
    span = float(np.abs(traj.max(axis=0) - traj.min(axis=0)).mean())      # per-joint motion
    off = float(np.abs(traj.mean(axis=0) - home).mean())                   # distance from home
    verdict = "standing-like" if span < 0.02 else "has stepping motion"
    print("%-10.3f %-10.3f %-12.4f %-14.4f %s" % (v, prm.dxs[int(ix)], span, off, verdict))

print()
print("per-joint cycle span at -0.074 and +0.074 (radians):")
for v in (0.074, -0.074):
    ix, iy, it = prm.vel_to_index(v, 0.0, 0.0)
    traj = np.stack([
        np.asarray(prm.get_reference_motion(v, 0.0, 0.0, k), dtype=float)[REF_TO_ACT]
        for k in range(nb)
    ])
    print("  %+0.3f -> %s" % (v, np.round(traj.max(axis=0) - traj.min(axis=0), 3)))
