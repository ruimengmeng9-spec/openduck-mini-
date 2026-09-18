"""Verify the getup model has torso collision and that the original is intact."""
import numpy as np
import mujoco

from playground.open_duck_mini_v2 import constants

base = mujoco.MjModel.from_xml_path(constants.task_to_xml("flat_terrain").as_posix())
getup = mujoco.MjModel.from_xml_path(constants.task_to_xml("flat_terrain_getup").as_posix())


def colliding(model):
    out = []
    for i in range(model.ngeom):
        ct, ca = int(model.geom_contype[i]), int(model.geom_conaffinity[i])
        if ct or ca:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i]))
            out.append((name, body, ct, ca, tuple(np.round(model.geom_size[i], 4))))
    return out


b = colliding(base)
g = colliding(getup)
print("original colliding geoms:", len(b))
for r in b:
    print("   ", r)
print("getup colliding geoms   :", len(g))
for r in g:
    print("   ", r)

assert len(b) == 3, "original model changed!"
assert len(g) == 5, "expected 2 extra collision geoms"
names = {r[0] for r in g}
assert "trunk_collision" in names and "head_collision" in names
print()
print("nq/nv/nu identical:", base.nq == getup.nq, base.nv == getup.nv, base.nu == getup.nu)
print("body count identical:", base.nbody == getup.nbody)
print("OK: original untouched, getup adds trunk+head collision")

# Reset the getup model to a fallen pose and check it settles on the floor
# instead of sinking through it.
m = mujoco.MjModel.from_xml_path(constants.task_to_xml("flat_terrain_getup").as_posix())
d = mujoco.MjData(m)
home = m.keyframe("home").qpos.copy()
d.qpos[:] = home
d.qpos[2] = 0.09
d.qpos[3:7] = np.array([0.7071, 0.7071, 0.0, 0.0])  # ~90 deg pitch
d.qvel[:] = 0.0
mujoco.mj_forward(m, d)
imu = m.site("imu").id
for _ in range(4000):
    d.ctrl[:] = m.keyframe("home").ctrl
    mujoco.mj_step(m, d)
print()
print("after settling on floor:")
print("  base z = %.4f" % d.qpos[2])
print("  imu upvector z = %.4f" % d.site_xmat[imu].reshape(3, 3)[2, 2])
assert d.qpos[2] > -0.05, "robot sank through the floor -> collision not working"
print("OK: robot rests on the floor instead of sinking")
