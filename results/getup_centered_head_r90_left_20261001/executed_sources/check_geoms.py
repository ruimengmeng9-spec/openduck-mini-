"""Authoritatively report which geoms can collide with the floor."""
import numpy as np
import mujoco

from playground.open_duck_mini_v2 import constants

path = constants.task_to_xml("flat_terrain").as_posix()
model = mujoco.MjModel.from_xml_path(path)
print("total geoms:", model.ngeom)
colliding = []
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    ct, ca = int(model.geom_contype[i]), int(model.geom_conaffinity[i])
    if ct != 0 or ca != 0:
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i]))
        colliding.append((i, name, body, ct, ca))
print("geoms with contype/conaffinity != 0:", len(colliding))
for row in colliding:
    print("   idx=%d name=%s body=%s contype=%d conaffinity=%d" % row)

floor = model.geom("floor").id
print("floor contype=%d conaffinity=%d" % (model.geom_contype[floor], model.geom_conaffinity[floor]))
