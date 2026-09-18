"""Per-geom local AABB for the trunk and head bodies, to size collision boxes."""
import numpy as np
import mujoco

from playground.open_duck_mini_v2 import constants

model = mujoco.MjModel.from_xml_path(constants.task_to_xml("flat_terrain").as_posix())

mesh_aabb = {}
for m in range(model.nmesh):
    adr, num = model.mesh_vertadr[m], model.mesh_vertnum[m]
    v = model.mesh_vert[adr : adr + num]
    mesh_aabb[m] = (v.min(axis=0), v.max(axis=0))

TARGETS = {"trunk_assembly", "head_assembly", "neck_yaw_assembly", "foot_assembly"}
for b in range(model.nbody):
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
    if bname not in TARGETS:
        continue
    print(f"--- body {bname} (pos={np.round(model.body_pos[b],4)}) ---")
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != b:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        gt = int(model.geom_type[g])
        if gt != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mg = int(model.geom_dataid[g])
        if mg < 0:
            continue
        a, z = mesh_aabb[mg]
        pos = model.geom_pos[g]
        print("   %-24s mesh=%-26s pos=%-28s ext=[%s] .. [%s]" % (
            str(gname), str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mg)),
            np.round(pos, 4),
            " ".join("%+.4f" % v for v in (pos + a)),
            " ".join("%+.4f" % v for v in (pos + z)),
        ))
