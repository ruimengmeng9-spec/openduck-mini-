"""Report per-body local bounding boxes of visual meshes, to size collision geoms."""
import numpy as np
import mujoco

from playground.open_duck_mini_v2 import constants

model = mujoco.MjModel.from_xml_path(constants.task_to_xml("flat_terrain").as_posix())
print("nbody:", model.nbody, "ngeom:", model.ngeom, "nmesh:", model.nmesh)

# Cache each mesh's local vertex AABB.
mesh_aabb = {}
for m in range(model.nmesh):
    adr, num = model.mesh_vertadr[m], model.mesh_vertnum[m]
    verts = model.mesh_vert[adr : adr + num]
    mesh_aabb[m] = (verts.min(axis=0), verts.max(axis=0))

print()
print("%-34s %-30s %s" % ("body", "local AABB min", "local AABB max"))
for b in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
    if name is None:
        continue
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    n = 0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != b:
            continue
        gt = int(model.geom_type[g])
        pos = model.geom_pos[g]
        size = model.geom_size[g]
        # Only meshes matter for visual extent, but include primitives too.
        if gt == int(mujoco.mjtGeom.mjGEOM_MESH) or gt == int(mujoco.mjtGeom.mjGEOM_SDF):
            mg = int(model.geom_dataid[g])
            if mg >= 0:
                a, z = mesh_aabb[mg]
                p_lo, p_hi = pos + a, pos + z
            else:
                continue
        elif gt == int(mujoco.mjtGeom.mjGEOM_BOX):
            p_lo, p_hi = pos - size, pos + size
        elif gt in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            r, h = float(size[0]), float(size[1])
            p_lo = pos - np.array([r, r, h + r])
            p_hi = pos + np.array([r, r, h + r])
        elif gt == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            r = float(size[0])
            p_lo, p_hi = pos - r, pos + r
        else:
            continue
        lo = np.minimum(lo, p_lo)
        hi = np.maximum(hi, p_hi)
        n += 1
    if n:
        print("%-34s [%s] [%s]" % (
            name,
            " ".join("%+.4f" % v for v in lo),
            " ".join("%+.4f" % v for v in hi),
        ))
