"""Check whether the getup start pose drives the trunk box through the floor."""
import math
import numpy as np
import mujoco

scene = "/data/shijinsheng/open_duck/projects/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain_getup.xml"
m = mujoco.MjModel.from_xml_path(scene)
gid = m.geom("trunk_collision").id
bid = m.geom_bodyid[gid]
print("trunk box: pos=%s size=%s bodyid=%d" % (
    np.round(m.geom_pos[gid], 4), np.round(m.geom_size[gid], 4), bid))
hx, hy, hz = m.geom_size[gid]
c = m.geom_pos[gid]

print()
print("%-12s %-8s %-10s %-12s" % ("tilt", "base_z", "extent_z", "box_bottom"))
for label, axis, ang in [
    ("face_down", np.array([1.0, 0, 0]), 1.57),
    ("on_back", np.array([1.0, 0, 0]), -1.57),
    ("side", np.array([0, 1.0, 0]), 1.57),
]:
    half = ang / 2.0
    s = math.sin(half)
    q = np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])
    # rotation matrix from quaternion
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)],
    ])
    extent = hx*abs(R[2, 0]) + hy*abs(R[2, 1]) + hz*abs(R[2, 2])
    center_z_off = (R @ c)[2]
    for base_z in (0.075, 0.090):
        bottom = base_z + center_z_off - extent
        print("%-12s %-8.3f %-10.4f %-12.4f  %s" % (
            label, base_z, extent, bottom,
            "<-- PENETRATES" if bottom < 0 else "ok"))
