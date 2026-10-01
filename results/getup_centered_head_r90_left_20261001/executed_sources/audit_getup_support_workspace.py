"""Kinematic foot-support screening; never interpreted as a dynamic get-up.

Root pose is prescribed only in this geometry diagnostic. No state from this
file may be used to count a recovery rollout as physically achieved.
"""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from diagnostics.getup_independent_native import digest, feature_targets


class Workspace:
    def __init__(self, scene):
        self.model = m = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(m)
        self.joints = m.actuator_trnid[:, 0]
        self.qadr = m.jnt_qposadr[self.joints]
        self.lower, self.upper = m.jnt_range[self.joints].T.copy()
        self.home = m.keyframe('home').qpos[self.qadr].copy()
        self.feet = [m.body(name).id for name in ('foot_assembly', 'foot_assembly_2')]
        self.torso = {m.body(name).id for name in ('trunk_assembly', 'head_assembly')}
        self.base = m.body('base').id
        self.vertices = {}
        for i in range(m.ngeom):
            if m.geom_bodyid[i] == 0 or m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = m.geom_dataid[i]
            adr, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            self.vertices[i] = m.mesh_vert[adr:adr+n].copy()

    def evaluate(self, pitch, features):
        m, d = self.model, self.data
        d.qpos[:] = m.keyframe('home').qpos
        d.qpos[:3] = 0.
        d.qpos[3:7] = [math.cos(pitch/2), 0., math.sin(pitch/2), 0.]
        d.qpos[self.qadr] = feature_targets(np.array([features]), self.home, self.lower, self.upper)[0]
        mujoco.mj_forward(m, d)
        world = {g: v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for g, v in self.vertices.items()}
        foot_vertices = [np.concatenate([v for g, v in world.items() if m.geom_bodyid[g] == b]) for b in self.feet]
        min_foot_z = min(float(v[:,2].min()) for v in foot_vertices)
        root_height = -min_foot_z
        sole_points = np.concatenate([v[v[:,2] <= min_foot_z+.001] for v in foot_vertices])
        torso_clearance = min(float(v[:,2].min()-min_foot_z) for g, v in world.items() if m.geom_bodyid[g] in self.torso)
        other_clearance = min(float(v[:,2].min()-min_foot_z) for g, v in world.items() if m.geom_bodyid[g] not in self.feet)
        com = d.subtree_com[self.base].copy()
        # Rectangle is an optimistic outer bound of the actual support polygon.
        lo, hi = sole_points[:,:2].min(axis=0), sole_points[:,:2].max(axis=0)
        outside = float(np.maximum(np.maximum(lo-com[:2], com[:2]-hi), 0.).max())
        foot_gap = float(abs(foot_vertices[0][:,2].min()-foot_vertices[1][:,2].min()))
        return dict(torso_clearance_m=torso_clearance, nonfoot_clearance_m=other_clearance,
                    com_outside_support_rectangle_m=outside, left_right_ground_gap_m=foot_gap,
                    implied_root_height_m=root_height, com_height_m=float(com[2]+root_height),
                    sole_support_bounds_xy_m=[lo.tolist(),hi.tolist()], com_xy_m=com[:2].tolist(),
                    features_rad=np.asarray(features).tolist(),
                    optimistic_feet_only_geometry_feasible=bool(other_clearance >= -.001 and outside <= .001 and foot_gap <= .001))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=50)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.root/'projects/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml'
    w = Workspace(scene)
    bounds = [(-1.2217,.5236),(-1.5708,1.5708),(-1.5708,1.5708),(-.3491,1.1345),(-.7854,.7854),(-.4363,.4363),(-.5236,.5236)]
    rows = []
    for degrees in (0,30,45,60,75,90,-30,-45,-60,-75,-90):
        pitch = math.radians(degrees)
        def objective(x):
            r = w.evaluate(pitch, x)
            return (max(-r['nonfoot_clearance_m'],0.)*5.+r['com_outside_support_rectangle_m']*3.
                    +r['left_right_ground_gap_m']*2.)
        result = differential_evolution(objective, bounds, maxiter=args.iterations, popsize=8,
                                        seed=138+degrees, polish=False, workers=1, tol=.0001)
        row = dict(pitch_deg=degrees, objective=float(result.fun), evaluations=int(result.nfev), **w.evaluate(pitch,result.x))
        rows.append(row)
        print(json.dumps(row), flush=True)
        summary = dict(prescribed_root_pose=True, kinematic_only=True, dynamic_recovery_success=False,
                       hardware_readiness=False, self_collision_checked=False,
                       note='optimistic support rectangle; symmetric search is not a proof of global infeasibility',
                       rows=rows, hashes={str(scene):digest(scene), str(Path(__file__)):digest(__file__)})
        (args.output/'results.json').write_text(json.dumps(summary,indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
