"""Inspect reachable crouches, not prescribed or teleported root states."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from diagnostics.getup_aligned_extension import AlignedSim
from diagnostics.getup_beam_reference import primitive
from diagnostics.getup_independent_native import digest


class SupportSim(AlignedSim):
    def __init__(self, scene, stand):
        super().__init__(scene, stand)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.keyframe('home').id)
        mujoco.mj_forward(self.model, self.data)
        # Reference foot-body orientations, NOT a claim that a sole sensor exists.
        self.home_up_axes = [self.data.xmat[b].reshape(3, 3).T @ np.array([0., 0., 1.])
                             for b in self.feet]

    def measure(self):
        m = super().measure()
        m['foot_up_alignment_to_home'] = [float((self.data.xmat[b].reshape(3, 3) @ axis)[2])
                                           for b, axis in zip(self.feet, self.home_up_axes)]
        m['foot_centers_xyz_m'] = self.data.xpos[self.feet].tolist()
        m['joint_positions_rad'] = self.data.qpos[self.qadr].tolist()
        m['joint_targets_rad'] = self.prev.tolist()
        m['center_of_mass_xyz_m'] = self.data.subtree_com[0].tolist()
        points = []
        forces = []
        for i, contact in enumerate(self.data.contact):
            if self.floor not in contact.geom or contact.dist > .001:
                continue
            other = int(contact.geom[1] if contact.geom[0] == self.floor else contact.geom[0])
            if int(self.model.geom_bodyid[other]) not in self.feet:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            # Separate geometric proximity from loaded support.
            if force[0] > .001:
                points.append(contact.pos[:2].copy())
                forces.append(float(force[0]))
        m['loaded_foot_contact_points_xy_m'] = np.unique(np.round(points, 8), axis=0).tolist() if points else []
        m['foot_contact_normal_force_sum_n'] = sum(forces)
        m['com_support_margin_m'] = None
        if len(m['loaded_foot_contact_points_xy_m']) >= 3:
            try:
                hull = ConvexHull(m['loaded_foot_contact_points_xy_m'])
                eq = hull.equations
                m['com_support_margin_m'] = float(np.min(-(eq[:, :2] @ self.data.subtree_com[0, :2] + eq[:, 2])))
            except Exception as exc:
                # Degenerate/collinear contacts do not define an area of support.
                m['support_hull_error'] = type(exc).__name__
        return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    training = args.root / 'training'
    scene = training / 'getup_decomposed_r4/model/scene.xml'
    sim = SupportSim(scene, args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    rows = []
    sim.prepare('standing')
    for _ in range(150):
        sim.step_target(sim.home + .25 * sim.stand_action())
    rows.append(dict(case='official_standing', metric=sim.measure()))
    for case in ('getup_dynamic_r4b_prone', 'getup_crouch_r5_prone', 'getup_aligned_r6_prone'):
        ref = np.load(training / case / 'best_reference.npz', allow_pickle=False)
        sim.prepare('prone')
        stages = []
        for target, duration in zip(ref['targets'], ref['durations_s']):
            result = primitive(sim, sim.snapshot(), target, float(duration))
            stages.append(result['metric'])
        rows.append(dict(case=case, metric=sim.measure(), stages=stages))
    result = dict(rows=rows, root_state_edits_after_initialization=0,
                  simulation_only=True, hardware_readiness=False,
                  foot_orientation_definition='foot body orientation relative to nominal home; not an independent sole-normal sensor',
                  support_margin_definition='COM projection margin to convex hull of loaded foot-floor contact points; diagnostic only, not dynamic stability proof',
                  hashes={str(f): digest(f) for f in (Path(__file__), scene)})
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for row in rows:
        m = row['metric']
        print(json.dumps(dict(case=row['case'], height=m['height_m'], up=m['up_z'],
                              knees=np.asarray(m['joint_positions_rad'])[[3, 12]].tolist(),
                              foot_up=m['foot_up_alignment_to_home'], com_margin=m['com_support_margin_m'],
                              max_joint_error=m['joint_home_error_max_rad'])), flush=True)


if __name__ == '__main__':
    main()
