"""Screen convex-hull self collisions without modifying any training scene.

This is deliberately a readiness gate, NOT a proof of exact CAD clearance.
MuJoCo convex hulls may overapproximate hollow parts; review pairs explicitly.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from diagnostics.getup_independent_native import RecoverySim, POSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--experiment', type=Path, required=True)
    args = p.parse_args()
    folder = args.experiment/'model'
    robot = ET.parse(folder/'robot_ground_mesh.xml')
    for body in robot.getroot().find('worldbody').iter('body'):
        for geom in body.findall('geom'):
            if geom.get('class') in ('visual', 'collision'):
                geom.set('contype', '2')
                geom.set('conaffinity', '3')
    robot_path = folder/'robot_self_collision_audit.xml'
    robot.write(robot_path, encoding='utf-8', xml_declaration=True)
    scene = ET.parse(folder/'scene.xml')
    scene.getroot().find('include').set('file', str(robot_path))
    scene_path = folder/'scene_self_collision_audit.xml'
    scene.write(scene_path, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    sim = RecoverySim(folder/'scene.xml', args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    pairs, counts = defaultdict(lambda: dict(max_penetration_m=0., frames=0)), []

    def check(qpos, qvel, label):
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)
        found = set()
        for c in data.contact:
            ids = list(map(int, c.geom))
            bodies = [model.body(int(model.geom_bodyid[g])).name for g in ids]
            if 'floor' in bodies or 'world' in bodies or c.dist >= -.001:
                continue
            key = ' / '.join(sorted(bodies))
            pairs[key]['max_penetration_m'] = max(pairs[key]['max_penetration_m'], float(-c.dist))
            found.add(key)
        for key in found:
            pairs[key]['frames'] += 1
        counts.append(dict(label=label, self_contact_pairs=len(found)))

    for pose in ('standing', *POSES):
        state = sim.prepare(pose)
        check(state['qpos'], state['qvel'], pose)
    trajectory = args.experiment/'best_trajectory.npz'
    if trajectory.exists():
        tr = np.load(trajectory, allow_pickle=False)
        for i in range(len(tr['time'])):
            check(tr['qpos'][i], tr['qvel'][i], f"trajectory_{i}")
    result = dict(note='convex-hull screen, not exact CAD validation; no audit collisions fed into training',
                  hardware_readiness=False, geometry_source='original visual meshes',
                  initial_pose_counts=counts[:5], frames_screened=len(counts),
                  frames_with_self_penetration=sum(c['self_contact_pairs'] > 0 for c in counts),
                  pairs=dict(sorted(pairs.items(), key=lambda x: -x[1]['max_penetration_m'])))
    (args.experiment/'self_collision_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
