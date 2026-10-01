"""Build an independent CoACD collision experiment from original CAD meshes.

Uses compiled mesh frames AND compiled geom poses together: mixing these with
source XML geom poses would silently displace the derived collision shapes.
All inertial, joint and motor parameters are retained. Approximation metrics
are diagnostic only; this generator does not declare hardware readiness.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from diagnostics.getup_independent_native import digest, RecoverySim, POSES


def decompose(job):
    import coacd
    coacd.set_log_level('warn')
    name, vertices, faces, threshold = job
    started = time.time()
    parts = coacd.run_coacd(coacd.Mesh(vertices.astype(np.float64), faces.astype(np.int32)),
                           threshold=threshold, real_metric=True, preprocess_mode='auto',
                           preprocess_resolution=100, resolution=2000,
                           mcts_nodes=12, mcts_iterations=40, mcts_max_depth=3,
                           merge=True, max_convex_hull=32, seed=127)
    if not parts:
        raise RuntimeError(f'empty decomposition: {name}')
    directions = np.array([[x, y, z] for x in (-1., 0., 1.) for y in (-1., 0., 1.)
                           for z in (-1., 0., 1.) if x*x+y*y+z*z > 0])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    all_vertices = np.concatenate([v for v, _ in parts])
    original_support = np.max(vertices @ directions.T, axis=0)
    derived_support = np.max(all_vertices @ directions.T, axis=0)
    audit = dict(name=name, components=len(parts), original_vertices=len(vertices),
                 original_faces=len(faces), maximum_sampled_support_error_m=float(np.abs(original_support-derived_support).max()),
                 wall_seconds=time.time()-started, threshold_m=threshold)
    return name, parts, audit


def text_array(values):
    return ' '.join(f'{float(x):.10g}' for x in np.asarray(values).ravel())


def self_contacts(model, data):
    pairs = {}
    floor = model.geom('floor').id
    for c in data.contact:
        if floor in c.geom or c.dist >= -.001:
            continue
        bodies = [model.body(int(model.geom_bodyid[g])).name for g in c.geom]
        key = ' / '.join(sorted(bodies))
        pairs[key] = max(pairs.get(key, 0.), float(-c.dist))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threshold', type=float, default=.002)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    xmls = args.root/'projects/Open_Duck_Playground/playground/open_duck_mini_v2/xmls'
    robot_src, scene_src = xmls/'open_duck_mini_v2.xml', xmls/'scene_flat_terrain.xml'
    original = mujoco.MjModel.from_xml_path(str(scene_src))
    robot = ET.parse(robot_src)
    robot.getroot().find('compiler').set('meshdir', str(xmls/'assets'))
    asset = robot.getroot().find('asset')
    geom_records, used_meshes = [], set()
    for body in robot.getroot().find('worldbody').iter('body'):
        geoms = body.findall('geom')
        body_id = original.body(body.get('name')).id
        compiled_ids = np.flatnonzero(original.geom_bodyid == body_id)
        if len(geoms) != len(compiled_ids):
            raise RuntimeError(f'body geometry mapping mismatch: {body.get("name")}')
        for element, g in zip(geoms, compiled_ids):
            element.set('contype', '0')
            element.set('conaffinity', '0')
            if element.get('class') == 'visual':
                mesh_id = int(original.geom_dataid[g])
                used_meshes.add(mesh_id)
                geom_records.append((body, g, mesh_id))
    jobs = []
    for mesh_id in sorted(used_meshes):
        name = original.mesh(mesh_id).name
        a, n = original.mesh_vertadr[mesh_id], original.mesh_vertnum[mesh_id]
        fa, fn = original.mesh_faceadr[mesh_id], original.mesh_facenum[mesh_id]
        jobs.append((name, original.mesh_vert[a:a+n].copy(), original.mesh_face[fa:fa+fn].copy(), args.threshold))
    parts_map, audits = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, parts, audit in pool.map(decompose, jobs, chunksize=1):
            parts_map[name] = parts
            audits.append(audit)
            print(json.dumps(audit), flush=True)
            np.savez_compressed(args.output/f'pieces_{name}.npz',
                                **{f'vertices_{i}': v for i, (v, _) in enumerate(parts)},
                                **{f'faces_{i}': f for i, (_, f) in enumerate(parts)})
            (args.output/'decomposition_progress.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    for name, parts in parts_map.items():
        for i, (vertices, faces) in enumerate(parts):
            ET.SubElement(asset, 'mesh', name=f'recovery_{name}_{i}', vertex=text_array(vertices),
                          face=' '.join(str(int(x)) for x in np.asarray(faces).ravel()))
    collider_count = 0
    for body, geom_id, mesh_id in geom_records:
        name = original.mesh(mesh_id).name
        for i in range(len(parts_map[name])):
            ET.SubElement(body, 'geom', name=f'getup_collision_{geom_id}_{i}', type='mesh',
                          mesh=f'recovery_{name}_{i}', pos=text_array(original.geom_pos[geom_id]),
                          quat=text_array(original.geom_quat[geom_id]), contype='2', conaffinity='3',
                          group='3', rgba='.3 .5 .9 .25')
            collider_count += 1
    ET.indent(robot, space='  ')
    robot_path = args.output/'robot_decomposed.xml'
    robot.write(robot_path, encoding='utf-8', xml_declaration=True)
    scene = ET.parse(scene_src)
    scene.getroot().find('include').set('file', str(robot_path))
    ET.indent(scene, space='  ')
    scene_path = args.output/'scene.xml'
    scene.write(scene_path, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    parameters = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_range',
                  'dof_damping', 'dof_frictionloss', 'dof_armature', 'actuator_forcerange',
                  'actuator_ctrlrange', 'actuator_gainprm', 'actuator_biasprm')
    checks = {key: bool(np.array_equal(getattr(original, key), getattr(model, key))) for key in parameters}
    if not all(checks.values()):
        raise RuntimeError(f'physical parameter preservation failed: {checks}')
    sim = RecoverySim(scene_path, args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    rows = []
    for pose in ('standing', *POSES):
        sim.prepare(pose)
        m = sim.measure()
        collisions = self_contacts(sim.model, sim.data)
        stable_count = 0
        for _ in range(100):
            sim.step_target(sim.home if pose != 'standing' else sim.home+.25*sim.stand_action())
            stable_count += sim.measure()['stable']
        rows.append(dict(pose=pose, settled=m, self_contacts=collisions, final=sim.measure(),
                         standing_stable_samples_out_of_100=stable_count))
        print(json.dumps(rows[-1]), flush=True)
    audit = dict(coacd_version=version('coacd'), threshold_m=args.threshold, mesh_audits=audits,
                 unchanged_parameters=checks, collision_geoms=collider_count,
                 friction=model.geom_friction[model.geom('floor').id].tolist(),
                 collision_contract='CoACD decomposed CAD meshes, floor and nonadjacent self collisions enabled',
                 pose_checks=rows, hardware_readiness=False,
                 model_approximation_validated=False,
                 hashes={str(path): digest(path) for path in (robot_src, scene_src, robot_path, scene_path, Path(__file__))})
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
