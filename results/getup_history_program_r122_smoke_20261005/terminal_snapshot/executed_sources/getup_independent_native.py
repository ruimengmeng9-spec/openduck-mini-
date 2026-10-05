"""Independent, simulation-only recovery reference search.

Never replaces a deployed walking actor. Ground contact uses convex mesh hulls;
self collision remains a separate, unresolved hardware-readiness gate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

DT, SUBSTEPS, SLEW = .02, 10, 5.24
POSES = ('prone', 'supine', 'left_side', 'right_side')


def canonical_quaternion(pose, tilt_offset=0.):
    # Body +X is forward, +Y is left, +Z is up. Signs refer to body axes.
    axis, angle = {'prone': (1, math.pi/2), 'supine': (1, -math.pi/2),
                   'left_side': (0, -math.pi/2), 'right_side': (0, math.pi/2)}[pose]
    angle += tilt_offset
    q = np.zeros(4)
    q[0], q[axis+1] = math.cos(angle/2), math.sin(angle/2)
    return q


def decode_full_range(action, lower, upper):
    action = np.asarray(action)
    if action.shape != lower.shape or not np.isfinite(action).all():
        raise ValueError('invalid independent recovery action')
    return (lower+upper)/2 + np.clip(action, -1., 1.)*(upper-lower)/2


def standing_sample(up, height, feet, body_contact, linear_speed, angular_speed):
    return bool(up > .95 and .125 < height < .24 and any(feet)
                and not body_contact and linear_speed < .08 and angular_speed < .5)


def sustained_tail(flags, dt=DT):
    count = 0
    for flag in reversed(flags):
        if not flag:
            break
        count += 1
    return count*dt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_model(root, output):
    """Generate experiment assets only; retain all physical parameter arrays."""
    xmls = root/'projects/Open_Duck_Playground/playground/open_duck_mini_v2/xmls'
    output.mkdir(parents=True, exist_ok=True)
    robot_src, scene_src = xmls/'open_duck_mini_v2.xml', xmls/'scene_flat_terrain.xml'
    robot = ET.parse(robot_src)
    robot.getroot().find('compiler').set('meshdir', str(xmls/'assets'))
    count = 0
    for body in robot.getroot().find('worldbody').iter('body'):
        for geom in body.findall('geom'):
            if geom.get('class') == 'visual':
                # Only floor has contype=1; hulls cannot collide with one another.
                geom.set('contype', '0')
                geom.set('conaffinity', '1')
                count += 1
    robot_path = output/'robot_ground_mesh.xml'
    robot.write(robot_path, encoding='utf-8', xml_declaration=True)
    scene = ET.parse(scene_src)
    scene.getroot().find('include').set('file', str(robot_path))
    scene_path = output/'scene.xml'
    scene.write(scene_path, encoding='utf-8', xml_declaration=True)
    original = mujoco.MjModel.from_xml_path(str(scene_src))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    arrays = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_range',
              'dof_damping', 'dof_frictionloss', 'dof_armature', 'actuator_forcerange',
              'actuator_ctrlrange', 'actuator_gainprm', 'actuator_biasprm', 'geom_friction')
    checks = {name: bool(np.array_equal(getattr(original, name), getattr(model, name))) for name in arrays}
    if not all(checks.values()):
        raise RuntimeError(f'physical parameter preservation failed: {checks}')
    floor = model.geom('floor').id
    audit = dict(simulation_only=True, hardware_readiness=False,
                 collision_contract='original foot contacts + visual convex hull ground-only contacts',
                 self_collision_validated=False, enabled_visual_geoms=count,
                 unchanged_parameter_arrays=checks, solver_iterations=model.opt.iterations,
                 friction=model.geom_friction[floor].tolist(), motor_slew_rad_s=SLEW,
                 sim_dt_s=.002, control_dt_s=DT,
                 source_hashes={str(p): digest(p) for p in (robot_src, scene_src)},
                 generated_hashes={str(p): digest(p) for p in (robot_path, scene_path)})
    (output/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return scene_path, audit


class RecoverySim:
    def __init__(self, scene_path, stand_model):
        import onnxruntime as ort
        self.model = m = mujoco.MjModel.from_xml_path(str(scene_path))
        m.opt.timestep = .002
        self.data = mujoco.MjData(m)
        self.joints = m.actuator_trnid[:, 0]
        self.qadr, self.vadr = m.jnt_qposadr[self.joints], m.jnt_dofadr[self.joints]
        self.lower, self.upper = m.jnt_range[self.joints].T.copy()
        self.home = m.keyframe('home').qpos[self.qadr].copy()
        self.feet = [m.body(n).id for n in ('foot_assembly', 'foot_assembly_2')]
        self.floor = m.geom('floor').id
        self.torso = {m.body(n).id for n in ('trunk_assembly', 'head_assembly')}
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.actor = ort.InferenceSession(str(stand_model), sess_options=options,
                                         providers=['CPUExecutionProvider'])
        self.sensor_indices = {s: (int(m.sensor_adr[m.sensor(s).id]), int(m.sensor_dim[m.sensor(s).id]))
                               for s in ('gyro', 'accelerometer', 'local_linvel', 'global_angvel', 'upvector')}
        self.prev = self.home.copy()
        self.history = np.zeros((3, m.nu), dtype=np.float32)

    def sensor(self, name):
        adr, size = self.sensor_indices[name]
        return self.data.sensordata[adr:adr+size].copy()

    def contacts(self):
        bodies = set()
        penetration = 0.
        for c in self.data.contact:
            if self.floor not in c.geom:
                continue
            penetration = max(penetration, float(-c.dist))
            if c.dist <= .001:
                other = int(c.geom[1] if c.geom[0] == self.floor else c.geom[0])
                bodies.add(int(self.model.geom_bodyid[other]))
        return [b in bodies for b in self.feet], bool(bodies & self.torso), penetration

    def lowest_geom(self):
        lows = []
        m, d = self.model, self.data
        for i in range(m.ngeom):
            if i == self.floor or not (m.geom_contype[i] or m.geom_conaffinity[i]):
                continue
            r = d.geom_xmat[i].reshape(3, 3)
            if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[i]
                a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                z = (m.mesh_vert[a:a+n] @ r[2]) + d.geom_xpos[i, 2]
                lows.append(float(z.min()))
            else:
                # Original contact shapes are meshes. This branch is conservative.
                lows.append(float(d.geom_xpos[i, 2]-np.dot(np.abs(r[2]), m.geom_size[i])))
        return min(lows)

    def step_target(self, target):
        target = np.clip(np.asarray(target), self.lower, self.upper)
        applied = np.clip(target, self.prev-SLEW*DT, self.prev+SLEW*DT)
        raw = ((target-self.home)/.25).astype(np.float32)
        self.history[2] = self.history[1].copy()
        self.history[1] = self.history[0].copy()
        self.history[0] = raw
        self.data.ctrl[:] = applied
        self.prev = applied.copy()
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        return applied

    def stand_action(self):
        feet, _, _ = self.contacts()
        accel = self.sensor('accelerometer')
        accel[0] += 1.3  # Preserve the official native inference contract.
        obs = np.concatenate([self.sensor('gyro'), accel, np.zeros(7),
                              self.data.qpos[self.qadr]-self.home,
                              self.data.qvel[self.vadr]*.05, *self.history,
                              self.prev, feet, [1., 0.]]).astype(np.float32)
        return self.actor.run(None, {self.actor.get_inputs()[0].name: obs[None]})[0].reshape(-1)

    def prepare(self, pose, seed=0, perturb=False):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.keyframe('home').id)
        rng = np.random.default_rng(seed)
        if pose != 'standing':
            self.data.qpos[3:7] = canonical_quaternion(pose, rng.uniform(-.05, .05) if perturb else 0.)
        if perturb:
            self.data.qpos[self.qadr] = np.clip(self.home+rng.uniform(-.02, .02, self.model.nu), self.lower, self.upper)
        self.data.qpos[2] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[2] += .005-self.lowest_geom()
        self.data.qvel[:] = rng.uniform(-.01, .01, self.model.nv) if perturb else 0.
        self.prev, self.history = self.home.copy(), np.zeros_like(self.history)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(40):
            self.step_target(self.home)
        return self.snapshot()

    def snapshot(self):
        return dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
                    act=self.data.act.copy(), warmstart=self.data.qacc_warmstart.copy(),
                    time=float(self.data.time), prev=self.prev.copy(), history=self.history.copy())

    def restore(self, state):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = state['qpos']
        self.data.qvel[:] = state['qvel']
        self.data.act[:] = state['act']
        self.data.time = state['time']
        self.prev, self.history = state['prev'].copy(), state['history'].copy()
        self.data.ctrl[:] = self.prev
        mujoco.mj_forward(self.model, self.data)
        self.data.qacc_warmstart[:] = state['warmstart']

    def measure(self):
        feet, torso, penetration = self.contacts()
        up, height = float(self.sensor('upvector')[2]), float(self.data.qpos[2])
        speed, angular = float(np.linalg.norm(self.sensor('local_linvel'))), float(np.linalg.norm(self.sensor('global_angvel')))
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        valid = finite and penetration < .02
        stable = valid and standing_sample(up, height, feet, torso, speed, angular)
        return dict(up_z=up, height_m=height, feet=feet, torso_contact=torso,
                    penetration_m=penetration, linear_speed_mps=speed, angular_speed_rad_s=angular,
                    stable=stable, finite=finite)


def feature_targets(features, home, lower, upper):
    """Seven bilateral sagittal/roll/yaw controls per knot, in radians."""
    targets = np.tile(home, (len(features), 1))
    for target, row in zip(targets, features):
        hip, knee, ankle, neck, head, roll, yaw = row
        target[[2, 11]] = [hip, -hip]
        target[[3, 12]], target[[4, 13]] = knee, ankle
        target[5], target[6] = neck, head
        target[[1, 10]] = [roll, -roll]
        target[[0, 9]] = [yaw, -yaw]
    return np.clip(targets, lower, upper)


def episode(sim, initial, targets, duration=5., record=False):
    sim.restore(initial)
    initial_measure = sim.measure()
    # Start and end at home, so search cannot declare an exotic pose a stand.
    knots = np.concatenate([sim.home[None], targets, sim.home[None]])
    flags, measurements, trajectory = [], [], []
    max_force = max_slew = 0.
    previous = sim.prev.copy()
    for i in range(round((duration+4.)/DT)):
        t = i*DT
        if t < duration:
            f = t/duration*(len(knots)-1)
            j = min(int(f), len(knots)-2)
            a = f-j
            a = a*a*(3.-2.*a)
            target = (1-a)*knots[j]+a*knots[j+1]
        elif t < duration+1.:
            target = sim.home
        else:
            raw = sim.stand_action()
            stand = sim.home+.25*raw
            u = min(t-duration-1., 1.)
            target = (1-u)*sim.home+u*stand
        applied = sim.step_target(target)
        max_slew = max(max_slew, float(np.abs(applied-previous).max()/DT))
        previous = applied.copy()
        max_force = max(max_force, float(np.abs(sim.data.actuator_force).max()))
        m = sim.measure()
        measurements.append(m)
        flags.append(m['stable'])
        if record:
            trajectory.append((float(sim.data.time), sim.data.qpos.copy(), sim.data.qvel.copy(), applied.copy()))
        if not m['finite']:
            break
    tail = measurements[-100:]
    initial_fallen = initial_measure['up_z'] < .5 and initial_measure['torso_contact']
    sustained = sustained_tail(flags)
    valid = all(x['finite'] for x in measurements) and max(x['penetration_m'] for x in measurements) < .02
    success = initial_fallen and valid and sustained >= 2.-1e-8 and len(measurements) == round((duration+4.)/DT)
    score = np.mean([3.*max(x['up_z'], -1.) + 2.*min(x['height_m']/.16, 1.)
                     + .6*sum(x['feet'])-.7*x['torso_contact']
                     -.2*min(x['linear_speed_mps'], 3.)-.08*min(x['angular_speed_rad_s'], 5.) for x in tail])
    score += 3.*sustained + 30.*success - 50.*max(x['penetration_m'] for x in measurements)
    result = dict(success=bool(success), score=float(score), initial_fallen=bool(initial_fallen),
                  initial=initial_measure, final=measurements[-1], sustained_final_stand_s=sustained,
                  maximum_up_z=max(x['up_z'] for x in measurements),
                  maximum_penetration_m=max(x['penetration_m'] for x in measurements),
                  max_sampled_actuator_force_nm=max_force, max_command_slew_rad_s=max_slew,
                  stand_policy_handoff_tested=True, duration_s=len(measurements)*DT,
                  simulation_only=True, hardware_readiness=False)
    return result, trajectory


_worker = None


def worker_init(scene, stand_model, pose, duration):
    global _worker
    sim = RecoverySim(scene, stand_model)
    _worker = sim, sim.prepare(pose), duration


def worker_evaluate(features):
    sim, initial, duration = _worker
    target = feature_targets(features, sim.home, sim.lower, sim.upper)
    return episode(sim, initial, target, duration)[0]


def search(root, output, pose, generations, population, workers, knots, duration, seed):
    scene, audit = build_model(root, output/'model')
    stand_model = root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = RecoverySim(scene, stand_model)
    home = np.array([sim.home[2], sim.home[3], sim.home[4], 0., 0., sim.home[1], 0.])
    lower = np.array([-1.2217, -1.5708, -1.5708, -.3491, -.7854, -.4363, -.5236])
    upper = np.array([.5236, 1.5708, 1.5708, 1.1345, .7854, .4363, .5236])
    mean = np.tile(home, (knots, 1))
    std = np.tile((upper-lower)*.4, (knots, 1))
    rng = np.random.default_rng(seed)
    best_score, best_features = -np.inf, mean.copy()
    history = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                             initargs=(str(scene), str(stand_model), pose, duration)) as pool:
        for generation in range(generations):
            samples = np.clip(rng.normal(mean, std, (population, knots, 7)), lower, upper)
            samples[0], samples[1] = best_features, mean
            results = list(pool.map(worker_evaluate, samples, chunksize=1))
            scores = np.array([x['score'] for x in results])
            elites = samples[np.argsort(scores)[-max(4, population//8):]]
            mean = .25*mean+.75*elites.mean(axis=0)
            std = np.maximum(.25*std+.75*elites.std(axis=0), (upper-lower)*.025)
            j = int(scores.argmax())
            if scores[j] > best_score:
                best_score, best_features = float(scores[j]), samples[j].copy()
                np.savez_compressed(output/'best_reference.npz', features=best_features,
                                    targets=feature_targets(best_features, sim.home, sim.lower, sim.upper),
                                    duration_s=duration)
            row = dict(generation=generation, best_score=best_score, mean_score=float(scores.mean()),
                       successful_candidates=sum(x['success'] for x in results),
                       best_this_generation=results[j], wall_seconds=time.time()-started)
            history.append(row)
            (output/'search_progress.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
    targets = feature_targets(best_features, sim.home, sim.lower, sim.upper)
    validations = []
    for test_seed in range(20):
        initial = sim.prepare(pose, 9000+test_seed, perturb=test_seed>0)
        result, trajectory = episode(sim, initial, targets, duration, record=test_seed == 0)
        result['seed'] = 9000+test_seed
        validations.append(result)
        if trajectory:
            np.savez_compressed(output/'best_trajectory.npz', time=[x[0] for x in trajectory],
                                qpos=[x[1] for x in trajectory], qvel=[x[2] for x in trajectory], ctrl=[x[3] for x in trajectory])
    summary = dict(pose=pose, seed=seed, method='bounded joint-reference CEM; not yet neural distillation',
                   generations=generations, population=population, duration_s=duration, knots=knots,
                   successful_validation_runs=sum(x['success'] for x in validations), validation_runs=20,
                   results=validations, audit=audit, stand_policy_sha256=digest(stand_model),
                   script_sha256=digest(__file__), hardware_readiness=False)
    (output/'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('VALIDATION:', summary['successful_validation_runs'], '/20', flush=True)


def precheck(root, output):
    scene, audit = build_model(root, output/'model')
    sim = RecoverySim(scene, root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    rows = []
    for pose in ('standing', *POSES):
        initial = sim.prepare(pose)
        m = sim.measure()
        result, _ = episode(sim, initial, np.tile(sim.home, (5, 1)))
        rows.append(dict(pose=pose, settled=m, hold_home_and_stand=result))
        print(json.dumps(rows[-1]), flush=True)
    (output/'precheck.json').write_text(json.dumps(dict(audit=audit, poses=rows), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precheck', action='store_true')
    parser.add_argument('--pose', choices=POSES, default='supine')
    parser.add_argument('--generations', type=int, default=16)
    parser.add_argument('--population', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--knots', type=int, default=5)
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--seed', type=int, default=72)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.precheck:
        precheck(args.root, args.output)
    else:
        search(args.root, args.output, args.pose, args.generations, args.population,
               args.workers, args.knots, args.duration, args.seed)


if __name__ == '__main__':
    main()
