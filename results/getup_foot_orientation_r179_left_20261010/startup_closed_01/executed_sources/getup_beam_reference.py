"""R3 kinodynamic beam search: retain reachable intermediate recovery states.

Branch snapshots are search bookkeeping. A recorded rollout always starts once
and executes the entire primitive chain without pose/velocity resets.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from diagnostics.getup_independent_native import (
    DT, RecoverySim, build_model, feature_targets, digest, sustained_tail, POSES)
from diagnostics.getup_feedback_reference import ready_to_handoff


_sim = None


def init_worker(scene, stand):
    global _sim
    _sim = RecoverySim(scene, stand)


def primitive(sim, state, target, duration, record=False):
    sim.restore(state)
    sim.data.ctrl[:] = sim.prev
    mujoco.mj_forward(sim.model, sim.data)
    start = sim.prev.copy()
    measures, trace = [], []
    max_slew = max_force = max_joint_overshoot = 0.
    previous = sim.prev.copy()
    for i in range(round(duration/DT)):
        u = min((i+1)*DT/(duration*.7), 1.)
        u = u*u*(3.-2.*u)
        applied = sim.step_target((1-u)*start+u*target)
        max_slew = max(max_slew, float(np.abs(applied-previous).max()/DT))
        previous = applied.copy()
        max_force = max(max_force, float(np.abs(sim.data.actuator_force).max()))
        joints = sim.data.qpos[sim.qadr]
        max_joint_overshoot = max(max_joint_overshoot, float(np.maximum(sim.lower-joints, joints-sim.upper).max()))
        m = sim.measure()
        measures.append(m)
        if record:
            trace.append((float(sim.data.time), sim.data.qpos.copy(), sim.data.qvel.copy(), applied.copy(), 'reference'))
        if not m['finite']:
            break
    last = measures[-1]
    progress = np.mean([2.*m['up_z']+min(m['height_m']/.16, 1.2)+.25*sum(m['feet'])
                        -.35*m['torso_contact']-.08*min(m['angular_speed_rad_s'], 8.) for m in measures[-10:]])
    score = progress+2.*ready_to_handoff(last)-40.*max(m['penetration_m'] for m in measures)
    return dict(state=sim.snapshot(), metric=last, score=float(score), trace=trace,
                measurements=measures, maximum_penetration_m=max(m['penetration_m'] for m in measures),
                maximum_joint_overshoot_rad=max_joint_overshoot, max_sampled_actuator_force_nm=max_force,
                max_command_slew_rad_s=max_slew)


def evaluate(job):
    state, target, duration = job
    return primitive(_sim, state, target, duration)


def replay(sim, pose, targets, duration, seed=0, perturb=False, record=False):
    initial = sim.prepare(pose, seed, perturb)
    init = sim.measure()
    trace, measures, handoff_t = [], [], None
    max_slew = max_force = max_joint_overshoot = 0.
    for target in targets:
        result = primitive(sim, sim.snapshot(), target, duration, record)
        trace.extend(result['trace'])
        measures.extend(result['measurements'])
        max_slew = max(max_slew, result['max_command_slew_rad_s'])
        max_force = max(max_force, result['max_sampled_actuator_force_nm'])
        max_joint_overshoot = max(max_joint_overshoot, result['maximum_joint_overshoot_rad'])
        if ready_to_handoff(result['metric']):
            handoff_t = float(sim.data.time)-initial['time']
            break
    from_target = sim.prev.copy()
    flags = []
    previous = sim.prev.copy()
    for i in range(200):
        raw = sim.stand_action()
        u = min(i*DT, 1.)
        u = u*u*(3.-2.*u)
        applied = sim.step_target((1-u)*from_target+u*(sim.home+.25*raw))
        max_slew = max(max_slew, float(np.abs(applied-previous).max()/DT))
        previous = applied.copy()
        max_force = max(max_force, float(np.abs(sim.data.actuator_force).max()))
        joints = sim.data.qpos[sim.qadr]
        max_joint_overshoot = max(max_joint_overshoot, float(np.maximum(sim.lower-joints, joints-sim.upper).max()))
        m = sim.measure()
        measures.append(m)
        flags.append(m['stable'])
        if record:
            trace.append((float(sim.data.time), sim.data.qpos.copy(), sim.data.qvel.copy(), applied.copy(), 'stand_handoff'))
    sustained = sustained_tail(flags)
    initial_fallen = init['up_z'] < .5 and init['torso_contact']
    penetration = max(m['penetration_m'] for m in measures)
    success = (initial_fallen and sustained >= 2.-1e-8 and penetration < .01
               and max_joint_overshoot < .08 and max_slew <= 5.24+1e-8 and max_force <= 3.23+1e-8
               and all(m['finite'] for m in measures))
    return dict(success=bool(success), seed=seed, initial=init, final=measures[-1],
                sustained_final_stand_s=sustained, maximum_penetration_m=penetration,
                maximum_up_z=max(m['up_z'] for m in measures), state_gate_time_s=handoff_t,
                maximum_joint_overshoot_rad=max_joint_overshoot,
                max_sampled_actuator_force_nm=max_force, max_command_slew_rad_s=max_slew,
                simulation_only=True, hardware_readiness=False), trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pose', choices=POSES, default='prone')
    p.add_argument('--depth', type=int, default=7)
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--duration', type=float, default=.8)
    p.add_argument('--seed', type=int, default=94)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene, audit = build_model(args.root, args.output/'model')
    stand = args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = RecoverySim(scene, stand)
    initial = sim.prepare(args.pose)
    beam = [dict(state=initial, path=[], score=0., metric=sim.measure())]
    rng, history, started = np.random.default_rng(args.seed), [], time.time()
    best = beam[0]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand))) as pool:
        for depth in range(args.depth):
            rows = []
            for hip in (-1.2, -.6, .1, .52):
                for knee in (-1.5, -.3, 1.4):
                    for ankle in (-1.4, 0., 1.4):
                        for neck in (-.3, 1.1):
                            rows.append([hip, knee, ankle, neck, 0., .05, 0.])
            library = feature_targets(rows, sim.home, sim.lower, sim.upper)
            random_targets = rng.uniform(sim.lower, sim.upper, (80, sim.model.nu))
            random_targets[:, [7, 8]] = 0.
            library = np.concatenate([library, random_targets, sim.home[None]])
            jobs = [(parent['state'], target, args.duration) for parent in beam for target in library]
            evaluated = list(pool.map(evaluate, jobs, chunksize=4))
            candidates = []
            for i, result in enumerate(evaluated):
                parent = beam[i//len(library)]
                if (not all(m['finite'] for m in result['measurements']) or result['maximum_penetration_m'] > .01
                        or result['maximum_joint_overshoot_rad'] > .08):
                    continue
                result['path'] = parent['path']+[library[i%len(library)].copy()]
                candidates.append(result)
            candidates.sort(key=lambda x: -x['score'])
            # Retain multiple reachable posture bins, not only copies of one maximum.
            bins, beam = set(), []
            for candidate in candidates:
                m = candidate['metric']
                up = candidate['state']['qpos'][3:7]
                key = (*np.round(up/.15).astype(int), round(m['height_m']/.025),
                       *np.round(candidate['state']['prev'][[2, 3, 4, 5]]/.4).astype(int))
                if key not in bins:
                    bins.add(key)
                    beam.append(candidate)
                if len(beam) == args.width:
                    break
            if not beam:
                raise RuntimeError('no valid reachable nodes remain')
            best = beam[0]
            np.savez_compressed(args.output/'best_reference.npz', targets=best['path'], primitive_duration_s=args.duration)
            row = dict(depth=depth+1, expanded=len(jobs), best_score=best['score'],
                       best_metric=best['metric'], ready_nodes=sum(ready_to_handoff(x['metric']) for x in candidates),
                       wall_seconds=time.time()-started)
            history.append(row)
            (args.output/'search_progress.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
            if ready_to_handoff(best['metric']):
                break
    results = []
    for seed in range(20):
        result, trace = replay(sim, args.pose, best['path'], args.duration, 11000+seed, seed>0, seed==0)
        results.append(result)
        if trace:
            np.savez_compressed(args.output/'best_trajectory.npz', time=[r[0] for r in trace],
                                qpos=[r[1] for r in trace], qvel=[r[2] for r in trace],
                                ctrl=[r[3] for r in trace], phase=[r[4] for r in trace])
    summary = dict(pose=args.pose, method='kinodynamic primitive beam search + state gate + stand handoff',
                   successful_validation_runs=sum(r['success'] for r in results), validation_runs=len(results),
                   results=results, audit=audit, simulation_only=True, hardware_readiness=False,
                   script_sha256=digest(__file__), native_sim_sha256=digest(Path(__file__).with_name('getup_independent_native.py')),
                   seed=args.seed, depth=args.depth, width=args.width, primitive_duration_s=args.duration)
    (args.output/'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('VALIDATION:', summary['successful_validation_runs'], '/20', flush=True)


if __name__ == '__main__':
    main()
