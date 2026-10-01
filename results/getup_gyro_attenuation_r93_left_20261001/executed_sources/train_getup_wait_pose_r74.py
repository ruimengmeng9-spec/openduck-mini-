"""Test the home-wait contact-basin hypothesis; simulation only.

R73 kept the zero correction and did not improve held-out full falls. R71
post-control trajectories show ten of eleven training failures tip in phase
1, before the terminal flip. Learn only a smooth phase-1 target offset, not
physics, timing, earlier commands, terminal targets or IMU gains. Restore is
used only at the start of independent trials, never during a trial.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.probe_getup_terminal_timing_r72 import timed_targets
from diagnostics.search_getup_reference_feedback_r64 import record_reference, rollout
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.train_getup_prefix_pose_r73 import success
from diagnostics.validate_getup_fullpath_r27 import StrictSim

_CTX = None
BOUND = .25


def edited_wait(sim, ck, flat):
    delta = np.asarray(flat, dtype=float)
    if delta.shape != (len(JOINTS),) or not np.isfinite(delta).all():
        raise ValueError('Need ten finite wait-pose offsets')
    if np.abs(delta).max() > BOUND + 1e-10:
        raise ValueError('Wait-pose correction exceeds search bound')
    prefix = ck['prefix_targets'].copy()
    indices = np.flatnonzero(ck['prefix_phases'] == 1)
    if len(indices) < 20 or np.any(np.diff(indices) != 1):
        raise ValueError('Need one contiguous audited home-wait phase')
    # Ramp in/out over 0.2 s; the final wait command remains exactly unchanged.
    ramp = round(.2 / DT)
    k = np.arange(len(indices))
    envelope = np.minimum(np.minimum((k + 1) / ramp,
                                    (len(indices) - 1 - k) / ramp), 1.)
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    prefix[np.ix_(indices, ids)] = np.clip(
        prefix[np.ix_(indices, ids)] + envelope[:, None] * delta,
        sim.lower[ids], sim.upper[ids])
    return {**ck, 'prefix_targets': prefix}


def init_worker(scene, stand, ck, seeds):
    global _CTX
    sim = StrictSim(scene, stand)
    cases = {}
    for seed in [None] + list(seeds):
        sim.prepare('left_side', 0 if seed is None else seed, seed is not None)
        initial = sim.measure()
        if not (initial['up_z'] < .5 and initial['torso_contact']):
            raise RuntimeError('Trial must start physically fallen')
        sim.clear_audit()
        cases[seed] = (sim.snapshot(), sim.peaks.copy(), initial, state_hash(sim))
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, cases, ck, ids


def run_path(flat, hold):
    sim, cases, ck, ids = _CTX
    changed = edited_wait(sim, ck, flat)
    targets, phases = timed_targets(sim, changed, (.4, .6, .8), hold)
    source, peaks, _, _ = cases[None]
    ref = record_reference(sim, source, peaks, True, targets)
    gains = np.concatenate([ck['prefix_gains'], ck['terminal_gains']])
    return targets, phases, ref, gains


def objective(flat):
    sim, cases, ck, ids = _CTX
    try:
        targets, phases, ref, gains = run_path(flat, 2.)
    except RuntimeError:
        return {'objective': -10000., 'successes': 0, 'nominal_success': False}
    nominal, _ = rollout(sim, *cases[None][:2], True, targets, phases, ids, ref, gains)
    if not success(nominal, len(targets), 1.):
        return {'objective': -1000. + nominal['score'], 'successes': 0,
                'nominal_success': False, 'nominal': nominal}
    rows = [rollout(sim, *case[:2], True, targets, phases, ids, ref, gains)[0]
            for seed, case in cases.items() if seed is not None]
    passed = sum(success(r, len(targets), 1.) for r in rows)
    scores = np.sort([r['score'] for r in rows])
    value = float(scores[:6].mean() + .2 * scores.mean() + 14 * passed
                  - .1 * np.dot(flat, flat))
    return {'objective': value, 'successes': passed, 'nominal_success': True,
            'nominal': nominal, 'cases': rows}


def qualify(job):
    seed, flat, directory, hold, required = job
    sim, cases, ck, ids = _CTX
    source, peaks, initial, initial_hash = cases[seed]
    row = {'seed': seed, 'initial_fallen': True, 'initial': initial,
           'initial_state_sha256': initial_hash}
    for name, params in [('baseline', np.zeros(len(JOINTS))), ('candidate', flat)]:
        targets, phases, ref, gains = run_path(params, hold)
        value, trace = rollout(sim, source, peaks, True, targets, phases, ids,
                               ref, gains, True)
        value['success'] = success(value, len(targets), required)
        row[name] = value
        np.savez_compressed(Path(directory) / f'{name}_{seed}.npz',
                            time=[r[0] for r in trace], qpos=[r[1] for r in trace],
                            qvel=[r[2] for r in trace], phase=[r[3] for r in trace],
                            strict=[r[4] for r in trace], margin_m=[r[5] for r in trace],
                            imu_error=[r[6] for r in trace], residual_rad=[r[7] for r in trace])
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--generations', type=int, default=32)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--seed', type=int, default=174)
    args = p.parse_args()
    if args.output.exists() or args.workers < 1 or args.generations < 1 or args.population < 24:
        p.error('Need fresh output and bounded positive search settings')
    root = Path('/data/shijinsheng/open_duck')
    scene = root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    with np.load(args.checkpoint, allow_pickle=False) as z:
        ck = {k: z[k] for k in z.files}
    train = list(range(769000, 769008)) + list(range(773000, 773016))
    heldout = list(range(777000, 777040))
    assert not set(train) & set(heldout)
    args.output.mkdir(parents=True)
    contract = {'simulation_only': True, 'hardware_readiness': False,
                'full_task_completed': False, 'no_midpath_state_reset': True,
                'method': 'smooth home-wait-only target offset CEM', 'pose': 'left_side',
                'hypothesis': 'home wait returns unstable contact configurations to wrong basin',
                'training_seeds': train, 'heldout_seeds': heldout,
                'learned_joint_order': list(JOINTS), 'delta_bound_rad': BOUND,
                'ramp_seconds': .2, 'training_gate_seconds': 1.,
                'required_strict_seconds': 30., 'hold_seconds': 35.,
                'frozen': ['prewait_commands', 'postwait_commands', 'phase_durations',
                           'feedback_gains', 'physics', 'motor_limits', 'strict_gate'],
                'source_sha256': digest(__file__), 'scene_sha256': digest(scene),
                'stand_sha256': digest(stand), 'checkpoint_sha256': digest(args.checkpoint),
                'dependency_hashes': {name: digest(Path(__file__).parent / name) for name in
                    ['probe_getup_terminal_timing_r72.py', 'search_getup_reference_feedback_r64.py',
                     'validate_getup_fullpath_r27.py', 'train_getup_fullpath_r27.py',
                     'getup_independent_native.py', 'train_getup_prefix_pose_r73.py']},
                'seed': args.seed, 'generation_budget': args.generations,
                'population': args.population, 'workers': args.workers}
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    rng = np.random.default_rng(args.seed)
    mean, std = np.zeros(len(JOINTS)), np.full(len(JOINTS), .055)
    history, best, stale = [], None, 0
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), ck, train)) as pool:
        for gen in range(args.generations):
            population = [np.zeros(len(JOINTS)), mean.copy()]
            if best is not None:
                population.append(best[0].copy())
            if gen == 0:
                for j in range(len(JOINTS)):
                    for sign in [-1., 1.]:
                        v = np.zeros(len(JOINTS)); v[j] = sign * .04
                        population.append(v)
            while len(population) < args.population:
                center = best[0] if best and len(population) % 3 == 0 else mean
                population.append(np.clip(center + rng.normal(0., std), -BOUND, BOUND))
            ranked = sorted(zip(population, pool.map(objective, population)),
                            key=lambda r: -r[1]['objective'])
            if best is None or ranked[0][1]['objective'] > best[1]['objective'] + 1e-8:
                best, stale = ranked[0], 0
            else:
                stale += 1
            elite = np.array([r[0] for r in ranked[:max(4, args.population // 5)]])
            mean = .5 * mean + .5 * elite.mean(axis=0)
            std = np.maximum(.025, .7 * std + .3 * elite.std(axis=0))
            history.append({'generation': gen + 1, 'best': best[1], 'stale': stale,
                            'wall_seconds': time.monotonic() - start})
            np.savez_compressed(args.output / 'checkpoint.npz', **ck,
                                best_wait_delta=best[0], wait_search_mean=mean, wait_search_std=std)
            (args.output / 'search_history.json').write_text(json.dumps(history, indent=2))
            print('GEN', gen + 1, 'SUCCESS', best[1]['successes'], '/24', 'NOMINAL',
                  int(best[1]['nominal_success']), 'STALE', stale, flush=True)
            if best[1]['successes'] == len(train) or stale >= 12:
                break
        training_dir = args.output / 'training_replay'; training_dir.mkdir()
        training_rows = list(pool.map(qualify, [(s, best[0], str(training_dir), 2., 1.)
                                               for s in [None] + train], chunksize=1))
        (args.output / 'training_replay.json').write_text(json.dumps(training_rows, indent=2))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), ck, heldout)) as pool:
        rows = list(pool.map(qualify, [(s, best[0], str(args.output), 35., 30.)
                                      for s in [None] + heldout], chunksize=1))
    rs = rows[1:]
    report = {**contract, 'training_best': best[1], 'generations_completed': len(history),
              'canonical': rows[0], 'heldout': rs, 'heldout_trials': len(rs),
              'baseline_successes': sum(r['baseline']['success'] for r in rs),
              'heldout_successes': sum(r['candidate']['success'] for r in rs)}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('HELDOUT', report['heldout_successes'], '/40 BASELINE', report['baseline_successes'], flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
