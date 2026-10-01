"""Optimize the pre-wait contact pose, not terminal timing or motor limits.

R71 found small encoder errors but most failures already tipped during the
home wait; R72 terminal dwell changes did not rescue them. Learn four smooth
leg-target correction knots over the final two seconds of recovery. Freeze
everything outside that window, including feedback gains and terminal path.
Full actual-fall rollouts, nominal preservation, and fresh 30 s qualification.
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
from diagnostics.validate_getup_fullpath_r27 import StrictSim

_CTX = None
LEG_NAMES = JOINTS[:8]


def edited_prefix(sim, ck, flat):
    prefix = ck['prefix_targets'].copy()
    indices = np.flatnonzero(ck['prefix_phases'] == 0)
    n = min(round(2. / DT), len(indices))
    indices = indices[-n:]
    knots = np.concatenate([np.zeros((1, 8)), np.asarray(flat).reshape(4, 8)])
    weights = np.linspace(0., 4., n + 1)[1:]
    correction = np.stack([np.interp(weights, np.arange(5), knots[:, j])
                           for j in range(8)], axis=1)
    ids = np.array([sim.model.actuator(name).id for name in LEG_NAMES])
    prefix[np.ix_(indices, ids)] = np.clip(prefix[np.ix_(indices, ids)] + correction,
                                         sim.lower[ids], sim.upper[ids])
    return {**ck, 'prefix_targets': prefix}


def init_worker(scene, stand, ck, seeds):
    global _CTX
    sim = StrictSim(scene, stand)
    cases = []
    for seed in [None] + list(seeds):
        sim.prepare('left_side', 0 if seed is None else seed, seed is not None)
        m = sim.measure()
        if not (m['up_z'] < .5 and m['torso_contact']):
            raise RuntimeError('Not actually fallen')
        sim.clear_audit()
        cases.append((sim.snapshot(), sim.peaks.copy()))
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, cases, ck, ids


def run_path(flat, hold):
    sim, cases, ck, ids = _CTX
    changed = edited_prefix(sim, ck, flat)
    targets, phases = timed_targets(sim, changed, (.4, .6, .8), hold)
    ref = record_reference(sim, *cases[0], True, targets)
    gains = np.concatenate([ck['prefix_gains'], ck['terminal_gains']])
    return targets, phases, ref, gains


def success(row, count, seconds):
    return bool(row['valid'] and row['completed_steps'] == count
                and row['strict_tail_s'] >= seconds - 1e-8)


def objective(flat):
    sim, cases, ck, ids = _CTX
    try:
        targets, phases, ref, gains = run_path(flat, 2.)
    except RuntimeError:
        return {'objective': -10000., 'successes': 0, 'nominal_success': False}
    nominal, _ = rollout(sim, *cases[0], True, targets, phases, ids, ref, gains)
    if not success(nominal, len(targets), 1.):
        return {'objective': -1000. + nominal['score'], 'successes': 0,
                'nominal_success': False, 'nominal': nominal}
    rows = [rollout(sim, state, peaks, True, targets, phases, ids, ref, gains)[0]
            for state, peaks in cases[1:]]
    passed = sum(success(r, len(targets), 1.) for r in rows)
    scores = np.sort([r['score'] for r in rows])
    value = float(scores[:6].mean() + .2 * scores.mean()
                  + 14 * passed - .1 * np.dot(flat, flat))
    return {'objective': value, 'successes': passed, 'nominal_success': True,
            'nominal': nominal, 'cases': rows}


def qualify(job):
    seed, flat, directory = job
    sim, cases, ck, ids = _CTX
    row = {'seed': seed, 'initial_fallen': True}
    index = 0 if seed is None else 1 + seed - 776000
    for name, params in [('baseline', np.zeros(32)), ('candidate', flat)]:
        targets, phases, ref, gains = run_path(params, 35.)
        value, trace = rollout(sim, *cases[index], True, targets, phases, ids, ref,
                               gains, name == 'candidate')
        value['success'] = success(value, len(targets), 30.)
        row[name] = value
        if name == 'candidate':
            np.savez_compressed(Path(directory) / f'qualification_{seed}.npz',
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
    p.add_argument('--generations', type=int, default=40)
    p.add_argument('--population', type=int, default=24)
    p.add_argument('--seed', type=int, default=173)
    args = p.parse_args()
    if args.output.exists() or args.workers < 1 or args.generations < 1 or args.population < 8:
        p.error('Need fresh output and valid search settings')
    root = Path('/data/shijinsheng/open_duck')
    scene = root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    with np.load(args.checkpoint, allow_pickle=False) as z:
        ck = {k: z[k] for k in z.files}
    train = list(range(769000, 769008)) + list(range(773000, 773016))
    heldout = list(range(776000, 776040))
    assert not set(train) & set(heldout)
    args.output.mkdir(parents=True)
    contract = {'simulation_only': True, 'hardware_readiness': False,
                'full_task_completed': False, 'no_midpath_state_reset': True,
                'method': 'whole-fall pre-wait leg-pose knot CEM', 'pose': 'left_side',
                'training_seeds': train, 'heldout_seeds': heldout,
                'learned_window_s': 2., 'learned_knots': 4, 'delta_bound_rad': .35,
                'frozen': ['feedback_gains', 'wait_duration', 'terminal_path', 'physics', 'strict_gate'],
                'required_strict_seconds': 30., 'hold_seconds': 35.,
                'source_sha256': digest(__file__), 'scene_sha256': digest(scene),
                'stand_sha256': digest(stand), 'checkpoint_sha256': digest(args.checkpoint),
                'dependency_hashes': {name: digest(Path(__file__).parent / name) for name in
                    ['probe_getup_terminal_timing_r72.py', 'search_getup_reference_feedback_r64.py',
                     'validate_getup_fullpath_r27.py', 'train_getup_fullpath_r27.py']},
                'seed': args.seed, 'generation_budget': args.generations, 'population': args.population}
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    rng = np.random.default_rng(args.seed)
    mean, std = np.zeros(32), np.full(32, .055)
    history, best, stale = [], None, 0
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), ck, train)) as pool:
        for gen in range(args.generations):
            population = [np.zeros(32), mean.copy()]
            if best is not None:
                population.append(best[0].copy())
            while len(population) < args.population:
                center = best[0] if best and len(population) % 3 == 0 else mean
                candidate = center + rng.normal(0., std)
                population.append(np.clip(candidate, -.35, .35))
            ranked = sorted(zip(population, pool.map(objective, population)),
                            key=lambda r: -r[1]['objective'])
            if best is None or ranked[0][1]['objective'] > best[1]['objective'] + 1e-8:
                best, stale = ranked[0], 0
            else:
                stale += 1
            elite = np.array([r[0] for r in ranked[:max(4, args.population // 5)]])
            mean = .5 * mean + .5 * elite.mean(axis=0)
            std = np.maximum(.025, .7 * std + .3 * elite.std(axis=0))
            history.append({'generation': gen + 1, 'best': best[1],
                            'stale': stale, 'wall_seconds': time.monotonic() - start})
            np.savez_compressed(args.output / 'checkpoint.npz', **ck,
                                best_pose_delta=best[0].reshape(4, 8), search_mean=mean, search_std=std)
            (args.output / 'search_history.json').write_text(json.dumps(history, indent=2))
            print('GEN', gen + 1, 'SUCCESS', best[1]['successes'], '/24', 'NOMINAL',
                  int(best[1]['nominal_success']), 'STALE', stale, flush=True)
            if best[1]['successes'] == len(train) or stale >= 15:
                break
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), ck, heldout)) as pool:
        rows = list(pool.map(qualify, [(seed, best[0], str(args.output))
                                      for seed in [None] + heldout], chunksize=1))
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
