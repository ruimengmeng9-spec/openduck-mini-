"""R79-informed early recovery correction, with unchanged physical gates.

Only leg targets between 0.5 and 3.5 s change, before the observed early
divergence. R73 changed 4.16--6.16 s instead. Causal execution still uses the
original feedback, 50 Hz limiter and substep audit. Each trial restores only
its own actually-fallen initial state, never an intermediate reference state.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time

import numpy as np

from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.getup_independent_native import DT, digest
from diagnostics.probe_getup_terminal_timing_r72 import timed_targets
from diagnostics.probe_getup_wait_library_r77 import save_trace, TRAIN
from diagnostics.search_getup_reference_feedback_r64 import record_reference, rollout
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS
from diagnostics.train_getup_prefix_pose_r73 import success

BOUND = .20
KNOT_TIMES = np.array([.5, 1.5, 2.5, 3.5])
HELDOUT = tuple(range(780000, 780040))
LEG_NAMES = JOINTS[:8]


def edited_early(sim, ck, flat):
    flat = np.asarray(flat, dtype=float)
    if flat.shape != (16,) or not np.isfinite(flat).all() or np.abs(flat).max() > BOUND + 1e-10:
        raise ValueError('Need sixteen finite early-pose offsets within the search bound')
    phases = ck['prefix_phases']
    ix = np.flatnonzero(phases == 0)
    if len(ix) < round(KNOT_TIMES[-1] / DT) + 1 or not np.array_equal(ix, np.arange(len(ix))):
        raise ValueError('Need an initial contiguous recovery phase longer than 3.5 s')
    target = ck['prefix_targets'].copy()
    # The target issued at control index i is pre-step, at elapsed i*DT.
    t = np.arange(len(ix)) * DT
    knots = np.vstack([np.zeros(8), flat.reshape(2, 8), np.zeros(8)])
    delta = np.stack([np.interp(t, KNOT_TIMES, knots[:, j], left=0., right=0.)
                      for j in range(8)], axis=1)
    ids = np.array([sim.model.actuator(name).id for name in LEG_NAMES])
    target[np.ix_(ix, ids)] = np.clip(target[np.ix_(ix, ids)] + delta,
                                     sim.lower[ids], sim.upper[ids])
    return {**ck, 'prefix_targets': target}


def run_path(flat, hold):
    sim, cases, ck, ids = base._CTX
    changed = edited_early(sim, ck, flat)
    targets, phases = timed_targets(sim, changed, (.4, .6, .8), hold)
    ref = record_reference(sim, *cases[None][:2], True, targets)
    gains = np.concatenate([ck['prefix_gains'], ck['terminal_gains']])
    return targets, phases, ref, gains


def objective(flat):
    sim, cases, ck, ids = base._CTX
    try:
        targets, phases, ref, gains = run_path(flat, 2.)
    except RuntimeError as exc:
        return {'objective': -10000., 'successes': 0, 'nominal_success': False,
                'reference_rejected': str(exc)}
    nominal, _ = rollout(sim, *cases[None][:2], True, targets, phases, ids, ref, gains)
    if not success(nominal, len(targets), 1.):
        return {'objective': -1000. + nominal['score'], 'successes': 0,
                'nominal_success': False, 'nominal': nominal}
    rows = []
    for seed, case in cases.items():
        if seed is None:
            continue
        row, _ = rollout(sim, *case[:2], True, targets, phases, ids, ref, gains)
        rows.append({'seed': seed, 'success': success(row, len(targets), 1.), **row})
    passed = sum(r['success'] for r in rows)
    scores = np.sort([r['score'] for r in rows])
    return {'objective': float(scores[:6].mean() + .2 * scores.mean() + 14 * passed
                               - .1 * np.dot(flat, flat)),
            'successes': passed, 'nominal_success': True, 'nominal': nominal, 'cases': rows}


def qualify(job):
    seed, flat, directory, hold, required = job
    sim, cases, ck, ids = base._CTX
    source, peaks, initial, initial_hash = cases[seed]
    row = {'seed': seed, 'initial_fallen': True, 'initial': initial,
           'initial_state_sha256': initial_hash}
    for name, params in [('baseline', np.zeros(16)), ('candidate', flat)]:
        targets, phases, ref, gains = run_path(params, hold)
        value, trace = rollout(sim, source, peaks, True, targets, phases, ids, ref, gains, True)
        value['success'] = success(value, len(targets), required)
        row[name] = value
        save_trace(Path(directory) / f'{name}_{seed}.npz', trace)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--generations', type=int, default=24)
    parser.add_argument('--population', type=int, default=24)
    parser.add_argument('--seed', type=int, default=180)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.workers <= 6 or args.generations < 1 or args.population < 24:
        parser.error('Need fresh output, 1--6 workers, positive generations and population >=24')
    root = Path('/data/shijinsheng/open_duck')
    scene = root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    diagnosis = root / 'outputs/getup_failure_audit_r79_20261001/results.json'
    library = root / 'outputs/getup_wait_library_r77_20261001/results.json'
    evidence = json.loads(diagnosis.read_text())
    lib = json.loads(library.read_text())
    if (evidence['scene_sha256'] != digest(scene) or lib['checkpoint_sha256'] != digest(args.checkpoint)
            or evidence['library_result_sha256'] != digest(library)):
        raise RuntimeError('R79 diagnosis and frozen baseline contract disagree')
    with np.load(args.checkpoint, allow_pickle=False) as z:
        ck = {key: z[key] for key in z.files}
    if not np.array_equal(ck['best_pose_delta'], np.zeros_like(ck['best_pose_delta'])):
        raise RuntimeError('Need frozen identity R73 checkpoint')
    assert not set(TRAIN) & set(HELDOUT)
    args.output.mkdir(parents=True)
    executed = args.output / 'executed_sources'
    executed.mkdir()
    # Capture all transitive project diagnostics, not just the entry point.
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, executed / path.name)
    contract = {'simulation_only': True, 'hardware_readiness': False, 'full_task_completed': False,
                'method': 'R79-informed early leg-pose two-knot CEM', 'pose': 'left_side',
                'hypothesis': 'R73 corrected too late: intervene before 1.94--2.50 s divergence',
                'training_seeds': list(TRAIN), 'heldout_seeds': list(HELDOUT),
                'no_midpath_state_reset': True, 'nominal_gate_required': True,
                'knot_times_s': KNOT_TIMES.tolist(), 'learned_knots': 2, 'leg_order': list(LEG_NAMES),
                'delta_bound_rad': BOUND, 'training_gate_seconds': 1.,
                'required_strict_seconds': 30., 'hold_seconds': 35.,
                'frozen': ['targets_outside_early_window', 'head_targets', 'wait_duration',
                           'phase_timing', 'feedback_gains', 'physics', 'motor_limits', 'strict_gate'],
                'source_sha256': digest(__file__), 'checkpoint_sha256': digest(args.checkpoint),
                'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
                'r79_result_sha256': digest(diagnosis), 'r77_result_sha256': digest(library),
                'executed_source_hashes': {p.name: digest(p) for p in executed.glob('*.py')},
                'seed': args.seed, 'generation_budget': args.generations,
                'population': args.population, 'workers': args.workers}
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    rng = np.random.default_rng(args.seed)
    mean, std = np.zeros(16), np.full(16, .045)
    best, history, stale = None, [], 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=base.init_worker,
                             initargs=(str(scene), str(stand), ck, TRAIN)) as pool:
        for gen in range(args.generations):
            population = [np.zeros(16), mean.copy()]
            if best is not None:
                population.append(best[0].copy())
            if gen == 0:
                for joint in range(8):
                    for sign in (-1., 1.):
                        delta = np.zeros((2, 8)); delta[:, joint] = sign * .035
                        population.append(delta.ravel())
            while len(population) < args.population:
                center = best[0] if best and len(population) % 3 == 0 else mean
                population.append(np.clip(center + rng.normal(0., std), -BOUND, BOUND))
            evaluated = list(pool.map(objective, population, chunksize=1))
            if gen == 0 and (not evaluated[0]['nominal_success'] or
                             evaluated[0]['successes'] != lib['baseline_successes']):
                raise RuntimeError('Zero-delta baseline does not reproduce R77; stop training')
            ranked = sorted(zip(population, evaluated), key=lambda r: -r[1]['objective'])
            if best is None or ranked[0][1]['objective'] > best[1]['objective'] + 1e-8:
                best, stale = ranked[0], 0
            else:
                stale += 1
            elite = np.array([r[0] for r in ranked[:max(4, args.population // 5)]])
            mean = .5 * mean + .5 * elite.mean(axis=0)
            std = np.maximum(.015, .7 * std + .3 * elite.std(axis=0))
            history.append({'generation': gen + 1, 'best': best[1], 'stale': stale,
                            'nominal_eligible_candidates': sum(r['nominal_success'] for r in evaluated),
                            'wall_seconds': time.monotonic() - started})
            checkpoint = {**ck, 'r80_early_delta': best[0], 'r80_search_mean': mean,
                          'r80_search_std': std, 'r80_generation': gen + 1}
            np.savez_compressed(args.output / 'checkpoint.tmp.npz', **checkpoint)
            (args.output / 'checkpoint.tmp.npz').replace(args.output / 'checkpoint.npz')
            (args.output / 'search_history.json').write_text(json.dumps(history, indent=2))
            (args.output / 'rng_state.json').write_text(json.dumps(rng.bit_generator.state))
            print('GEN', gen + 1, 'SUCCESS', best[1]['successes'], '/24', 'NOMINAL',
                  int(best[1]['nominal_success']), 'ELIGIBLE', history[-1]['nominal_eligible_candidates'],
                  'STALE', stale, flush=True)
            if best[1]['successes'] == len(TRAIN) or stale >= 12:
                break
        review = args.output / 'training_review'; review.mkdir()
        training_rows = list(pool.map(qualify, [(seed, best[0], str(review), 2., 1.)
                                              for seed in (None, *TRAIN)], chunksize=1))
    qualification = args.output / 'qualification'; qualification.mkdir()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=base.init_worker,
                             initargs=(str(scene), str(stand), ck, HELDOUT)) as pool:
        for row in pool.map(qualify, [(seed, best[0], str(qualification), 35., 30.)
                                     for seed in (None, *HELDOUT)], chunksize=1):
            rows.append(row)
            (args.output / 'qualification_progress.json').write_text(json.dumps(rows, indent=2))
            print('HELDOUT', row['seed'], 'BASE', int(row['baseline']['success']),
                  'CANDIDATE', int(row['candidate']['success']), flush=True)
    rs = rows[1:]
    report = {**contract, 'generations_completed': len(history), 'training_best': best[1],
              'training_review': training_rows, 'canonical': rows[0], 'heldout': rs,
              'heldout_trials': len(rs), 'baseline_successes': sum(r['baseline']['success'] for r in rs),
              'heldout_successes': sum(r['candidate']['success'] for r in rs),
              'paired_rescues': [r['seed'] for r in rs if r['candidate']['success'] and not r['baseline']['success']],
              'paired_regressions': [r['seed'] for r in rs if r['baseline']['success'] and not r['candidate']['success']]}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('QUALIFIED', report['heldout_successes'], '/40 BASELINE', report['baseline_successes'], flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
