"""R82: bounded causal sensor-clock alignment, not another fixed time warp.

Only the phase-0 reference cursor is advanced at a sensor-dependent rate.
The cursor never rewinds, skips a phase, or resets physical state. Fixed
reference targets, residual cap and all StrictSim physics remain unchanged.
Nominal and training trials precede any fresh paired qualification.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np

from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.getup_independent_native import DT, digest
from diagnostics.probe_getup_terminal_timing_r72 import timed_targets
from diagnostics.probe_getup_wait_library_r77 import TRAIN, save_trace
from diagnostics.search_getup_reference_feedback_r64 import features, residual, record_reference, rollout
from diagnostics.search_getup_support_margin_r49 import foot_support, support_quality, supported_entry

HELDOUT = tuple(range(782000, 782040))
RATE_MIN, RATE_MAX = .25, 1.75


def interpolate(array, cursor):
    if not np.isfinite(cursor) or not 0 <= cursor <= len(array) - 1:
        raise ValueError('Reference cursor outside saved path')
    lower = int(np.floor(cursor)); upper = min(lower + 1, len(array) - 1)
    alpha = cursor - lower
    return (1. - alpha) * array[lower] + alpha * array[upper]


def clock_rate(observed, reference, cursor, last, gain, radius):
    if not np.isfinite(gain) or not 0 <= gain <= .6 or radius not in (10, 20):
        raise ValueError('Unsupported bounded clock policy')
    if gain == 0 or cursor < 25 or cursor >= last:
        return 1., cursor
    current = interpolate(reference, cursor)
    if np.linalg.norm(np.asarray(observed) - current) < 1e-10:
        return 1., cursor
    indices = np.arange(max(0, int(np.floor(cursor)) - radius),
                        min(last, int(np.ceil(cursor)) + radius) + 1)
    weights = np.array([1., 1., .5, .5])
    distance = np.sum(((reference[indices] - observed) * weights) ** 2, axis=1)
    # Penalize large matches in ambiguous parts of the reference, never use
    # a future observation. Future stored reference values are not observations.
    distance += 1e-5 * (indices - cursor) ** 2
    estimate = float(indices[np.argmin(distance)])
    return float(np.clip(1. + gain * (estimate - cursor), RATE_MIN, RATE_MAX)), estimate


def phase0_end(phases):
    indices = np.flatnonzero(phases == 0)
    if len(indices) < 26 or not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError('Need a contiguous initial phase 0')
    return int(indices[-1])


def advance(cursor, last, rate):
    if not np.isfinite(rate) or not RATE_MIN <= rate <= RATE_MAX:
        raise ValueError('Invalid reference clock rate')
    if cursor >= last:
        return cursor + 1.
    # Always execute the final phase-0 target before phase 1. Thereafter
    # the untouched integer cursor executes every remaining stored command.
    return min(float(last), cursor + rate)


def passed(row, required):
    return bool(row['valid'] and row['path_completed']
                and row['strict_tail_s'] >= required - 1e-8)


def clock_rollout(sim, source, peaks, targets, phases, ids, ref, gains, gain, radius):
    last = phase0_end(phases)
    if gain == 0:
        row, trace = rollout(sim, source, peaks, True, targets, phases, ids, ref, gains, True)
        row['path_completed'] = row['completed_steps'] == len(targets)
        cursor = np.arange(len(trace), dtype=float)
        return row, trace, {'cursor': cursor, 'clock_rate': np.ones(len(trace)),
                            'matched_reference_index': cursor.copy()}
    sim.restore(source)  # The sole restore is the initial fallen trial state.
    sim.peaks, sim.finite = peaks.copy(), True
    cursor = 0.; best = -np.inf; tail = longest = run = 0
    trace, cursors, rates, matches = [], [], [], []
    budget = int(np.ceil((last + 1) / RATE_MIN)) + len(targets) - last + 10
    for step in range(budget):
        observed = features(sim)  # Sample before the command.
        phase = int(phases[int(cursor)])
        target_base = interpolate(targets, cursor)
        error = observed - interpolate(ref, cursor)
        rate, estimate = clock_rate(observed, ref, cursor, last, gain, radius) if cursor <= last else (1., cursor)
        correction = residual(error, gains[phase])
        target = target_base.copy()
        target[ids] = np.clip(target[ids] + correction, sim.lower[ids], sim.upper[ids])
        sim.step_target(target)
        valid = sim.physical_valid(); metric = sim.measure(); support = foot_support(sim)
        best = max(best, support_quality(metric, support))
        run = run + 1 if supported_entry(metric, support) else 0
        longest = max(longest, run); tail = tail + 1 if metric['stable'] else 0
        trace.append((sim.data.time, sim.data.qpos.copy(), sim.data.qvel.copy(), phase,
                      int(metric['stable']), support['margin_m'], error.copy(), correction.copy()))
        cursors.append(cursor); rates.append(rate); matches.append(estimate)
        completed = cursor >= len(targets) - 1
        if not valid or completed: break
        cursor = advance(cursor, last, rate)
    else:
        raise RuntimeError('Monotonic bounded cursor exceeded its budget')
    score = -100. if not valid else best + 10 * min(longest * DT, 1.) + 20 * min(tail * DT, 1.)
    row = {'score': float(score), 'valid': bool(valid), 'path_completed': bool(completed),
           'strict_tail_s': tail * DT, 'supported_longest_s': longest * DT,
           'completed_steps': step + 1, 'final': metric, 'final_support': support,
           'peaks': sim.peaks.copy(), 'phase0_controls': sum(c <= last for c in cursors)}
    return row, trace, {'cursor': np.asarray(cursors), 'clock_rate': np.asarray(rates),
                        'matched_reference_index': np.asarray(matches)}


def run_path(hold):
    sim, cases, ck, ids = base._CTX
    targets, phases = timed_targets(sim, ck, (.4, .6, .8), hold)
    ref = record_reference(sim, *cases[None][:2], True, targets)
    gains = np.concatenate([ck['prefix_gains'], ck['terminal_gains']])
    return targets, phases, ref, gains


def evaluate(job):
    gain, radius, directory, hold, required = job
    sim, cases, ck, ids = base._CTX
    targets, phases, ref, gains = run_path(hold)
    dest = Path(directory); dest.mkdir(exist_ok=False)
    result = {'gain': gain, 'radius_controls': radius, 'nominal_success': False,
              'successes': 0, 'cases': [], 'target_steps': len(targets)}
    for seed, case in cases.items():
        row, trace, clock = clock_rollout(sim, *case[:2], targets, phases, ids, ref, gains, gain, radius)
        row = {'seed': seed, 'initial': case[2], 'initial_state_sha256': case[3],
               'success': passed(row, required), **row}
        save_trace(dest / f'case_{seed}.npz', trace)
        np.savez_compressed(dest / f'clock_{seed}.npz', **clock)
        if seed is None:
            result['nominal'] = row; result['nominal_success'] = row['success']
            if not row['success']: break
        else:
            result['cases'].append(row); result['successes'] += int(row['success'])
    (dest / 'summary.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.workers <= 6:
        parser.error('Need a new output and 1--6 workers')
    root = Path('/data/shijinsheng/open_duck')
    scene = root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    prior = root / 'outputs/getup_early_feedback_r81_left_20261001/results.json'
    r81 = json.loads(prior.read_text())
    if any(r81[key] != digest(path) for key, path in
           [('checkpoint_sha256', args.checkpoint), ('scene_sha256', scene), ('stand_sha256', stand)]):
        raise RuntimeError('Frozen inputs differ from R81')
    assert not set(TRAIN) & set(HELDOUT)
    assert not set(r81['heldout_seeds']) & set(HELDOUT)
    with np.load(args.checkpoint, allow_pickle=False) as z: ck = {k: z[k] for k in z.files}
    args.output.mkdir(); executed = args.output / 'executed_sources'; executed.mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copy2(path, executed / path.name)
    grid = [(0., 10)] + [(gain, radius) for gain in (.05, .15, .3, .6) for radius in (10, 20)]
    contract = {'simulation_only': True, 'hardware_readiness': False, 'full_task_completed': False,
                'hypothesis': 'case-dependent body progress mismatches the fixed phase-0 reference clock',
                'method': 'causal IMU nearest-reference monotonic phase clock grid',
                'training_seeds': list(TRAIN), 'heldout_seeds': list(HELDOUT), 'grid': grid,
                'reference_rate_bounds': [RATE_MIN, RATE_MAX], 'minimum_adaptive_index': 25,
                'matching_weights': [1., 1., .5, .5], 'index_penalty': 1e-5,
                'no_midpath_state_reset': True, 'required_strict_seconds': 30.,
                'qualification_hold_seconds': 35., 'training_gate_seconds': 1.,
                'qualification_only_after_training_improvement': True,
                'frozen': ['reference_target_geometry', 'existing_feedback', 'residual_cap',
                           'postphase0_timing', 'collision', 'torque', 'joint_limits', 'slew', 'strict_gate'],
                'source_sha256': digest(__file__), 'checkpoint_sha256': digest(args.checkpoint),
                'scene_sha256': digest(scene), 'stand_sha256': digest(stand), 'r81_result_sha256': digest(prior),
                'executed_source_hashes': {p.name: digest(p) for p in executed.glob('*.py')}}
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=base.init_worker,
                             initargs=(str(scene), str(stand), ck, TRAIN)) as pool:
        jobs = [(g, rad, str(args.output / f'grid_{i:02d}'), 2., 1.) for i, (g, rad) in enumerate(grid)]
        for row in pool.map(evaluate, jobs, chunksize=1):
            rows.append(row)
            (args.output / 'grid_progress.json').write_text(json.dumps(rows, indent=2))
            print('CLOCK', row['gain'], row['radius_controls'], 'NOMINAL', row['nominal_success'],
                  'SUCCESS', row['successes'], '/24', flush=True)
    baseline = rows[0]
    if not baseline['nominal_success'] or baseline['successes'] != r81['training_best']['successes']:
        raise RuntimeError('Identity clock does not reproduce R81')
    eligible = [r for r in rows if r['nominal_success']]
    best = max(eligible, key=lambda r: (r['successes'], sum(c['score'] for c in r['cases']), -r['gain']))
    report = {**contract, 'training_grid': rows, 'selected': best,
              'qualification_run': False, 'qualification_pending': False}
    np.savez_compressed(args.output / 'selected.npz', **ck, r82_gain=best['gain'], r82_radius=best['radius_controls'])
    if best['successes'] > baseline['successes']:
        report['qualification_run'] = True
        with ProcessPoolExecutor(max_workers=2, initializer=base.init_worker,
                                 initargs=(str(scene), str(stand), ck, HELDOUT)) as pool:
            qualified = list(pool.map(evaluate, [(0., 10, str(args.output / 'qualification_baseline'), 35., 30.),
                                                (best['gain'], best['radius_controls'],
                                                 str(args.output / 'qualification_candidate'), 35., 30.)]))
        report['qualification'] = qualified
        a, b = qualified
        report['baseline_successes'], report['heldout_successes'] = a['successes'], b['successes']
        report['paired_rescues'] = [y['seed'] for x,y in zip(a['cases'], b['cases']) if y['success'] and not x['success']]
        report['paired_regressions'] = [y['seed'] for x,y in zip(a['cases'], b['cases']) if x['success'] and not y['success']]
        print('QUALIFIED', b['successes'], '/40 BASELINE', a['successes'], flush=True)
    else:
        report['qualification_skipped_reason'] = 'No training-count improvement; fresh qualification seeds not simulated'
        print('NO TRAINING IMPROVEMENT; independent seeds remain unused', flush=True)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__': main()
