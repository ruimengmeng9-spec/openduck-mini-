"""R81: causal asymmetric early IMU feedback, not a fixed pose correction.

R80 retained zero offsets (13/40). Here only the 0.5--3.5 s correction law
changes: independent leg responses to body-frame reference tilt errors.
The combined old+new residual retains the original +/-0.18 rad cap. There
is no increased actuation budget, phase change or intermediate state reset.
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
from diagnostics.probe_getup_wait_library_r77 import TRAIN, save_trace
from diagnostics.search_getup_reference_feedback_r64 import features, residual, record_reference, rollout
from diagnostics.search_getup_support_margin_r49 import foot_support, support_quality, supported_entry
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS
from diagnostics.train_getup_prefix_pose_r73 import success

GAIN_BOUND = 2.
CAP = .18
HELDOUT = tuple(range(781000, 781040))


def checked_matrix(flat):
    values = np.asarray(flat, dtype=float)
    if values.shape != (16,) or not np.isfinite(values).all() or np.abs(values).max() > GAIN_BOUND + 1e-10:
        raise ValueError('Need sixteen finite feedback gains in [-2,2]')
    return values.reshape(8, 2)


def early_envelope(elapsed_s):
    if not np.isfinite(elapsed_s):
        raise ValueError('Elapsed time must be finite')
    return float(np.interp(elapsed_s, [.5, 1.5, 2.5, 3.5], [0., 1., 1., 0.], left=0., right=0.))


def combined_residual(error, original_gains, matrix, elapsed_s):
    original = residual(error, original_gains)
    envelope = early_envelope(elapsed_s)
    if envelope == 0.:
        return original
    extra = np.zeros(len(JOINTS))
    extra[:8] = envelope * (matrix @ np.asarray(error)[:2])
    return np.clip(original + extra, -CAP, CAP)


def feedback_rollout(sim, source, peaks, finite, targets, phases, ids, ref, gains,
                     flat, record=False):
    matrix = checked_matrix(flat)
    if not np.any(matrix):
        # Baseline is literally the original evaluator, not an approximate rewrite.
        return rollout(sim, source, peaks, finite, targets, phases, ids, ref, gains, record)
    sim.restore(source)
    sim.peaks, sim.finite = peaks.copy(), finite
    best = -np.inf
    tail = longest = run = 0
    trace = []
    for i, (target_base, phase) in enumerate(zip(targets, phases)):
        error = features(sim) - ref[i]  # Read before the control; no future observation.
        correction = combined_residual(error, gains[phase], matrix, i * DT)
        target = target_base.copy()
        target[ids] = np.clip(target[ids] + correction, sim.lower[ids], sim.upper[ids])
        sim.step_target(target)  # Unmodified substep physics and 50 Hz slew limiter.
        valid = sim.physical_valid()
        metric = sim.measure()
        support = foot_support(sim)
        best = max(best, support_quality(metric, support))
        run = run + 1 if supported_entry(metric, support) else 0
        longest = max(longest, run)
        tail = tail + 1 if metric['stable'] else 0
        if record:
            trace.append((sim.data.time, sim.data.qpos.copy(), sim.data.qvel.copy(),
                          int(phase), int(metric['stable']), support['margin_m'],
                          error.copy(), correction.copy()))
        if not valid:
            break
    score = (-100. if not valid else best + 10 * min(longest * DT, 1.) + 20 * min(tail * DT, 1.))
    return {'score': float(score), 'valid': bool(valid), 'strict_tail_s': tail * DT,
            'supported_longest_s': longest * DT, 'completed_steps': i + 1,
            'final': metric, 'final_support': support, 'peaks': sim.peaks.copy()}, trace


def run_path(hold):
    sim, cases, ck, ids = base._CTX
    targets, phases = timed_targets(sim, ck, (.4, .6, .8), hold)
    ref = record_reference(sim, *cases[None][:2], True, targets)
    gains = np.concatenate([ck['prefix_gains'], ck['terminal_gains']])
    return targets, phases, ref, gains


def objective(flat):
    sim, cases, ck, ids = base._CTX
    targets, phases, ref, gains = run_path(2.)
    nominal, _ = feedback_rollout(sim, *cases[None][:2], True, targets, phases, ids, ref, gains, flat)
    if not success(nominal, len(targets), 1.):
        return {'objective': -1000. + nominal['score'], 'successes': 0,
                'nominal_success': False, 'nominal': nominal}
    rows = []
    for seed, case in cases.items():
        if seed is None:
            continue
        row, _ = feedback_rollout(sim, *case[:2], True, targets, phases, ids, ref, gains, flat)
        rows.append({'seed': seed, 'success': success(row, len(targets), 1.), **row})
    passed = sum(r['success'] for r in rows)
    scores = np.sort([r['score'] for r in rows])
    return {'objective': float(scores[:6].mean() + .2 * scores.mean() + 14 * passed
                               - .03 * np.dot(flat, flat)), 'successes': passed,
            'nominal_success': True, 'nominal': nominal, 'cases': rows}


def qualify(job):
    seed, flat, directory, hold, required = job
    sim, cases, ck, ids = base._CTX
    source, peaks, initial, initial_hash = cases[seed]
    row = {'seed': seed, 'initial_fallen': True, 'initial': initial,
           'initial_state_sha256': initial_hash}
    targets, phases, ref, gains = run_path(hold)
    for name, params in [('baseline', np.zeros(16)), ('candidate', flat)]:
        value, trace = feedback_rollout(sim, source, peaks, True, targets, phases, ids, ref, gains, params, True)
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
    parser.add_argument('--seed', type=int, default=181)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.workers <= 6 or args.generations < 1 or args.population < 24:
        parser.error('Need fresh output and bounded search settings')
    root = Path('/data/shijinsheng/open_duck')
    scene = root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    prior = root / 'outputs/getup_early_pose_r80_left_20261001/results.json'
    report80 = json.loads(prior.read_text())
    if (report80['scene_sha256'] != digest(scene) or report80['stand_sha256'] != digest(stand)
            or report80['checkpoint_sha256'] != digest(args.checkpoint)):
        raise RuntimeError('Frozen R80 inputs changed')
    assert not set(TRAIN) & set(HELDOUT)
    assert not set(report80['heldout_seeds']) & set(HELDOUT)
    with np.load(args.checkpoint, allow_pickle=False) as z:
        ck = {key: z[key] for key in z.files}
    args.output.mkdir(parents=True)
    executed = args.output / 'executed_sources'; executed.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, executed / path.name)
    contract = {'simulation_only': True, 'hardware_readiness': False, 'full_task_completed': False,
                'method': 'early asymmetric reference-tilt feedback CEM', 'pose': 'left_side',
                'hypothesis': 'case-dependent early body error needs asymmetric sensor feedback rather than universal offsets',
                'training_seeds': list(TRAIN), 'heldout_seeds': list(HELDOUT),
                'no_midpath_state_reset': True, 'nominal_gate_required': True,
                'envelope_times_s': [.5, 1.5, 2.5, 3.5], 'leg_order': list(JOINTS[:8]),
                'inputs': ['pre-control body upvector X reference error', 'pre-control body upvector Y reference error'],
                'gain_bound': GAIN_BOUND, 'combined_old_new_residual_cap_rad': CAP,
                'training_gate_seconds': 1., 'required_strict_seconds': 30., 'hold_seconds': 35.,
                'frozen': ['all_reference_targets', 'phase_timing', 'existing_feedback_gains',
                           'head_targets', 'total_residual_cap', 'physics', 'motor_limits', 'strict_gate'],
                'source_sha256': digest(__file__), 'checkpoint_sha256': digest(args.checkpoint),
                'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
                'r80_result_sha256': digest(prior),
                'executed_source_hashes': {p.name: digest(p) for p in executed.glob('*.py')},
                'seed': args.seed, 'generation_budget': args.generations,
                'population': args.population, 'workers': args.workers}
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    rng = np.random.default_rng(args.seed)
    mean, std = np.zeros(16), np.full(16, .25)
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
                        delta = np.zeros((8, 2)); delta[joint, joint % 2] = sign * .35
                        population.append(delta.ravel())
            while len(population) < args.population:
                center = best[0] if best and len(population) % 3 == 0 else mean
                population.append(np.clip(center + rng.normal(0., std), -GAIN_BOUND, GAIN_BOUND))
            evaluated = list(pool.map(objective, population, chunksize=1))
            if gen == 0 and (not evaluated[0]['nominal_success'] or
                             evaluated[0]['successes'] != report80['training_best']['successes']):
                raise RuntimeError('Zero-gain baseline does not reproduce R80; stop fitting')
            ranked = sorted(zip(population, evaluated), key=lambda r: -r[1]['objective'])
            if best is None or ranked[0][1]['objective'] > best[1]['objective'] + 1e-8:
                best, stale = ranked[0], 0
            else:
                stale += 1
            elite = np.array([r[0] for r in ranked[:max(4, args.population // 5)]])
            mean = .5 * mean + .5 * elite.mean(axis=0)
            std = np.maximum(.06, .7 * std + .3 * elite.std(axis=0))
            history.append({'generation': gen + 1, 'best': best[1], 'stale': stale,
                            'nominal_eligible_candidates': sum(r['nominal_success'] for r in evaluated),
                            'wall_seconds': time.monotonic() - started})
            np.savez_compressed(args.output / 'checkpoint.tmp.npz', **ck,
                                r81_early_gains=best[0].reshape(8, 2), r81_search_mean=mean,
                                r81_search_std=std, r81_generation=gen + 1)
            (args.output / 'checkpoint.tmp.npz').replace(args.output / 'checkpoint.npz')
            (args.output / 'search_history.json').write_text(json.dumps(history, indent=2))
            (args.output / 'rng_state.json').write_text(json.dumps(rng.bit_generator.state))
            print('GEN', gen + 1, 'SUCCESS', best[1]['successes'], '/24', 'NOMINAL',
                  int(best[1]['nominal_success']), 'ELIGIBLE', history[-1]['nominal_eligible_candidates'],
                  'STALE', stale, flush=True)
            if best[1]['successes'] == len(TRAIN) or stale >= 12:
                break
        review = args.output / 'training_review'; review.mkdir()
        trainrows = list(pool.map(qualify, [(s, best[0], str(review), 2., 1.)
                                          for s in (None, *TRAIN)], chunksize=1))
    qualification = args.output / 'qualification'; qualification.mkdir()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=base.init_worker,
                             initargs=(str(scene), str(stand), ck, HELDOUT)) as pool:
        for row in pool.map(qualify, [(s, best[0], str(qualification), 35., 30.)
                                    for s in (None, *HELDOUT)], chunksize=1):
            rows.append(row)
            (args.output / 'qualification_progress.json').write_text(json.dumps(rows, indent=2))
            print('HELDOUT', row['seed'], 'BASE', int(row['baseline']['success']),
                  'CANDIDATE', int(row['candidate']['success']), flush=True)
    rs = rows[1:]
    report = {**contract, 'generations_completed': len(history), 'training_best': best[1],
              'training_review': trainrows, 'canonical': rows[0], 'heldout': rs,
              'heldout_trials': len(rs), 'baseline_successes': sum(r['baseline']['success'] for r in rs),
              'heldout_successes': sum(r['candidate']['success'] for r in rs),
              'paired_rescues': [r['seed'] for r in rs if r['candidate']['success'] and not r['baseline']['success']],
              'paired_regressions': [r['seed'] for r in rs if r['baseline']['success'] and not r['candidate']['success']]}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('QUALIFIED', report['heldout_successes'], '/40 BASELINE', report['baseline_successes'], flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__': main()
