"""Search a small IMU-feedback residual around the successful R49 get-up.

R57 tests whether one open-loop trajectory can tolerate transition-state
perturbations.  This experiment keeps the same physically audited trajectory
and adds only eight structured balance gains driven by body tilt and angular
velocity.  It is still simulation-only and never changes model physics,
actuator limits, target slew, or the strict success gate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_robust_transition_r57 import perturb_snapshot
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, DURATIONS, capture
from diagnostics.search_getup_support_margin_r49 import (
    HOME_HOLD_S, foot_support, support_quality, supported_entry)
from diagnostics.validate_getup_fullpath_r27 import StrictSim


_CTX = None


def feedback_residual(sim, gains):
    """Return structured bilateral corrections in actuator radians."""
    up = sim.sensor('upvector')
    gyro = sim.sensor('gyro')
    pitch = float(up[0])
    roll = float(up[1])
    pitch_rate = float(gyro[1]) * .15
    roll_rate = float(gyro[0]) * .15
    pitch_terms = np.array([
        gains[0] * pitch + gains[3] * pitch_rate,
        gains[1] * pitch + gains[4] * pitch_rate,
        gains[2] * pitch + gains[5] * pitch_rate,
    ])
    roll_term = gains[6] * roll + gains[7] * roll_rate
    correction = np.zeros(len(JOINTS))
    # Physical sagittal symmetry uses opposite hip-pitch signs but equal knee
    # and ankle signs in this model.  Hip roll is antisymmetric.
    correction[[1, 5]] = [pitch_terms[0], -pitch_terms[0]]
    correction[[2, 6]] = pitch_terms[1]
    correction[[3, 7]] = pitch_terms[2]
    correction[[0, 4]] = [roll_term, -roll_term]
    return np.clip(correction, -.25, .25)


def evaluate_feedback(sim, snapshot, peaks, finite, offsets, ids, gains,
                      record=False):
    sim.restore(snapshot)
    sim.peaks, sim.finite = peaks.copy(), finite
    best = None
    supports, strict, trace = [], [], []
    valid = True
    phases = [(f'phase_{i}', seconds, row)
              for i, (seconds, row) in enumerate(zip(DURATIONS, offsets))]
    phases.append(('home_hold', HOME_HOLD_S, None))
    for phase, seconds, row in phases:
        base = sim.home.copy()
        if row is not None:
            base[ids] = np.clip(sim.home[ids] + row,
                                sim.lower[ids], sim.upper[ids])
        for _ in range(round(seconds / DT)):
            residual = feedback_residual(sim, gains)
            target = base.copy()
            target[ids] = np.clip(target[ids] + residual,
                                  sim.lower[ids], sim.upper[ids])
            sim.step_target(target)
            m = sim.measure()
            valid = sim.physical_valid()
            if not valid:
                break
            support = foot_support(sim)
            quality = support_quality(m, support)
            if best is None or quality > best['quality']:
                best = {'quality': quality, 'time_s': float(sim.data.time),
                        'metric': m, 'support': support, 'phase': phase}
            supports.append(supported_entry(m, support))
            strict.append(bool(m['stable']))
            if record:
                trace.append((float(sim.data.time), sim.data.qpos.copy(),
                              sim.data.qvel.copy(), support['margin_m'],
                              int(supports[-1]), int(strict[-1]), phase,
                              residual.copy()))
        if not valid:
            break
    if not valid:
        return {'score': -100., 'valid': False,
                'supported_longest_s': 0., 'strict_tail_s': 0.,
                'final': m, 'peaks': sim.peaks.copy()}, trace
    longest = current = 0
    for passed in supports:
        current = current + 1 if passed else 0
        longest = max(longest, current)
    tail = 0
    for passed in reversed(strict):
        if not passed:
            break
        tail += 1
    final_support = foot_support(sim)
    score = (best['quality'] + 10 * min(longest * DT, 1.)
             + 20 * min(tail * DT, 1.))
    return {'score': score, 'valid': True,
            'supported_longest_s': longest * DT,
            'strict_tail_s': tail * DT, 'best': best, 'final': m,
            'final_support': final_support, 'peaks': sim.peaks.copy()}, trace


def init_worker(scene, stand, snapshot, peaks, finite, offsets, seeds):
    global _CTX
    sim = StrictSim(scene, stand)
    cases = [perturb_snapshot(sim, snapshot, seed) for seed in seeds]
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, cases, peaks, finite, offsets, ids


def evaluate_candidate(gains):
    sim, cases, peaks, finite, offsets, ids = _CTX
    rows = [evaluate_feedback(sim, state, peaks, finite, offsets, ids, gains)[0]
            for state in cases]
    scores = np.array([row['score'] for row in rows])
    successes = sum(row['strict_tail_s'] >= 1. for row in rows)
    valid = sum(row['valid'] for row in rows)
    objective = float(scores.min() + .25 * scores.mean()
                      + 15. * successes - .05 * np.dot(gains, gains))
    return {'robust_score': objective, 'successes': successes,
            'valid_cases': valid, 'case_count': len(rows), 'cases': rows}


def evaluate_set(sim, source, peaks, finite, offsets, ids, gains, seeds):
    rows = []
    for seed in seeds:
        state = perturb_snapshot(sim, source, seed)
        result, _ = evaluate_feedback(sim, state, peaks, finite, offsets,
                                      ids, gains)
        rows.append({'seed': seed, **result})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--r49-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=64)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--training-cases', type=int, default=8)
    p.add_argument('--seed', type=int, default=158)
    args = p.parse_args()
    if (args.output.exists() or args.generations < 1 or args.population < 8
            or args.training_cases < 2 or args.workers < 1):
        p.error('Need new output and positive search settings')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    offsets = np.load(args.r49_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, captured = capture(sim, reference, 0)
    snapshot, peaks, finite = (captured.pop('snapshot'),
                               captured.pop('peaks'), captured.pop('finite'))
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    train_seeds = [759000 + i for i in range(args.training_cases)]
    heldout_seeds = [760000 + i for i in range(20)]
    baseline = evaluate_set(sim, snapshot, peaks, finite, offsets, ids,
                            np.zeros(8), heldout_seeds)
    rng = np.random.default_rng(args.seed)
    mean, std = np.zeros(8), np.full(8, .35)
    best, history = None, []
    args.output.mkdir(parents=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), snapshot, peaks,
                                       finite, offsets, train_seeds)) as pool:
        for generation in range(args.generations):
            population = [mean.copy(), np.zeros(8)]
            if best is not None:
                population.append(best[0].copy())
            while len(population) < args.population:
                base = best[0] if best is not None and len(population) % 3 == 0 else mean
                population.append(np.clip(base + rng.normal(0., std), -2., 2.))
            results = list(pool.map(evaluate_candidate, population, chunksize=1))
            ranked = sorted(zip(population, results),
                            key=lambda pair: -pair[1]['robust_score'])
            if best is None or ranked[0][1]['robust_score'] > best[1]['robust_score']:
                best = ranked[0]
            elites = np.stack([x for x, _ in ranked[:max(4, args.population // 5)]])
            mean = .5 * mean + .5 * elites.mean(axis=0)
            std = np.maximum(.05, .7 * std + .3 * elites.std(axis=0))
            row = {'generation': generation + 1,
                   'generation_best': ranked[0][1], 'overall_best': best[1],
                   'wall_seconds': time.monotonic() - started}
            history.append(row)
            np.savez_compressed(args.output / 'checkpoint.npz', mean=mean,
                                std=std, best_gains=best[0], offsets=offsets)
            (args.output / 'search_history.json').write_text(
                json.dumps(history, indent=2), encoding='utf-8')
            b = best[1]
            print('GEN', generation + 1, 'ROBUST', round(b['robust_score'], 2),
                  'SUCCESS', b['successes'], '/', b['case_count'],
                  'VALID', b['valid_cases'], '/', b['case_count'], flush=True)
            if b['successes'] == b['case_count']:
                break
    replay_prefix, replay_capture = capture(sim, reference, 0)
    if replay_capture['state_hash'] != captured['state_hash']:
        raise RuntimeError('Canonical actual-fall capture changed')
    canonical, trace = evaluate_feedback(
        sim, replay_capture['snapshot'], replay_capture['peaks'],
        replay_capture['finite'], offsets, ids, best[0], True)
    heldout = evaluate_set(sim, replay_capture['snapshot'],
                           replay_capture['peaks'], replay_capture['finite'],
                           offsets, ids, best[0], heldout_seeds)
    np.savez_compressed(args.output / 'winner_trace.npz',
                        time=np.array([x[0] for x in trace]),
                        qpos=np.stack([x[1] for x in trace]),
                        qvel=np.stack([x[2] for x in trace]),
                        margin_m=np.array([x[3] for x in trace]),
                        supported=np.array([x[4] for x in trace]),
                        strict=np.array([x[5] for x in trace]),
                        phase=np.array([x[6] for x in trace]),
                        residual_rad=np.stack([x[7] for x in trace]))
    report = {'simulation_only': True, 'hardware_readiness': False,
              'stage': 'captured R49 transition IMU feedback; not full recovery',
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'r49_checkpoint_sha256': digest(args.r49_checkpoint),
              'source_sha256': digest(__file__), 'prefix': prefix,
              'captured': captured, 'training_seeds': train_seeds,
              'heldout_seeds': heldout_seeds,
              'feedback_features': ['up_x', 'up_y', 'gyro_y_x0.15',
                                    'gyro_x_x0.15'],
              'best_gains': best[0].tolist(),
              'generations_completed': len(history),
              'training_best': best[1],
              'canonical_actual_fall_replay': canonical,
              'baseline_heldout_successes': sum(
                  row['strict_tail_s'] >= 1. for row in baseline),
              'heldout_successes': sum(row['strict_tail_s'] >= 1.
                                       for row in heldout),
              'heldout_trials': len(heldout), 'heldout': heldout,
              'full_task_completed': False}
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('HELDOUT', report['heldout_successes'], '/', len(heldout),
          'BASELINE', report['baseline_heldout_successes'], '/', len(baseline),
          flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
