"""Phase-aware IMU residual search around the selected R59 get-up path.

R58 used one feedback gain vector for every part of recovery and overfit.  R63
separates the three recovery phases and the final home hold, while preserving
the original model, contacts, actuator limits, target slew and strict success
gate.  Search and evaluation are simulation-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_feedback_balance_r58 import feedback_residual
from diagnostics.search_getup_robust_transition_r57 import perturb_snapshot
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, DURATIONS, capture
from diagnostics.search_getup_support_margin_r49 import (
    HOME_HOLD_S, foot_support, support_quality, supported_entry)
from diagnostics.validate_getup_fullpath_r27 import StrictSim


TRAIN_SEEDS = tuple(range(763000, 763016))
HELDOUT_SEEDS = tuple(range(764000, 764040))
_CTX = None


def evaluate_phase_feedback(sim, snapshot, peaks, finite, offsets, ids,
                            gains, record=False):
    sim.restore(snapshot)
    sim.peaks, sim.finite = peaks.copy(), finite
    best = None
    supports, strict, trace = [], [], []
    valid = True
    phases = [(f'phase_{i}', seconds, row)
              for i, (seconds, row) in enumerate(zip(DURATIONS, offsets))]
    phases.append(('home_hold', HOME_HOLD_S, None))
    for phase_index, (phase, seconds, row) in enumerate(phases):
        base = sim.home.copy()
        if row is not None:
            base[ids] = np.clip(sim.home[ids] + row,
                                sim.lower[ids], sim.upper[ids])
        for _ in range(round(seconds / DT)):
            residual = feedback_residual(sim, gains[phase_index])
            target = base.copy()
            target[ids] = np.clip(target[ids] + residual,
                                  sim.lower[ids], sim.upper[ids])
            sim.step_target(target)
            metric = sim.measure()
            valid = sim.physical_valid()
            if not valid:
                break
            support = foot_support(sim)
            quality = support_quality(metric, support)
            if best is None or quality > best['quality']:
                best = {'quality': quality, 'time_s': float(sim.data.time),
                        'metric': metric, 'support': support, 'phase': phase}
            supports.append(supported_entry(metric, support))
            strict.append(bool(metric['stable']))
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
                'final': metric, 'peaks': sim.peaks.copy()}, trace
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
            'strict_tail_s': tail * DT, 'best': best, 'final': metric,
            'final_support': final_support, 'peaks': sim.peaks.copy()}, trace


def init_worker(scene, stand, snapshot, peaks, finite, offsets):
    global _CTX
    sim = StrictSim(scene, stand)
    cases = [perturb_snapshot(sim, snapshot, seed) for seed in TRAIN_SEEDS]
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, cases, peaks, finite, offsets, ids


def evaluate_candidate(flat_gains):
    sim, cases, peaks, finite, offsets, ids = _CTX
    gains = np.asarray(flat_gains).reshape(4, 8)
    rows = [evaluate_phase_feedback(sim, state, peaks, finite, offsets,
                                    ids, gains)[0] for state in cases]
    scores = np.sort(np.array([row['score'] for row in rows]))
    successes = sum(row['strict_tail_s'] >= 1. for row in rows)
    valid = sum(row['valid'] for row in rows)
    lower_tail = scores[:5].mean()
    objective = float(lower_tail + .2 * scores.mean()
                      + 14. * successes - .03 * np.dot(flat_gains, flat_gains))
    return {'robust_score': objective, 'successes': successes,
            'valid_cases': valid, 'case_count': len(rows),
            'lower_tail_mean': float(lower_tail), 'cases': rows}


def evaluate_set(sim, source, peaks, finite, offsets, ids, gains, seeds):
    rows = []
    for seed in seeds:
        state = perturb_snapshot(sim, source, seed)
        result, _ = evaluate_phase_feedback(sim, state, peaks, finite,
                                            offsets, ids, gains)
        rows.append({'seed': seed, **result})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--path-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=64)
    p.add_argument('--population', type=int, default=40)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=163)
    args = p.parse_args()
    if (args.output.exists() or args.generations < 1
            or args.population < 8 or args.workers < 1):
        p.error('Need new output and positive search settings')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    offsets = np.load(args.path_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, captured = capture(sim, reference, 0)
    snapshot, peaks, finite = (captured.pop('snapshot'),
                               captured.pop('peaks'), captured.pop('finite'))
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    zero = np.zeros((4, 8))
    baseline_train = evaluate_set(sim, snapshot, peaks, finite, offsets,
                                  ids, zero, TRAIN_SEEDS)
    baseline_heldout = evaluate_set(sim, snapshot, peaks, finite, offsets,
                                    ids, zero, HELDOUT_SEEDS)
    rng = np.random.default_rng(args.seed)
    mean, std = np.zeros(32), np.full(32, .18)
    best, history = None, []
    args.output.mkdir(parents=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), snapshot,
                                       peaks, finite, offsets)) as pool:
        for generation in range(args.generations):
            population = [mean.copy(), np.zeros(32)]
            if best is not None:
                population.append(best[0].copy())
            while len(population) < args.population:
                base = best[0] if best is not None and len(population) % 3 == 0 else mean
                population.append(np.clip(base + rng.normal(0., std), -1.5, 1.5))
            results = list(pool.map(evaluate_candidate, population, chunksize=1))
            ranked = sorted(zip(population, results),
                            key=lambda pair: -pair[1]['robust_score'])
            if best is None or ranked[0][1]['robust_score'] > best[1]['robust_score']:
                best = ranked[0]
            elites = np.stack([x for x, _ in ranked[:max(5, args.population // 6)]])
            mean = .55 * mean + .45 * elites.mean(axis=0)
            std = np.maximum(.04, .72 * std + .28 * elites.std(axis=0))
            row = {'generation': generation + 1,
                   'generation_best': ranked[0][1], 'overall_best': best[1],
                   'wall_seconds': time.monotonic() - started}
            history.append(row)
            np.savez_compressed(args.output / 'checkpoint.npz', mean=mean,
                                std=std, best_phase_gains=best[0].reshape(4, 8),
                                offsets=offsets)
            (args.output / 'search_history.json').write_text(
                json.dumps(history, indent=2), encoding='utf-8')
            b = best[1]
            print('GEN', generation + 1, 'ROBUST', round(b['robust_score'], 2),
                  'LOWER', round(b['lower_tail_mean'], 2),
                  'SUCCESS', b['successes'], '/', b['case_count'],
                  'VALID', b['valid_cases'], '/', b['case_count'], flush=True)
            if b['successes'] == b['case_count']:
                break
    replay_prefix, replay_capture = capture(sim, reference, 0)
    if replay_capture['state_hash'] != captured['state_hash']:
        raise RuntimeError('Canonical actual-fall capture changed')
    gains = best[0].reshape(4, 8)
    canonical, trace = evaluate_phase_feedback(
        sim, replay_capture['snapshot'], replay_capture['peaks'],
        replay_capture['finite'], offsets, ids, gains, True)
    heldout = evaluate_set(sim, replay_capture['snapshot'],
                           replay_capture['peaks'], replay_capture['finite'],
                           offsets, ids, gains, HELDOUT_SEEDS)
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
              'stage': 'phase-aware transition feedback; not full recovery',
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'path_checkpoint_sha256': digest(args.path_checkpoint),
              'source_sha256': digest(__file__), 'prefix': prefix,
              'captured': captured, 'training_seeds': list(TRAIN_SEEDS),
              'heldout_seeds': list(HELDOUT_SEEDS),
              'generations_completed': len(history),
              'best_phase_gains': gains.tolist(), 'training_best': best[1],
              'canonical_actual_fall_replay': canonical,
              'baseline_training_successes': sum(
                  row['strict_tail_s'] >= 1. for row in baseline_train),
              'baseline_heldout_successes': sum(
                  row['strict_tail_s'] >= 1. for row in baseline_heldout),
              'heldout_successes': sum(row['strict_tail_s'] >= 1.
                                       for row in heldout),
              'heldout_trials': len(heldout), 'heldout': heldout,
              'full_task_completed': False}
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('HELDOUT', report['heldout_successes'], '/', len(heldout),
          'BASELINE', report['baseline_heldout_successes'], '/', len(heldout),
          flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
