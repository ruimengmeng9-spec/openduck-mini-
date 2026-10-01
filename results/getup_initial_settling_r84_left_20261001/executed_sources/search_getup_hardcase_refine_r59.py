"""Refine R57 on a balanced set of observed boundary failures and successes.

The R57 winner generalized from 1/20 to 11/20 transition perturbations.  This
search actively replays nine independent R57 failures plus three successes and
optimizes the lower tail of their dense physical scores.  Final selection is
reported on forty new seeds that are never used by the search.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from diagnostics.getup_independent_native import digest
from diagnostics.search_getup_robust_transition_r57 import (
    evaluate_set, perturb_snapshot)
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, capture
from diagnostics.search_getup_support_margin_r49 import evaluate
from diagnostics.validate_getup_fullpath_r27 import StrictSim


TRAIN_SEEDS = (758000, 758003, 758007, 758008, 758009, 758010,
               758013, 758016, 758019, 758001, 758004, 758017)
HELDOUT_SEEDS = tuple(range(761000, 761040))
_CTX = None


def init_worker(scene, stand, snapshot, peaks, finite):
    global _CTX
    sim = StrictSim(scene, stand)
    cases = [perturb_snapshot(sim, snapshot, seed) for seed in TRAIN_SEEDS]
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, cases, peaks, finite, ids


def evaluate_candidate(offsets):
    sim, cases, peaks, finite, ids = _CTX
    rows = [evaluate(sim, state, peaks, finite, offsets, ids)[0]
            for state in cases]
    scores = np.sort(np.array([row['score'] for row in rows]))
    successes = sum(row['strict_tail_s'] >= 1. for row in rows)
    valid = sum(row['valid'] for row in rows)
    lower_tail = scores[:min(5, len(scores))].mean()
    objective = float(lower_tail + .2 * scores.mean() + 14. * successes)
    return {'robust_score': objective, 'successes': successes,
            'valid_cases': valid, 'case_count': len(rows),
            'lower_tail_mean': float(lower_tail), 'cases': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--prior-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=64)
    p.add_argument('--population', type=int, default=36)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=159)
    args = p.parse_args()
    if (args.output.exists() or args.generations < 1
            or args.population < 8 or args.workers < 1):
        p.error('Need new output and positive search settings')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    prior = np.load(args.prior_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, captured = capture(sim, reference, 0)
    snapshot, peaks, finite = (captured.pop('snapshot'),
                               captured.pop('peaks'), captured.pop('finite'))
    baseline_train = evaluate_set(sim, snapshot, peaks, finite, prior,
                                  TRAIN_SEEDS)
    baseline_heldout = evaluate_set(sim, snapshot, peaks, finite, prior,
                                    HELDOUT_SEEDS)
    rng = np.random.default_rng(args.seed)
    mean, std = prior.copy(), np.full_like(prior, .14)
    best, history = None, []
    args.output.mkdir(parents=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), snapshot,
                                       peaks, finite)) as pool:
        for generation in range(args.generations):
            population = [mean.copy(), prior.copy()]
            if best is not None:
                population.append(best[0].copy())
            while len(population) < args.population:
                if best is not None and len(population) % 3 == 0:
                    base = best[0]
                elif len(population) % 4 == 0:
                    base = prior
                else:
                    base = mean
                population.append(np.clip(base + rng.normal(0., std), -.9, .9))
            results = list(pool.map(evaluate_candidate, population, chunksize=1))
            ranked = sorted(zip(population, results),
                            key=lambda pair: -pair[1]['robust_score'])
            if best is None or ranked[0][1]['robust_score'] > best[1]['robust_score']:
                best = ranked[0]
            elites = np.stack([x for x, _ in ranked[:max(5, args.population // 6)]])
            mean = .55 * mean + .45 * elites.mean(axis=0)
            std = np.maximum(.045, .72 * std + .28 * elites.std(axis=0))
            row = {'generation': generation + 1,
                   'generation_best': ranked[0][1], 'overall_best': best[1],
                   'wall_seconds': time.monotonic() - started}
            history.append(row)
            np.savez_compressed(args.output / 'checkpoint.npz', mean=mean,
                                std=std, best_offsets=best[0], prior=prior)
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
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    canonical, trace = evaluate(sim, replay_capture['snapshot'],
                                replay_capture['peaks'], replay_capture['finite'],
                                best[0], ids, True)
    heldout = evaluate_set(sim, replay_capture['snapshot'],
                           replay_capture['peaks'], replay_capture['finite'],
                           best[0], HELDOUT_SEEDS)
    np.savez_compressed(args.output / 'winner_trace.npz',
                        time=np.array([x[0] for x in trace]),
                        qpos=np.stack([x[1] for x in trace]),
                        qvel=np.stack([x[2] for x in trace]),
                        margin_m=np.array([x[3] for x in trace]),
                        supported=np.array([x[4] for x in trace]),
                        strict=np.array([x[5] for x in trace]),
                        phase=np.array([x[6] for x in trace]))
    report = {'simulation_only': True, 'hardware_readiness': False,
              'stage': 'captured transition hard-case refinement; not full recovery',
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'prior_checkpoint_sha256': digest(args.prior_checkpoint),
              'source_sha256': digest(__file__), 'prefix': prefix,
              'captured': captured, 'training_seeds': list(TRAIN_SEEDS),
              'heldout_seeds': list(HELDOUT_SEEDS),
              'generations_completed': len(history),
              'best_offsets_rad': best[0].tolist(), 'training_best': best[1],
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
