"""Robust CEM around the physically reached R49 transition state.

Training resets perturb only the captured transition between episodes. Runtime
dynamics, contacts, motor limits, target slew, and strict standing gates remain
unchanged. The canonical winner is still replayed from the original full fall.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from diagnostics.getup_independent_native import digest
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, capture
from diagnostics.search_getup_support_margin_r49 import evaluate
from diagnostics.validate_getup_fullpath_r27 import StrictSim


_CTX = None


def perturb_snapshot(sim, source, seed, orientation_rad=.018,
                     joint_rad=.008, velocity=.008):
    rng = np.random.default_rng(seed)
    state = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in source.items()}
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-12)
    dq = np.empty(4)
    mujoco.mju_axisAngle2Quat(dq, axis, rng.uniform(-orientation_rad,
                                                    orientation_rad))
    q = np.empty(4)
    mujoco.mju_mulQuat(q, dq, state['qpos'][3:7])
    state['qpos'][3:7] = q / np.linalg.norm(q)
    state['qpos'][sim.qadr] = np.clip(
        state['qpos'][sim.qadr]
        + rng.uniform(-joint_rad, joint_rad, sim.model.nu),
        sim.lower, sim.upper)
    state['qvel'] += rng.uniform(-velocity, velocity, sim.model.nv)
    state['warmstart'][:] = 0.
    sim.restore(state)
    if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
        raise RuntimeError('non-finite perturbed transition reset')
    return sim.snapshot()


def init_worker(scene, stand, snapshot, peaks, finite, seeds):
    global _CTX
    sim = StrictSim(scene, stand)
    cases = [perturb_snapshot(sim, snapshot, seed) for seed in seeds]
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, cases, peaks, finite, ids


def evaluate_candidate(offsets):
    sim, cases, peaks, finite, ids = _CTX
    rows = [evaluate(sim, state, peaks, finite, offsets, ids)[0]
            for state in cases]
    scores = np.array([row['score'] for row in rows])
    successes = sum(row['strict_tail_s'] >= 1. for row in rows)
    valid = sum(row['valid'] for row in rows)
    robust_score = float(scores.min() + .25 * scores.mean() + 12. * successes)
    return {'robust_score': robust_score, 'successes': successes,
            'valid_cases': valid, 'case_count': len(rows), 'cases': rows}


def evaluate_set(sim, source, peaks, finite, offsets, seeds):
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    rows = []
    for seed in seeds:
        state = perturb_snapshot(sim, source, seed)
        result, _ = evaluate(sim, state, peaks, finite, offsets, ids)
        rows.append({'seed': seed, **result})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--prior-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=48)
    p.add_argument('--population', type=int, default=24)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--training-cases', type=int, default=6)
    p.add_argument('--seed', type=int, default=157)
    args = p.parse_args()
    if (args.output.exists() or args.generations < 1 or args.population < 8
            or args.training_cases < 2 or args.workers < 1):
        p.error('Need new output and positive search settings')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    prior = np.load(args.prior_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, captured = capture(sim, reference, 0)
    snapshot, peaks, finite = (captured.pop('snapshot'),
                               captured.pop('peaks'), captured.pop('finite'))
    train_seeds = [757000 + i for i in range(args.training_cases)]
    heldout_seeds = [758000 + i for i in range(20)]
    baseline = evaluate_set(sim, snapshot, peaks, finite, prior, heldout_seeds)
    rng = np.random.default_rng(args.seed)
    mean, std = prior.copy(), np.full_like(prior, .25)
    best = None
    history = []
    args.output.mkdir(parents=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), snapshot, peaks,
                                       finite, train_seeds)) as pool:
        for generation in range(args.generations):
            population = [mean.copy(), prior.copy(), np.zeros_like(mean)]
            if best is not None:
                population.append(best[0].copy())
            while len(population) < args.population:
                base = (best[0] if best is not None and len(population) % 3 == 0
                        else prior if len(population) % 4 == 0 else mean)
                population.append(np.clip(base + rng.normal(0., std), -.9, .9))
            results = list(pool.map(evaluate_candidate, population, chunksize=1))
            ranked = sorted(zip(population, results),
                            key=lambda pair: -pair[1]['robust_score'])
            if best is None or ranked[0][1]['robust_score'] > best[1]['robust_score']:
                best = ranked[0]
            elites = np.stack([x for x, _ in ranked[:max(4, args.population // 5)]])
            mean = .5 * mean + .5 * elites.mean(axis=0)
            std = np.maximum(.07, .7 * std + .3 * elites.std(axis=0))
            row = {'generation': generation + 1,
                   'generation_best': ranked[0][1],
                   'overall_best': best[1],
                   'wall_seconds': time.monotonic() - started}
            history.append(row)
            np.savez_compressed(args.output / 'checkpoint.npz', mean=mean,
                                std=std, best_offsets=best[0], prior=prior)
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
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    canonical, trace = evaluate(sim, replay_capture['snapshot'],
                                replay_capture['peaks'], replay_capture['finite'],
                                best[0], ids, True)
    heldout = evaluate_set(sim, replay_capture['snapshot'],
                           replay_capture['peaks'], replay_capture['finite'],
                           best[0], heldout_seeds)
    np.savez_compressed(args.output / 'winner_trace.npz',
                        time=np.array([x[0] for x in trace]),
                        qpos=np.stack([x[1] for x in trace]),
                        qvel=np.stack([x[2] for x in trace]),
                        margin_m=np.array([x[3] for x in trace]),
                        supported=np.array([x[4] for x in trace]),
                        strict=np.array([x[5] for x in trace]),
                        phase=np.array([x[6] for x in trace]))
    report = {'simulation_only': True, 'hardware_readiness': False,
              'stage': 'captured R49 transition only; not full fallen recovery',
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'prior_checkpoint_sha256': digest(args.prior_checkpoint),
              'source_sha256': digest(__file__), 'prefix': prefix,
              'captured': captured, 'training_seeds': train_seeds,
              'heldout_seeds': heldout_seeds,
              'transition_perturbations': {'orientation_rad': .018,
                                           'joint_position_rad': .008,
                                           'generalized_velocity': .008},
              'generations_completed': len(history),
              'best_offsets_rad': best[0].tolist(), 'training_best': best[1],
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
