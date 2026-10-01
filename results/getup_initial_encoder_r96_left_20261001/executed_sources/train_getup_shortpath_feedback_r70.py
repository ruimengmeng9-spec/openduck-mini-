"""Expanded fallen-start CEM on the physically qualified 2 s wait prefix.

This is an independent simulation skill, not a replacement walking actor.
Only prefix IMU gains are trained; reference targets and terminal gains stay
fixed. Fresh forty-seed long-hold qualification is performed after training.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from diagnostics import search_getup_fullfall_tracking_r67 as training
from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_reference_feedback_r64 import rollout
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def qualify(job):
    index, baseline_flat, candidate_flat = job
    sim, cases, peaks, targets, phases, ref, terminal, ids = training._CTX
    row = {'seed': None if index == 0 else 774000 + index - 1}
    for name, flat in [('baseline', baseline_flat), ('candidate', candidate_flat)]:
        gains = np.concatenate([np.asarray(flat).reshape(4, 8), terminal])
        value, _ = rollout(sim, cases[index], peaks, True, targets, phases, ids, ref, gains)
        value['success'] = bool(value['valid'] and value['completed_steps'] == len(targets)
                                and value['strict_tail_s'] >= 30. - 1e-8)
        row[name] = value
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--prefix-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=32)
    p.add_argument('--population', type=int, default=24)
    p.add_argument('--training-cases', type=int, default=24)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=170)
    args = p.parse_args()
    if (args.output.exists() or args.generations < 1 or args.population < 8
            or args.training_cases < 8 or args.workers < 1):
        p.error('Need fresh output and valid training settings')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    ck = np.load(args.prefix_checkpoint, allow_pickle=False)
    prefix, pp = ck['prefix_targets'], ck['prefix_phases']
    terminal, offsets = ck['terminal_gains'], ck['offsets']
    baseline = ck['prefix_gains'].reshape(-1).copy()
    sim = StrictSim(scene, stand)
    sim.prepare('left_side', 0, False)
    sim.clear_audit()
    initial, peaks = sim.snapshot(), sim.peaks.copy()
    targets, phases, ref = training.full_reference(sim, initial, peaks, prefix, pp, offsets, 2.)
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    nominal, _ = rollout(sim, initial, peaks, True, targets, phases, ids, ref,
                          np.concatenate([baseline.reshape(4, 8), terminal]))
    if not nominal['valid'] or nominal['strict_tail_s'] < 1.:
        raise RuntimeError('Short path failed nominal physical standing gate')
    train_seeds = list(range(769000, 769008)) + list(range(773000, 773000 + args.training_cases - 8))
    heldout_seeds = list(range(774000, 774040))
    if set(train_seeds) & set(heldout_seeds):
        raise RuntimeError('Train/test split overlaps')
    args.output.mkdir(parents=True)
    contract = {'simulation_only': True, 'hardware_readiness': False,
                'full_task_completed': False, 'pose': 'left_side',
                'method': 'short-wait whole-fallen-path reference-error feedback CEM',
                'training_seeds': train_seeds, 'heldout_seeds': heldout_seeds,
                'prefix_seconds': len(prefix) * DT, 'wait_seconds': float(ck['wait_s']),
                'target_stream_frozen': True, 'terminal_gains_frozen': True,
                'nominal_required': True, 'no_midpath_root_reset': True,
                'generation_budget': args.generations, 'population': args.population,
                'workers': args.workers, 'source_sha256': digest(__file__),
                'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
                'prefix_checkpoint_sha256': digest(args.prefix_checkpoint)}
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    rng = np.random.default_rng(args.seed)
    mean, std = baseline.copy(), np.full(32, .14)
    best, history, stale = None, [], 0
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=training.init_worker,
                             initargs=(str(scene), str(stand), targets, phases, ref, terminal,
                                       train_seeds)) as pool:
        for generation in range(args.generations):
            population = [baseline.copy(), np.zeros(32), mean.copy()]
            if best is not None:
                population.append(best[0].copy())
            while len(population) < args.population:
                center = best[0] if best is not None and len(population) % 3 == 0 else mean
                population.append(np.clip(center + rng.normal(0., std), -2., 2.))
            ranked = sorted(zip(population, pool.map(training.objective, population)),
                            key=lambda x: -x[1]['objective'])
            if best is None or ranked[0][1]['objective'] > best[1]['objective'] + 1e-8:
                best, stale = ranked[0], 0
            else:
                stale += 1
            elite = np.stack([x[0] for x in ranked[:max(4, args.population // 5)]])
            mean = .5 * mean + .5 * elite.mean(axis=0)
            std = np.maximum(.04, .7 * std + .3 * elite.std(axis=0))
            history.append({'generation': generation + 1, 'best': best[1],
                            'stale_generations': stale, 'wall_seconds': time.monotonic() - start})
            np.savez_compressed(args.output / 'checkpoint.npz', mean=mean, std=std,
                                prefix_gains=best[0].reshape(4, 8), terminal_gains=terminal,
                                offsets=offsets, prefix_targets=prefix, prefix_phases=pp,
                                wait_s=float(ck['wait_s']), baseline_gains=baseline)
            (args.output / 'search_history.json').write_text(json.dumps(history, indent=2))
            print('GEN', generation + 1, 'SUCCESS', best[1]['successes'], '/', len(train_seeds),
                  'NOMINAL', int(best[1]['nominal_success']), 'OBJECTIVE',
                  round(best[1]['objective'], 2), 'STALE', stale, flush=True)
            if best[1]['successes'] == len(train_seeds) or stale >= 12:
                break
    targets, phases, ref = training.full_reference(sim, initial, peaks, prefix, pp, offsets, 35.)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=training.init_worker,
                             initargs=(str(scene), str(stand), targets, phases, ref, terminal,
                                       heldout_seeds)) as pool:
        for row in pool.map(qualify, [(i, baseline, best[0]) for i in range(41)], chunksize=1):
            rows.append(row)
            print('HELDOUT', row['seed'], int(row['baseline']['success']),
                  int(row['candidate']['success']), flush=True)
    rs = rows[1:]
    report = {**contract, 'generations_completed': len(history), 'training_best': best[1],
              'canonical': rows[0], 'heldout': rs, 'heldout_trials': len(rs),
              'baseline_successes': sum(x['baseline']['success'] for x in rs),
              'heldout_successes': sum(x['candidate']['success'] for x in rs),
              'required_strict_seconds': 30., 'hold_seconds': 35.}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
