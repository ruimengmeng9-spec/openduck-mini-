"""Learn causal IMU tracking across the entire fallen-start command stream.

Only prefix gains are learned. Terminal R64 gains are frozen. Training falls
are independently initialized and physically settled, with no mid-path root
reset. Candidate qualification includes nominal preservation and unseen falls.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_reference_feedback_r64 import (
    targets_for, record_reference, rollout)
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS
from diagnostics.validate_getup_fullfall_feedback_r66 import frozen_prefix
from diagnostics.validate_getup_fullpath_r27 import StrictSim


_CTX = None


def init_worker(scene, stand, targets, phases, ref, terminal_gains, seeds):
    global _CTX
    sim = StrictSim(scene, stand)
    cases = []
    for seed in [None] + list(seeds):
        sim.prepare('left_side', 0 if seed is None else seed, seed is not None)
        m = sim.measure()
        if not (m['up_z'] < .5 and m['torso_contact']):
            raise RuntimeError('Training reset is not actually fallen')
        cases.append(sim.snapshot())
    sim.clear_audit()
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, cases, sim.peaks.copy(), targets, phases, ref, terminal_gains, ids


def objective(flat):
    sim, cases, peaks, targets, phases, ref, terminal, ids = _CTX
    gains = np.concatenate([np.asarray(flat).reshape(4, 8), terminal], axis=0)
    rows = [rollout(sim, case, peaks, True, targets, phases, ids, ref, gains)[0]
            for case in cases]
    passed = [r['valid'] and r['completed_steps'] == len(targets)
              and r['strict_tail_s'] >= 1. for r in rows]
    scores = np.sort([r['score'] for r in rows[1:]])
    value = float(scores[:3].mean() + .2 * scores.mean()
                  + 14 * sum(passed[1:]) - .05 * np.dot(flat, flat))
    if not passed[0]:
        value -= 1000.
    return {'objective': value, 'successes': sum(passed[1:]),
            'nominal_success': passed[0], 'cases': rows}


def full_reference(sim, initial, peaks, prefix, prefix_phases, offsets, hold):
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    terminal, phases = targets_for(sim, offsets, ids, hold)
    targets = np.concatenate([prefix, terminal])
    phases = np.concatenate([prefix_phases, phases + 4])
    ref = record_reference(sim, initial, peaks, True, targets)
    return targets, phases, ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--terminal-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=24)
    p.add_argument('--population', type=int, default=24)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--training-cases', type=int, default=8)
    p.add_argument('--seed', type=int, default=167)
    args = p.parse_args()
    if (args.output.exists() or args.generations < 1 or args.population < 8
            or args.training_cases < 3 or args.workers < 1):
        p.error('Need fresh output and positive search settings')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    z = np.load(args.reference, allow_pickle=False)
    ck = np.load(args.terminal_checkpoint, allow_pickle=False)
    prefix, prefix_result, captured = frozen_prefix(scene, stand, z)
    recovery_n = sum(round(float(d) / DT) for d in z['durations_s'])
    hold_end, pulse_end = recovery_n + round(31. / DT), recovery_n + round(31.4 / DT)
    if not (len(prefix) > hold_end and len(prefix) <= pulse_end + round(2. / DT)):
        raise RuntimeError('Canonical prefix phase boundaries need explicit re-audit')
    index = np.arange(len(prefix))
    prefix_phases = np.where(index < recovery_n, 0,
                            np.where(index < hold_end, 1,
                                     np.where(index < pulse_end, 2, 3)))
    sim = StrictSim(scene, stand)
    sim.prepare('left_side', 0, False)
    sim.clear_audit()
    initial, peaks = sim.snapshot(), sim.peaks.copy()
    targets, phases, ref = full_reference(sim, initial, peaks, prefix,
                                          prefix_phases, ck['offsets'], 2.)
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    terminal_gains = ck['best_phase_gains']
    base_gains = np.concatenate([np.zeros((4, 8)), terminal_gains])
    nominal, _ = rollout(sim, initial, peaks, True, targets, phases, ids, ref, base_gains)
    if not nominal['valid'] or nominal['strict_tail_s'] < 1.:
        raise RuntimeError('Nominal whole-path baseline did not stand')
    args.output.mkdir(parents=True)
    train_seeds = list(range(769000, 769000 + args.training_cases))
    heldout_seeds = list(range(770000, 770020))
    contract = {'simulation_only': True, 'hardware_readiness': False,
                'full_task_completed': False, 'method': 'whole-path reference-error IMU feedback CEM',
                'pose': 'left_side', 'training_seeds': train_seeds,
                'heldout_seeds': heldout_seeds, 'generation_budget': args.generations,
                'population': args.population, 'workers': args.workers,
                'no_midpath_reset': True, 'nominal_prefix_exact': True,
                'nominal_required': True, 'terminal_gains_frozen': True,
                'prefix_steps': len(prefix), 'phase_counts': np.bincount(prefix_phases).tolist(),
                'source_sha256': digest(__file__), 'scene_sha256': digest(scene),
                'stand_sha256': digest(stand), 'reference_sha256': digest(args.reference),
                'terminal_checkpoint_sha256': digest(args.terminal_checkpoint)}
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    rng = np.random.default_rng(args.seed)
    mean, std = np.zeros(32), np.full(32, .18)
    best, history = None, []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), targets, phases, ref,
                                       terminal_gains, train_seeds)) as pool:
        for generation in range(args.generations):
            population = [np.zeros(32), mean.copy()]
            if best is not None:
                population.append(best[0].copy())
            while len(population) < args.population:
                base = best[0] if best is not None and len(population) % 3 == 0 else mean
                population.append(np.clip(base + rng.normal(0., std), -2., 2.))
            ranked = sorted(zip(population, pool.map(objective, population)),
                            key=lambda x: -x[1]['objective'])
            if best is None or ranked[0][1]['objective'] > best[1]['objective']:
                best = ranked[0]
            elite = np.stack([x[0] for x in ranked[:max(4, args.population // 5)]])
            mean = .5 * mean + .5 * elite.mean(axis=0)
            std = np.maximum(.035, .7 * std + .3 * elite.std(axis=0))
            history.append({'generation': generation + 1, 'best': best[1],
                            'wall_seconds': time.monotonic() - start})
            np.savez_compressed(args.output / 'checkpoint.npz', mean=mean, std=std,
                                prefix_gains=best[0].reshape(4, 8), terminal_gains=terminal_gains,
                                offsets=ck['offsets'], prefix_targets=prefix,
                                prefix_phases=prefix_phases)
            (args.output / 'search_history.json').write_text(json.dumps(history, indent=2))
            print('GEN', generation + 1, 'SUCCESS', best[1]['successes'], '/', len(train_seeds),
                  'NOMINAL', int(best[1]['nominal_success']),
                  'OBJECTIVE', round(best[1]['objective'], 2), flush=True)
            if best[1]['successes'] == len(train_seeds):
                break
    # Thirty-second qualification is strictly post-training and does not tune gains.
    targets, phases, ref = full_reference(sim, initial, peaks, prefix, prefix_phases,
                                          ck['offsets'], 35.)
    gains = np.concatenate([best[0].reshape(4, 8), terminal_gains])
    rows = []
    for seed in [None] + heldout_seeds:
        sim.prepare('left_side', 0 if seed is None else seed, seed is not None)
        m = sim.measure()
        fallen = bool(m['up_z'] < .5 and m['torso_contact'])
        state = sim.snapshot()
        row = {'seed': seed, 'initial_fallen': fallen}
        for name, feedback in [('baseline', base_gains), ('candidate', gains)]:
            value, _ = rollout(sim, state, peaks, True, targets, phases, ids, ref, feedback)
            value['success'] = bool(fallen and value['valid']
                                    and value['completed_steps'] == len(targets)
                                    and value['strict_tail_s'] >= 30. - 1e-8)
            row[name] = value
        rows.append(row)
        print('HELDOUT', seed, int(row['baseline']['success']), int(row['candidate']['success']), flush=True)
    rs = rows[1:]
    report = {**contract, 'generations_completed': len(history), 'training_best': best[1],
              'canonical': rows[0], 'heldout': rs,
              'baseline_successes': sum(x['baseline']['success'] for x in rs),
              'heldout_successes': sum(x['candidate']['success'] for x in rs),
              'heldout_trials': len(rs), 'required_strict_seconds': 30.}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
