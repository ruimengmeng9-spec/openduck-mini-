"""Learn phase gains on deviations from an audited successful R59 rollout.

The reference has body-frame IMU features only. It does not inject state or
change physics. Nominal recovery is included as a mandatory training case;
feedback is computed before each 50 Hz command and passed through StrictSim.
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
    evaluate, foot_support, support_quality, supported_entry)
from diagnostics.validate_getup_fullpath_r27 import StrictSim


_CTX = None


def features(sim):
    up, gyro = sim.sensor('upvector'), sim.sensor('gyro')
    return np.array([up[0], up[1], .15 * gyro[1], .15 * gyro[0]])


def residual(error, gains):
    pitch, roll, pitch_rate, roll_rate = error
    sagittal = gains[:3] * pitch + gains[3:6] * pitch_rate
    lateral = gains[6] * roll + gains[7] * roll_rate
    out = np.zeros(len(JOINTS))
    out[[1, 5]] = [sagittal[0], -sagittal[0]]
    out[[2, 6]] = sagittal[1]
    out[[3, 7]] = sagittal[2]
    out[[0, 4]] = [lateral, -lateral]
    return np.clip(out, -.18, .18)


def targets_for(sim, offsets, ids, hold_s):
    targets, phases = [], []
    for phase, duration in enumerate((*DURATIONS, hold_s)):
        target = sim.home.copy()
        if phase < 3:
            target[ids] = np.clip(sim.home[ids] + offsets[phase],
                                  sim.lower[ids], sim.upper[ids])
        for _ in range(round(duration / DT)):
            targets.append(target.copy())
            phases.append(phase)
    return np.array(targets), np.array(phases)


def record_reference(sim, source, peaks, finite, targets):
    sim.restore(source)
    sim.peaks, sim.finite = peaks.copy(), finite
    result = []
    for target in targets:
        result.append(features(sim))
        sim.step_target(target)
        if not sim.physical_valid():
            raise RuntimeError('Nominal R59 reference failed physical audit')
    return np.array(result)


def rollout(sim, source, peaks, finite, targets, phases, ids, ref, gains,
            record=False):
    sim.restore(source)
    sim.peaks, sim.finite = peaks.copy(), finite
    best = -np.inf
    tail = longest = run = 0
    valid = True
    trace = []
    m = sim.measure()
    for i, (base, phase) in enumerate(zip(targets, phases)):
        error = features(sim) - ref[i]
        correction = residual(error, gains[phase])
        target = base.copy()
        target[ids] = np.clip(target[ids] + correction,
                              sim.lower[ids], sim.upper[ids])
        sim.step_target(target)
        valid = sim.physical_valid()
        m = sim.measure()
        support = foot_support(sim)
        best = max(best, support_quality(m, support))
        run = run + 1 if supported_entry(m, support) else 0
        longest = max(longest, run)
        tail = tail + 1 if m['stable'] else 0
        if record:
            trace.append((sim.data.time, sim.data.qpos.copy(),
                          sim.data.qvel.copy(), int(phase), int(m['stable']),
                          support['margin_m'], error.copy(), correction.copy()))
        if not valid:
            break
    score = (-100. if not valid else
             best + 10 * min(longest * DT, 1.) + 20 * min(tail * DT, 1.))
    return {'score': float(score), 'valid': bool(valid),
            'strict_tail_s': tail * DT, 'supported_longest_s': longest * DT,
            'completed_steps': i + 1, 'final': m,
            'final_support': support, 'peaks': sim.peaks.copy()}, trace


def init_worker(scene, stand, source, peaks, finite, targets, phases, ref, seeds):
    global _CTX
    sim = StrictSim(scene, stand)
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    cases = [source] + [perturb_snapshot(sim, source, seed) for seed in seeds]
    _CTX = sim, cases, peaks, finite, targets, phases, ids, ref


def objective(flat):
    sim, cases, peaks, finite, targets, phases, ids, ref = _CTX
    gains = np.asarray(flat).reshape(4, 8)
    rows = [rollout(sim, case, peaks, finite, targets, phases, ids, ref, gains)[0]
            for case in cases]
    passed = [r['valid'] and r['strict_tail_s'] >= 1. for r in rows]
    scores = np.sort([r['score'] for r in rows[1:]])
    value = (float(scores[:6].mean() + .2 * scores.mean())
             + 14 * sum(passed[1:]) - .03 * float(np.dot(flat, flat)))
    if not passed[0]:
        value -= 1000.
    return {'objective': value, 'successes': sum(passed[1:]),
            'nominal_success': passed[0], 'valid_cases': sum(r['valid'] for r in rows),
            'cases': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--path-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=48)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--training-cases', type=int, default=24)
    p.add_argument('--patience', type=int, default=14)
    p.add_argument('--seed', type=int, default=164)
    args = p.parse_args()
    if (args.output.exists() or args.generations < 1 or args.population < 8
            or args.workers < 1 or args.training_cases < 6):
        p.error('Need fresh output and valid search settings')
    args.output.mkdir(parents=True)
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    z = np.load(args.reference, allow_pickle=False)
    offsets = np.load(args.path_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, captured = capture(sim, z, 0)
    source, peaks, finite = (captured.pop('snapshot'), captured.pop('peaks'),
                             captured.pop('finite'))
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    targets, phases = targets_for(sim, offsets, ids, 2.)
    ref = record_reference(sim, source, peaks, finite, targets)
    train_seeds = list(range(765000, 765000 + args.training_cases))
    heldout_seeds = list(range(766000, 766040))
    baseline, baseline_trace = rollout(sim, source, peaks, finite, targets,
                                      phases, ids, ref, np.zeros((4, 8)), True)
    original, _ = evaluate(sim, source, peaks, finite, offsets, ids)
    if abs(baseline['score'] - original['score']) > 1e-8:
        raise RuntimeError('Zero-feedback contract disagrees with R59 evaluator')
    if baseline['strict_tail_s'] < 1.:
        raise RuntimeError('Selected R59 canonical reference is not successful')
    rng = np.random.default_rng(args.seed)
    mean, std = np.zeros(32), np.full(32, .24)
    best, history, stale = None, [], 0
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), source, peaks,
                                       finite, targets, phases, ref, train_seeds)) as pool:
        for generation in range(args.generations):
            population = [np.zeros(32), mean.copy()]
            if best is not None:
                population.append(best[0].copy())
            while len(population) < args.population:
                base = best[0] if best is not None and len(population) % 3 == 0 else mean
                population.append(np.clip(base + rng.normal(0., std), -2., 2.))
            ranked = sorted(zip(population, pool.map(objective, population)),
                            key=lambda x: -x[1]['objective'])
            if best is None or ranked[0][1]['objective'] > best[1]['objective'] + 1e-6:
                best, stale = ranked[0], 0
            else:
                stale += 1
            elite = np.stack([x[0] for x in ranked[:max(5, args.population // 6)]])
            mean = .5 * mean + .5 * elite.mean(axis=0)
            std = np.maximum(.045, .7 * std + .3 * elite.std(axis=0))
            history.append({'generation': generation + 1, 'best': best[1],
                            'wall_seconds': time.monotonic() - start})
            np.savez_compressed(args.output / 'checkpoint.npz', mean=mean, std=std,
                                offsets=offsets, best_phase_gains=best[0].reshape(4, 8),
                                reference_features=ref, reference_targets=targets,
                                reference_phases=phases)
            (args.output / 'search_history.json').write_text(json.dumps(history, indent=2))
            print('GEN', generation + 1, 'OBJECTIVE', round(best[1]['objective'], 2),
                  'SUCCESS', best[1]['successes'], '/', len(train_seeds),
                  'NOMINAL', int(best[1]['nominal_success']), 'STALE', stale, flush=True)
            if best[1]['successes'] == len(train_seeds) or stale >= args.patience:
                break
    gains = best[0].reshape(4, 8)
    canonical, trace = rollout(sim, source, peaks, finite, targets, phases,
                               ids, ref, gains, True)
    rows = []
    for seed in heldout_seeds:
        state = perturb_snapshot(sim, source, seed)
        base, _ = rollout(sim, state, peaks, finite, targets, phases, ids, ref,
                           np.zeros((4, 8)))
        candidate, _ = rollout(sim, state, peaks, finite, targets, phases, ids, ref, gains)
        rows.append({'seed': seed, 'baseline': base, 'candidate': candidate})
    np.savez_compressed(args.output / 'winner_trace.npz',
                        time=np.array([x[0] for x in trace]),
                        qpos=np.array([x[1] for x in trace]),
                        qvel=np.array([x[2] for x in trace]),
                        phase=np.array([x[3] for x in trace]),
                        strict=np.array([x[4] for x in trace]),
                        margin_m=np.array([x[5] for x in trace]),
                        error=np.array([x[6] for x in trace]),
                        residual_rad=np.array([x[7] for x in trace]))
    report = {'simulation_only': True, 'hardware_readiness': False,
              'stage': 'captured transition reference-error feedback; not full recovery',
              'source_sha256': digest(__file__), 'scene_sha256': digest(scene),
              'stand_sha256': digest(stand), 'reference_sha256': digest(args.reference),
              'path_checkpoint_sha256': digest(args.path_checkpoint),
              'training_seeds': train_seeds, 'heldout_seeds': heldout_seeds,
              'prefix': prefix, 'captured': captured,
              'zero_feedback_contract_passed': True,
              'training_best': best[1], 'generations_completed': len(history),
              'canonical': canonical, 'heldout': rows,
              'baseline_heldout_successes': sum(x['baseline']['strict_tail_s'] >= 1. for x in rows),
              'heldout_successes': sum(x['candidate']['strict_tail_s'] >= 1. for x in rows),
              'heldout_trials': len(rows), 'full_task_completed': False}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('HELDOUT', report['heldout_successes'], '/', len(rows),
          'BASELINE', report['baseline_heldout_successes'], 'NOMINAL',
          canonical['strict_tail_s'], flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
