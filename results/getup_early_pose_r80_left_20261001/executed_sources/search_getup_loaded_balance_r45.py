"""Search a balance catch after an actual-fall replay reaches both feet.

The development branch is selected only by replaying R27, R42 and R44 from
the original fallen state, followed by 0.2 s of home-directed extension.
All candidates retain the same native physics and actuator audit. A winner
must be replayed from the fallen state; even a brief strict catch is not a
30-second recovered stand or a hardware-ready policy.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_peak_catch_r43 import reach_peak
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, DURATIONS
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


BALANCE_DURATIONS = (.3, .5, .8)
HOME_HOLD_S = 2.


def capture(sim, reference, r42_offsets, r44_offsets):
    prefix, first, peak = reach_peak(sim, reference, r42_offsets, 0)
    sim.restore(peak['snapshot'])
    sim.peaks, sim.finite = peak['peaks'].copy(), peak['finite']
    ids = np.array([sim.model.actuator(n).id for n in JOINTS])
    highest = None
    for phase, (duration, row) in enumerate(zip(DURATIONS, r44_offsets)):
        target = sim.home.copy()
        target[ids] = np.clip(sim.home[ids] + row,
                              sim.lower[ids], sim.upper[ids])
        for control in range(round(duration / DT)):
            sim.step_target(target)
            m = sim.measure()
            if not sim.physical_valid():
                raise RuntimeError('R44 replay violated physical audit')
            if highest is None or m['up_z'] > highest['metric']['up_z']:
                highest = {'phase': phase, 'control_index': control,
                           'metric': m, 'snapshot': sim.snapshot(),
                           'peaks': sim.peaks.copy(), 'finite': sim.finite,
                           'hash': state_hash(sim)}
    sim.restore(highest['snapshot'])
    sim.peaks, sim.finite = highest['peaks'].copy(), highest['finite']
    for _ in range(round(.2 / DT)):
        sim.step_target(sim.home)
        if not sim.physical_valid():
            raise RuntimeError('Pre-catch extension violated physical audit')
    branch = {'metric': sim.measure(), 'snapshot': sim.snapshot(),
              'peaks': sim.peaks.copy(), 'finite': sim.finite,
              'hash': state_hash(sim)}
    return prefix, first, highest, branch


def quality(m):
    height = float(np.clip(1 - abs(m['height_m'] - .17) / .08, 0., 1.))
    load = float(np.clip(m['foot_load_fraction'], 0., 1.))
    flat = float(np.clip(min(m['foot_up_alignment_to_home']), -1., 1.))
    home = float(np.clip(1 - m['joint_home_error_mean_rad'] / .5, 0., 1.))
    return (12 * m['up_z'] + 10 * height + 6 * load + 4 * flat + 4 * home
            - 8 * m['torso_contact']
            - 2 * min(m['angular_speed_rad_s'], 5.)
            - min(m['linear_speed_mps'], 5.))


def partial_stand(m):
    return bool(m['up_z'] > .9 and .14 < m['height_m'] < .20
                and not m['torso_contact'] and all(m['feet'])
                and m['foot_load_fraction'] > .9
                and min(m['foot_up_alignment_to_home']) > .8)


def evaluate(sim, branch, offsets, ids, record=False):
    sim.restore(branch['snapshot'])
    sim.peaks, sim.finite = branch['peaks'].copy(), branch['finite']
    values, strict, partial, trace = [], [], [], []
    best = None
    valid = True
    for phase, duration, row in [
        *(('catch_' + str(i), d, r)
          for i, (d, r) in enumerate(zip(BALANCE_DURATIONS, offsets))),
        ('home_hold', HOME_HOLD_S, None),
    ]:
        target = sim.home.copy()
        if row is not None:
            target[ids] = np.clip(sim.home[ids] + row,
                                  sim.lower[ids], sim.upper[ids])
        for _ in range(round(duration / DT)):
            sim.step_target(target)
            m = sim.measure()
            if not sim.physical_valid():
                valid = False
                break
            v = quality(m)
            values.append(v)
            strict.append(bool(m['stable']))
            partial.append(partial_stand(m))
            if best is None or v > best['quality']:
                best = {'quality': v, 'time_s': float(sim.data.time),
                        'metric': m}
            if record:
                trace.append((float(sim.data.time), sim.data.qpos.copy(),
                              sim.data.qvel.copy(), int(m['stable']),
                              int(partial[-1]), phase))
        if not valid:
            break
    if not valid:
        return {'score': -100., 'valid': False, 'strict_tail_s': 0.,
                'partial_longest_s': 0., 'final': m,
                'peaks': sim.peaks.copy()}, trace
    longest = current = 0
    for passed in partial:
        current = current + 1 if passed else 0
        longest = max(longest, current)
    tail = 0
    for passed in reversed(strict):
        if not passed:
            break
        tail += 1
    recent = values[-round(1. / DT):]
    score = (.2 * best['quality'] + 3 * float(np.mean(recent))
             + 6 * min(longest * DT, 1.)
             + 20 * min(tail * DT, 1.))
    return {'score': score, 'valid': True,
            'strict_tail_s': tail * DT,
            'partial_longest_s': longest * DT,
            'best': best, 'final': m,
            'peaks': sim.peaks.copy()}, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--r42-checkpoint', type=Path, required=True)
    p.add_argument('--r44-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=64)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--seed', type=int, default=145)
    args = p.parse_args()
    if args.output.exists() or args.generations < 1 or args.population < 8:
        p.error('Need new output directory, positive generations, population >=8')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    r42 = np.load(args.r42_checkpoint, allow_pickle=False)['best_offsets']
    r44 = np.load(args.r44_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, first, peak, branch = capture(sim, reference, r42, r44)
    ids = np.array([sim.model.actuator(n).id for n in JOINTS])
    prior = np.tile(r44[1], (len(BALANCE_DURATIONS), 1))
    mean = .5 * prior
    std = np.full_like(mean, .3)
    rng = np.random.default_rng(args.seed)
    best = None
    history = []
    args.output.mkdir(parents=True)
    started = time.monotonic()
    for generation in range(args.generations):
        population = [mean.copy(), prior.copy(), np.zeros_like(mean)]
        if best is not None:
            population.append(best[0].copy())
        while len(population) < args.population:
            base = (best[0] if best is not None and len(population) % 3 == 0
                    else prior if len(population) % 4 == 0 else mean)
            population.append(np.clip(base + rng.normal(0., std), -.9, .9))
        ranked = [(x, evaluate(sim, branch, x, ids)[0]) for x in population]
        ranked.sort(key=lambda pair: -pair[1]['score'])
        if best is None or ranked[0][1]['score'] > best[1]['score']:
            best = ranked[0]
        elites = np.stack([x for x, _ in ranked[:max(4, args.population // 5)]])
        mean = .5 * mean + .5 * elites.mean(axis=0)
        std = np.maximum(.1, .7 * std + .3 * elites.std(axis=0))
        row = {'generation': generation + 1,
               'valid_candidates': sum(r['valid'] for _, r in ranked),
               'generation_best': ranked[0][1], 'overall_best': best[1],
               'wall_seconds': time.monotonic() - started}
        history.append(row)
        np.savez_compressed(args.output / 'checkpoint.npz', mean=mean, std=std,
                            best_offsets=best[0], prior=prior)
        (args.output / 'search_history.json').write_text(
            json.dumps(history, indent=2), encoding='utf-8')
        print('GEN', generation + 1,
              'SCORE', round(best[1]['score'], 2),
              'PARTIAL', round(best[1]['partial_longest_s'], 2),
              'STRICT', round(best[1]['strict_tail_s'], 2),
              'VALID', row['valid_candidates'], flush=True)
        if best[1]['strict_tail_s'] >= 1.:
            break
    replay_prefix, _, replay_peak, replay_branch = capture(sim, reference, r42, r44)
    if replay_peak['hash'] != peak['hash'] or replay_branch['hash'] != branch['hash']:
        raise RuntimeError('Actual-fall replay did not reproduce catch branch')
    replay, trace = evaluate(sim, replay_branch, best[0], ids, True)
    if abs(replay['score'] - best[1]['score']) > 1e-5:
        raise RuntimeError('Winning catch did not reproduce from the actual fall')
    np.savez_compressed(args.output / 'winner_trace.npz',
                        time=np.array([x[0] for x in trace]),
                        qpos=np.stack([x[1] for x in trace]),
                        qvel=np.stack([x[2] for x in trace]),
                        strict=np.array([x[3] for x in trace]),
                        partial=np.array([x[4] for x in trace]),
                        phase=np.array([x[5] for x in trace]))
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'r42_checkpoint_sha256': digest(args.r42_checkpoint),
              'r44_checkpoint_sha256': digest(args.r44_checkpoint),
              'source_sha256': digest(__file__),
              'joint_names': JOINTS, 'balance_durations_s': BALANCE_DURATIONS,
              'home_hold_s': HOME_HOLD_S, 'prefix': prefix,
              'first': {k: v for k, v in first.items()
                        if k not in ('snapshot', 'peaks', 'finite')},
              'peak': {k: v for k, v in peak.items()
                       if k not in ('snapshot', 'peaks', 'finite')},
              'branch': {k: v for k, v in branch.items()
                         if k not in ('snapshot', 'peaks', 'finite')},
              'generations_completed': len(history),
              'best_offsets_rad': best[0].tolist(),
              'best': best[1], 'actual_fall_replay': replay,
              'full_task_completed': False}
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()

