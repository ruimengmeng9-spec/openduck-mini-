"""Test the existing stand policy as a feedback catch during actual-fall recovery.

The official ONNX stand actor has previously only been switched on after a
strictly loaded entry. This diagnostic deliberately evaluates earlier switch
times in simulation, under unchanged scene, motor limits and strict checks.
It does not alter the policy or imply hardware readiness.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_loaded_balance_r45 import capture, partial_stand, quality
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def prepare_branch(sim, peak, name):
    sim.restore(peak['snapshot'])
    sim.peaks, sim.finite = peak['peaks'].copy(), peak['finite']
    if name == 'peak':
        return sim.snapshot(), sim.peaks.copy(), sim.finite
    if name.startswith('home_'):
        duration = float(name.split('_')[1])
        target = sim.home
    elif name == 'neck_0.06':
        duration = .06
        target = sim.prev.copy()
        j = sim.model.actuator('neck_pitch').id
        target[j] = np.clip(target[j] + .6, sim.lower[j], sim.upper[j])
    else:
        raise ValueError(name)
    for _ in range(round(duration / DT)):
        sim.step_target(target)
        if not sim.physical_valid():
            raise RuntimeError('Feedback branch violated physical audit')
    return sim.snapshot(), sim.peaks.copy(), sim.finite


def test(sim, branch, ramp_s, record=False):
    snapshot, peaks, finite = branch
    sim.restore(snapshot)
    sim.peaks, sim.finite = peaks.copy(), finite
    start = sim.prev.copy()
    strict, partial, values, trace = [], [], [], []
    valid = True
    max_up = -1.
    for k in range(round(3. / DT)):
        alpha = 1. if ramp_s == 0 else min((k + 1) * DT / ramp_s, 1.)
        action = sim.stand_action()
        target = (1-alpha) * start + alpha * (sim.home + .25 * action)
        sim.step_target(target)
        m = sim.measure()
        valid = sim.physical_valid()
        if not valid:
            break
        strict.append(bool(m['stable']))
        partial.append(partial_stand(m))
        values.append(quality(m))
        max_up = max(max_up, m['up_z'])
        if record:
            trace.append((float(sim.data.time), sim.data.qpos.copy(),
                          sim.data.qvel.copy(), int(m['stable']),
                          int(partial[-1])))
    if not valid:
        return {'valid': False, 'score': -100., 'strict_tail_s': 0.,
                'final': m, 'peaks': sim.peaks.copy()}, trace
    tail = 0
    for passed in reversed(strict):
        if not passed:
            break
        tail += 1
    longest = current = 0
    for passed in partial:
        current = current + 1 if passed else 0
        longest = max(longest, current)
    score = (3 * float(np.mean(values[-round(1. / DT):]))
             + 20 * min(tail * DT, 1.) + 5 * min(longest * DT, 1.))
    return {'valid': True, 'score': score,
            'strict_tail_s': tail * DT,
            'partial_longest_s': longest * DT,
            'max_up_z': max_up,
            'final': m, 'peaks': sim.peaks.copy()}, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--r42-checkpoint', type=Path, required=True)
    p.add_argument('--r44-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output directory already exists')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    r42 = np.load(args.r42_checkpoint, allow_pickle=False)['best_offsets']
    r44 = np.load(args.r44_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, _, peak, _ = capture(sim, reference, r42, r44)
    rows = []
    for name, ramp_s in itertools.product(
            ('peak', 'home_0.08', 'home_0.14', 'neck_0.06'),
            (0., .2, .5, 1.)):
        branch = prepare_branch(sim, peak, name)
        result, _ = test(sim, branch, ramp_s)
        rows.append({'branch': name, 'ramp_s': ramp_s, 'result': result})
    ranked = sorted(rows, key=lambda row: -row['result']['score'])
    winner = ranked[0]
    replay_prefix, _, replay_peak, _ = capture(sim, reference, r42, r44)
    if replay_peak['hash'] != peak['hash']:
        raise RuntimeError('Actual-fall replay did not reproduce peak')
    branch = prepare_branch(sim, replay_peak, winner['branch'])
    replay, trace = test(sim, branch, winner['ramp_s'], True)
    if abs(replay['score'] - winner['result']['score']) > 1e-5:
        raise RuntimeError('Winner did not reproduce from actual fall')
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'winner_trace.npz',
                        time=np.array([x[0] for x in trace]),
                        qpos=np.stack([x[1] for x in trace]),
                        qvel=np.stack([x[2] for x in trace]),
                        strict=np.array([x[3] for x in trace]),
                        partial=np.array([x[4] for x in trace]))
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'r42_checkpoint_sha256': digest(args.r42_checkpoint),
              'r44_checkpoint_sha256': digest(args.r44_checkpoint),
              'source_sha256': digest(__file__),
              'prefix': prefix,
              'peak': {k: v for k, v in peak.items()
                       if k not in ('snapshot', 'peaks', 'finite')},
              'candidate_count': len(rows),
              'valid_count': sum(row['result']['valid'] for row in rows),
              'winner': winner, 'winner_full_fall_replay': replay,
              'all': rows, 'full_task_completed': False}
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    for row in ranked:
        m = row['result']['final']
        print(row['branch'], row['ramp_s'],
              'strict', row['result']['strict_tail_s'],
              'partial', row['result']['partial_longest_s'],
              'up', round(m['up_z'], 3),
              'torso', m['torso_contact'], flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()

