"""Sweep slow extension and a transient neck counter-motion after R44 peak.

Every development branch is physically reached by replay from the original
left-side fall. The winner is reconstructed from that fall. This sweep only
tests whether a smooth transition can prevent the measured early tip; it
does not relax the final strict loaded-standing test or change motor limits.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_loaded_balance_r45 import capture, quality, partial_stand
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def rollout(sim, peak, ramp_s, neck_rad, hip_rad, record=False):
    sim.restore(peak['snapshot'])
    sim.peaks, sim.finite = peak['peaks'].copy(), peak['finite']
    initial = sim.prev.copy()
    neck = sim.model.actuator('neck_pitch').id
    hip = sim.model.actuator('right_hip_pitch').id
    values, strict, partial, trace = [], [], [], []
    best_partial = None
    valid = True
    phases = [('ramp', round(ramp_s / DT)), ('home', round(2. / DT))]
    for phase, n in phases:
        for k in range(n):
            if phase == 'ramp':
                alpha = (k + 1) / n
                target = initial + alpha * (sim.home - initial)
                target[neck] += neck_rad * (1 - alpha)
                target[hip] += hip_rad * (1 - alpha)
            else:
                target = sim.home
            sim.step_target(target)
            m = sim.measure()
            valid = sim.physical_valid()
            if not valid:
                break
            values.append(quality(m))
            strict.append(bool(m['stable']))
            passed = partial_stand(m)
            partial.append(passed)
            if passed and (best_partial is None or m['up_z'] > best_partial['up_z']):
                best_partial = m
            if record:
                trace.append((float(sim.data.time), sim.data.qpos.copy(),
                              sim.data.qvel.copy(), int(m['stable']),
                              int(passed), phase))
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
    score = (3 * float(np.mean(recent)) + .3 * max(values)
             + 8 * min(longest * DT, 1.) + 30 * min(tail * DT, 1.))
    return {'score': score, 'valid': True,
            'strict_tail_s': tail * DT,
            'partial_longest_s': longest * DT,
            'best_partial': best_partial, 'final': m,
            'peaks': sim.peaks.copy()}, trace


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
    for ramp_s, neck_rad, hip_rad in itertools.product(
            (.3, .5, .8, 1.2, 2.), (0., .3, .6), (-.3, 0., .3)):
        result, _ = rollout(sim, peak, ramp_s, neck_rad, hip_rad)
        rows.append({'ramp_s': ramp_s, 'neck_rad': neck_rad,
                     'right_hip_pitch_rad': hip_rad, 'result': result})
    ranked = sorted(rows, key=lambda row: -row['result']['score'])
    winner = ranked[0]
    replay_prefix, _, replay_peak, _ = capture(sim, reference, r42, r44)
    if replay_peak['hash'] != peak['hash']:
        raise RuntimeError('Actual-fall replay did not reproduce the branch')
    replay, trace = rollout(sim, replay_peak, winner['ramp_s'],
                            winner['neck_rad'], winner['right_hip_pitch_rad'], True)
    if abs(replay['score'] - winner['result']['score']) > 1e-5:
        raise RuntimeError('Winner did not reproduce from the actual fall')
    args.output.mkdir(parents=True)
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
              'source_sha256': digest(__file__), 'prefix': prefix,
              'peak': {k: v for k, v in peak.items()
                       if k not in ('snapshot', 'peaks', 'finite')},
              'candidate_count': len(rows),
              'valid_count': sum(row['result']['valid'] for row in rows),
              'winner': winner, 'winner_full_fall_replay': replay,
              'all': rows, 'full_task_completed': False}
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    for row in ranked[:10]:
        m = row['result']['final']
        print(row['ramp_s'], row['neck_rad'], row['right_hip_pitch_rad'],
              'score', round(row['result']['score'], 2),
              'partial', row['result']['partial_longest_s'],
              'strict', row['result']['strict_tail_s'],
              'up', round(m['up_z'], 3), flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()

