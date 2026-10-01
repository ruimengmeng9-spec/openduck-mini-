"""Find a catch action after R42's physically reached partial rotation.

Every scored branch starts from a snapshot reached from an actual left-side
fall under unchanged scene and motor physics. The selected branch is replayed
from that fall and saved as a failure/success trace, never accepted on an
intermediate snapshot alone.
"""
import argparse
from pathlib import Path
import json

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_sustained_bridge_r42 import (
    JOINTS, DURATIONS, capture, moment_quality)
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def reach_peak(sim, reference, offsets, seed):
    prefix, initial = capture(sim, reference, seed)
    sim.restore(initial['snapshot'])
    sim.peaks, sim.finite = initial['peaks'].copy(), initial['finite']
    ids = np.array([sim.model.actuator(n).id for n in JOINTS])
    best = None
    for phase, (duration, row) in enumerate(zip(DURATIONS, offsets)):
        target = sim.home.copy()
        target[ids] = np.clip(sim.home[ids] + row,
                              sim.lower[ids], sim.upper[ids])
        for control in range(round(duration / DT)):
            sim.step_target(target)
            m = sim.measure()
            if not sim.physical_valid():
                raise RuntimeError('R42 winning path is no longer physical')
            if best is None or m['up_z'] > best['metric']['up_z']:
                best = {'phase': phase, 'control_index': control,
                        'time_s': float(sim.data.time), 'metric': m,
                        'snapshot': sim.snapshot(), 'peaks': sim.peaks.copy(),
                        'finite': sim.finite, 'hash': state_hash(sim)}
    return prefix, initial, best


def branch(sim, peak, target, duration, record=False):
    sim.restore(peak['snapshot'])
    sim.peaks, sim.finite = peak['peaks'].copy(), peak['finite']
    max_up, max_quality = -1., -1e3
    best_up = sim.measure()
    qualities, strict = [], []
    trace = []
    valid = True
    for phase, seconds, command in [('catch', duration, target),
                                    ('return_home', 1.5, sim.home)]:
        for _ in range(round(seconds / DT)):
            sim.step_target(command)
            m = sim.measure()
            valid = sim.physical_valid()
            if not valid:
                break
            q = moment_quality(m)
            qualities.append(q)
            strict.append(bool(m['stable']))
            if m['up_z'] > max_up:
                max_up, best_up = m['up_z'], m
            max_quality = max(max_quality, q)
            if record:
                trace.append((float(sim.data.time), sim.data.qpos.copy(),
                              sim.data.qvel.copy(), int(m['stable']), phase))
        if not valid:
            break
    tail_quality = float(np.mean(qualities[-round(1. / DT):])) if qualities else -100.
    strict_tail = 0
    for passed in reversed(strict):
        if not passed:
            break
        strict_tail += 1
    score = (2 * max_up + max_quality + 2 * tail_quality
             + 4 * min(strict_tail * DT, 1.)) if valid else -100.
    result = {'score': score, 'valid': valid, 'max_up_z': max_up,
              'max_quality': max_quality, 'tail_quality': tail_quality,
              'strict_tail_s': strict_tail * DT,
              'best_up': best_up, 'final': m, 'peaks': sim.peaks.copy()}
    return result, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--r42-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output directory already exists')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    offsets = np.load(args.r42_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, initial, peak = reach_peak(sim, reference, offsets, args.seed)
    rows = []
    current = peak['snapshot']['prev']
    for base_name, base in [('home', sim.home), ('current', current)]:
        for joint in range(sim.model.nu):
            for offset in (-.6, -.3, -.15, .15, .3, .6):
                target = base.copy()
                target[joint] = np.clip(target[joint] + offset,
                                        sim.lower[joint], sim.upper[joint])
                if abs(target[joint] - base[joint]) < .05:
                    continue
                for seconds in (.4, .8):
                    result, _ = branch(sim, peak, target, seconds)
                    rows.append({'base': base_name,
                                 'joint': sim.model.actuator(joint).name,
                                 'joint_index': joint, 'offset_rad': offset,
                                 'duration_s': seconds, 'result': result})
    ranked = sorted(rows, key=lambda item: -item['result']['score'])
    winner = ranked[0]
    # Rebuild the whole left-side path, verify its peak state, then evaluate
    # the same winner again. No manually set intermediate root pose is used.
    replay_prefix, _, replay_peak = reach_peak(sim, reference, offsets, args.seed)
    if replay_peak['hash'] != peak['hash']:
        raise RuntimeError('Actual-fall replay could not reproduce peak state')
    target = (sim.home if winner['base'] == 'home' else replay_peak['snapshot']['prev']).copy()
    j = winner['joint_index']
    target[j] = np.clip(target[j] + winner['offset_rad'], sim.lower[j], sim.upper[j])
    replay, trace = branch(sim, replay_peak, target, winner['duration_s'], True)
    if abs(replay['score'] - winner['result']['score']) > 1e-5:
        raise RuntimeError('Winning branch changed in full-fall replay')
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'winner_trace.npz',
                        time=np.array([x[0] for x in trace]),
                        qpos=np.stack([x[1] for x in trace]),
                        qvel=np.stack([x[2] for x in trace]),
                        strict=np.array([x[3] for x in trace]),
                        phase=np.array([x[4] for x in trace]))
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene),
              'reference_sha256': digest(args.reference),
              'r42_checkpoint_sha256': digest(args.r42_checkpoint),
              'source_sha256': digest(__file__), 'seed': args.seed,
              'prefix': prefix,
              'captured_first': {k: v for k, v in initial.items()
                                 if k not in ('snapshot', 'peaks', 'finite')},
              'peak': {k: v for k, v in peak.items()
                       if k not in ('snapshot', 'peaks', 'finite')},
              'candidate_count': len(rows),
              'valid_count': sum(row['result']['valid'] for row in rows),
              'winner': winner, 'replay': replay,
              'top': ranked[:20], 'all': rows,
              'full_task_completed': False}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in ranked[:10]:
        m = row['result']['best_up']
        print(row['base'], row['joint'], row['offset_rad'], row['duration_s'],
              'up', round(row['result']['max_up_z'], 3),
              'tail', round(row['result']['strict_tail_s'], 3),
              'torso', m['torso_contact'],
              'feet', [round(x, 1) for x in m['foot_normal_forces_n']], flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
