"""Search brief joint-target pulses at the first failing partial-tilt boundary.

Every candidate begins at the same physically generated state and then uses
unaltered MuJoCo, actuator and 50 Hz slew limits. A partial tilt is not an
actual fallen start; this search is for a curriculum bridge, not completion.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, POSES, digest
from diagnostics.probe_getup_partial_curriculum_r35 import prepare_partial
from diagnostics.search_getup_joint_sequences_r34 import simulate
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_curriculum_r35 import BOUNDARY
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def candidate_targets(sim, amplitudes):
    yield 'home', -1, 0., sim.home.copy()
    for joint in range(14):
        for amplitude in amplitudes:
            for sign in (-1, 1):
                target = sim.home.copy()
                target[joint] = np.clip(sim.home[joint] + sign * amplitude,
                                        sim.lower[joint], sim.upper[joint])
                yield f'j{joint}_{sign:+d}_{amplitude:g}', joint, sign * amplitude, target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--poses', nargs='+', choices=POSES, default=['prone', 'supine'])
    p.add_argument('--amplitudes', nargs='+', type=float, default=[.3, .6])
    p.add_argument('--pulse-s', nargs='+', type=float, default=[.2, .5])
    p.add_argument('--duration-s', type=float, default=4.)
    p.add_argument('--seed', type=int, default=1300000)
    args = p.parse_args()
    if args.duration_s < 1. or any(x <= 0. for x in args.amplitudes + args.pulse_s):
        p.error('Positive durations and amplitudes required')
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    results = {}
    total = round(args.duration_s / DT)
    for pose in args.poses:
        initial, initial_hash, valid, _ = prepare_partial(
            sim, pose, BOUNDARY[pose], 5, args.seed)
        if not valid:
            raise RuntimeError(f'Invalid initial physical state for {pose}')
        start = sim.snapshot()
        rows = []
        for name, joint, offset, target in candidate_targets(sim, args.amplitudes):
            for pulse_s in args.pulse_s:
                pulse = round(pulse_s / DT)
                if pulse >= total:
                    raise ValueError('Pulse must be shorter than episode')
                row, _ = simulate(sim, start, initial_hash, target, sim.home,
                                  (pulse, 0, total - pulse), record=False)
                row.update(name=name, joint=joint, target_offset_rad=offset,
                           pulse_s=pulse * DT,
                           strict_1s=bool(row['physically_valid'] and
                                          row['longest_strict_standing_s'] >= 1.))
                rows.append(row)
        def rank(row):
            return (int(row['strict_1s']), row['longest_strict_standing_s'],
                    int(row['physically_valid']), row['best_progress'], row['best_up_z'])
        winner = max(rows, key=rank)
        winning_target = sim.home.copy()
        if winner['joint'] >= 0:
            j = winner['joint']
            winning_target[j] = np.clip(sim.home[j] + winner['target_offset_rad'],
                                        sim.lower[j], sim.upper[j])
        pulse = round(winner['pulse_s'] / DT)
        replay, trace = simulate(sim, start, initial_hash, winning_target, sim.home,
                                 (pulse, 0, total - pulse), record=True)
        if replay['longest_strict_standing_s'] != winner['longest_strict_standing_s']:
            raise RuntimeError('Best pulse failed deterministic replay')
        np.savez_compressed(args.output / f'{pose}_best.npz',
                            time=np.asarray(trace['time']),
                            qpos=np.asarray(trace['qpos']), qvel=np.asarray(trace['qvel']),
                            ctrl=np.asarray(trace['ctrl']),
                            phase=np.asarray(trace['phase'], dtype='U8'))
        results[pose] = {'tilt_fraction': BOUNDARY[pose], 'seed': args.seed,
                         'initial_hash': initial_hash, 'initial': initial,
                         'trials': len(rows),
                         'strict_1s_discoveries': sum(r['strict_1s'] for r in rows),
                         'best': winner, 'all_trials': rows}
        print(pose, 'strict_1s', results[pose]['strict_1s_discoveries'],
              'of', len(rows), 'best', winner['name'], winner['pulse_s'],
              'best_up', round(winner['best_up_z'], 3), flush=True)
        (args.output / 'results.json').write_text(json.dumps({
            'experiment': 'R36 partial-boundary pulse search',
            'created_at': datetime.now().isoformat(),
            'scene_sha256': digest(scene), 'source_sha256': digest(__file__),
            'simulation_only': True, 'hardware_readiness': False,
            'full_fall_recovery_claim': False, 'results': results}, indent=2),
            encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
