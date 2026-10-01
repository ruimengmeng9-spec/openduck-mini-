"""Pair two physically valid pulse targets at a partial-tilt boundary.

Selection is based on single-joint probe results, but every pair is replayed
from the same initial state with unchanged original motor and contact physics.
This is curriculum-bridge search, not full-fall acceptance.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.probe_getup_partial_curriculum_r35 import prepare_partial
from diagnostics.search_getup_joint_sequences_r34 import simulate
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_curriculum_r35 import BOUNDARY
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def valid_singles(pose_result, count):
    rows = [r for r in pose_result['all_trials']
            if r['physically_valid'] and r['joint'] >= 0]
    rows.sort(key=lambda r: (r['longest_strict_standing_s'],
                             r['best_up_z'], r['best_progress']), reverse=True)
    chosen = []
    seen = set()
    for row in rows:
        if row['name'] in seen:
            continue
        seen.add(row['name'])
        chosen.append(row)
        if len(chosen) == count:
            break
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--pulse-s', type=float, nargs='+', default=[.2, .5])
    p.add_argument('--duration-s', type=float, default=4.)
    args = p.parse_args()
    if args.count < 2 or args.duration_s <= max(args.pulse_s):
        p.error('At least two candidates and pulse shorter than duration required')
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source.read_text())['results']
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    results = {}
    total = round(args.duration_s / DT)
    for pose, source_pose in source.items():
        initial, initial_hash, valid, _ = prepare_partial(
            sim, pose, BOUNDARY[pose], 5, source_pose['seed'])
        if not valid or initial_hash != source_pose['initial_hash']:
            raise RuntimeError('Paired search reset does not match source')
        start = sim.snapshot()
        singles = valid_singles(source_pose, args.count)
        rows = []
        for i, first in enumerate(singles):
            for second in singles[i + 1:]:
                if first['joint'] == second['joint']:
                    continue
                target = sim.home.copy()
                for item in (first, second):
                    j = item['joint']
                    target[j] = np.clip(sim.home[j] + item['target_offset_rad'],
                                        sim.lower[j], sim.upper[j])
                for pulse_s in args.pulse_s:
                    pulse = round(pulse_s / DT)
                    row, _ = simulate(sim, start, initial_hash, target, sim.home,
                                      (pulse, 0, total - pulse), record=False)
                    row.update(first=first['name'], second=second['name'],
                               joints=[first['joint'], second['joint']],
                               offsets_rad=[first['target_offset_rad'],
                                            second['target_offset_rad']],
                               pulse_s=pulse * DT,
                               strict_1s=bool(row['physically_valid'] and
                                              row['longest_strict_standing_s'] >= 1.))
                    rows.append(row)
        if not rows:
            raise RuntimeError(f'No distinct joint pairs for {pose}')
        def rank(row):
            return (int(row['strict_1s']), row['longest_strict_standing_s'],
                    int(row['physically_valid']), row['best_up_z'], row['best_progress'])
        winner = max(rows, key=rank)
        target = sim.home.copy()
        for j, offset in zip(winner['joints'], winner['offsets_rad']):
            target[j] = np.clip(sim.home[j] + offset, sim.lower[j], sim.upper[j])
        pulse = round(winner['pulse_s'] / DT)
        replay, trace = simulate(sim, start, initial_hash, target, sim.home,
                                 (pulse, 0, total - pulse), record=True)
        if replay['longest_strict_standing_s'] != winner['longest_strict_standing_s']:
            raise RuntimeError('Pair pulse failed deterministic replay')
        np.savez_compressed(args.output / f'{pose}_best.npz',
                            time=np.asarray(trace['time']),
                            qpos=np.asarray(trace['qpos']), qvel=np.asarray(trace['qvel']),
                            ctrl=np.asarray(trace['ctrl']),
                            phase=np.asarray(trace['phase'], dtype='U8'))
        results[pose] = {'initial_hash': initial_hash, 'initial': initial,
                         'trials': len(rows),
                         'strict_1s_discoveries': sum(r['strict_1s'] for r in rows),
                         'best': winner, 'all_trials': rows}
        print(pose, 'strict_1s', results[pose]['strict_1s_discoveries'],
              'of', len(rows), 'best', winner['first'], winner['second'],
              'tail', winner['longest_strict_standing_s'], flush=True)
        (args.output / 'results.json').write_text(json.dumps({
            'experiment': 'R36 partial-boundary joint pair search',
            'created_at': datetime.now().isoformat(),
            'source_sha256': digest(args.source),
            'scene_sha256': digest(scene), 'script_sha256': digest(__file__),
            'simulation_only': True, 'hardware_readiness': False,
            'full_fall_recovery_claim': False, 'results': results}, indent=2),
            encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
