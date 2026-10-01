"""Test whether the R40 momentary two-foot support can be sustained.

The starting transient is physically replayed from a left-side full fall.
No root teleports, altered motor limits, or relaxed standing labels are used.
"""
import argparse
from pathlib import Path
import json

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_terminal_bridge_r39 import score, trial
from diagnostics.train_getup_fullpath_r27 import rollout
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output directory already exists')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    ref = np.load(args.reference, allow_pickle=False)
    sim = StrictSim(scene, stand)
    prefix, _ = rollout(sim, 'left_side', ref['targets'], ref['durations_s'],
                        args.seed, perturb=False, hold_s=31.)
    if not prefix['valid'] or not prefix['initial_fallen']:
        raise RuntimeError('Invalid actual-fall prefix')
    first = sim.home.copy()
    first_joint = sim.model.actuator('right_hip_pitch').id
    second_joint = sim.model.actuator('left_hip_pitch').id
    first[first_joint] = np.clip(first[first_joint] - .6,
                                 sim.lower[first_joint], sim.upper[first_joint])
    captured = None
    for phase, duration, target in [('first', .4, first), ('return', 2., sim.home)]:
        for control in range(round(duration / DT)):
            sim.step_target(target)
            m = sim.measure()
            if not sim.physical_valid():
                raise RuntimeError('First pulse invalid')
            if (m['up_z'] > .45 and not m['torso_contact']
                    and min(m['foot_normal_forces_n']) > 5.):
                if captured is None or score(m) > captured['score']:
                    captured = {'phase': phase, 'control_index': control,
                                'time_s': float(sim.data.time), 'score': score(m),
                                'metric': m, 'snapshot': sim.snapshot(),
                                'peaks': sim.peaks.copy(), 'finite': sim.finite}
    if captured is None:
        raise RuntimeError('Two-foot transient not reproduced')
    snapshot, peaks, finite = (captured.pop('snapshot'),
                               captured.pop('peaks'), captured.pop('finite'))
    rows = []
    for offset in (.3, .6):
        target = first.copy()
        target[second_joint] = np.clip(target[second_joint] + offset,
                                       sim.lower[second_joint], sim.upper[second_joint])
        for hold_s in (.8, 1.5, 3., 5.):
            result = trial(sim, snapshot, peaks, finite, target, hold_s, 3.)
            rows.append({'second_joint_offset_rad': offset,
                         'second_hold_s': hold_s, 'result': result})
            print('offset', offset, 'hold', hold_s,
                  'valid', result['valid'],
                  'best_up', round(result['best']['up_z'], 3),
                  'final_up', round(result['final']['up_z'], 3),
                  'final_torso', result['final']['torso_contact'],
                  'strict', result['final']['stable'], flush=True)
    args.output.mkdir(parents=True)
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene),
              'reference_sha256': digest(args.reference),
              'source_sha256': digest(__file__), 'seed': args.seed,
              'prefix': prefix, 'captured': captured, 'rows': rows}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
