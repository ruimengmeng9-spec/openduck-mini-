"""Search a physical body-to-feet support transfer after an R27 full-fall path.

Candidate branching restores a development snapshot only for efficient search.
Every reported full-path result must replay the original path from actual fall.
No root state is edited after initialization; model and motor limits are fixed.
"""
import argparse
from pathlib import Path
import json

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.train_getup_fullpath_r27 import rollout
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def score(m):
    foot_force = min(m['foot_normal_forces_n'])
    foot_flat = min(m['foot_up_alignment_to_home'])
    return (4 * m['up_z'] + 2 * foot_flat + 2 * m['foot_load_fraction']
            + .1 * min(foot_force, 20.) - 2 * m['torso_contact']
            + 5 * m['stable'])


def trial(sim, snapshot, peaks, finite, target, pulse_s, home_s):
    sim.restore(snapshot)
    sim.peaks, sim.finite = peaks.copy(), finite
    best = sim.measure()
    best_score = score(best)
    rows = []
    for phase, duration, command in [('pulse', pulse_s, target),
                                     ('return_home', home_s, sim.home)]:
        for n in range(round(duration / DT)):
            sim.step_target(command)
            m = sim.measure()
            valid = sim.physical_valid()
            if valid and score(m) > best_score:
                best, best_score = m, score(m)
            if n % 10 == 0 or not valid:
                rows.append({'phase': phase, 'time_s': float(sim.data.time),
                             'up_z': m['up_z'],
                             'foot_load_fraction': m['foot_load_fraction'],
                             'foot_normal_forces_n': m['foot_normal_forces_n'],
                             'torso_contact': m['torso_contact'],
                             'strict_standing': m['stable'], 'valid': valid})
            if not valid:
                return {'valid': False, 'best_score': -1e3, 'best': best,
                        'final': m, 'peaks': sim.peaks.copy(), 'rows': rows}
    return {'valid': True, 'best_score': best_score, 'best': best,
            'final': m, 'peaks': sim.peaks.copy(), 'rows': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pose', default='left_side')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        p.error('A new output directory is required')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    ref = np.load(args.reference, allow_pickle=False)
    targets, durations = ref['targets'], ref['durations_s']
    sim = StrictSim(scene, stand)
    prefix, _ = rollout(sim, args.pose, targets, durations, args.seed,
                        perturb=False, hold_s=31.)
    if not prefix['valid'] or not prefix['initial_fallen']:
        raise RuntimeError(f'Prefix is not a physically valid full-fall replay: {prefix}')
    snapshot, peaks, finite = sim.snapshot(), sim.peaks.copy(), sim.finite
    base = sim.measure()
    rows = []
    for joint in range(sim.model.nu):
        for offset in (-.6, -.3, -.15, .15, .3, .6):
            target = sim.home.copy()
            target[joint] = np.clip(target[joint] + offset,
                                    sim.lower[joint], sim.upper[joint])
            if abs(target[joint] - sim.home[joint]) < .05:
                continue
            for pulse_s in (.4, .8):
                result = trial(sim, snapshot, peaks, finite, target, pulse_s, 2.)
                row = {'joint': sim.model.actuator(joint).name,
                       'joint_index': joint, 'offset_rad': offset,
                       'pulse_s': pulse_s, 'result': result}
                rows.append(row)
    ranked = sorted(rows, key=lambda r: r['result']['best_score'], reverse=True)
    args.output.mkdir(parents=True)
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'source_sha256': digest(__file__),
              'pose': args.pose, 'seed': args.seed,
              'prefix': prefix, 'base': base,
              'candidate_count': len(rows), 'valid_count': sum(r['result']['valid'] for r in rows),
              'top': ranked[:20], 'all': rows}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in ranked[:10]:
        r = row['result']
        print(row['joint'], row['offset_rad'], row['pulse_s'],
              'valid', r['valid'], 'best_up', round(r['best']['up_z'], 3),
              'best_load', round(r['best']['foot_load_fraction'], 3),
              'best_score', round(r['best_score'], 3), flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
