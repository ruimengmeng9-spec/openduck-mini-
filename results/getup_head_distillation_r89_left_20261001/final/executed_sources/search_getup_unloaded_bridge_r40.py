"""Branch one motor pulse at a genuine two-foot, torso-unloaded transient.

The transient is reached by replaying the R27 reference and an R39 pulse
from an actual full fall under unchanged physics. Snapshot restore is only
an efficiency device for development search; no restored state is counted
as an independent or completed get-up.
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
        raise RuntimeError('The actual-fall prefix failed its physical audit')
    first = sim.home.copy()
    joint = sim.model.actuator('right_hip_pitch').id
    first[joint] = np.clip(first[joint] - .6, sim.lower[joint], sim.upper[joint])
    captured = None
    for phase, duration, target in [('right_hip_pitch_pulse', .4, first),
                                    ('return_home', 2., sim.home)]:
        for control in range(round(duration / DT)):
            sim.step_target(target)
            m = sim.measure()
            if not sim.physical_valid():
                raise RuntimeError('First pulse exceeded physical limits')
            if (m['up_z'] > .45 and not m['torso_contact']
                    and min(m['foot_normal_forces_n']) > 5.):
                quality = score(m)
                if captured is None or quality > captured['score']:
                    captured = {'phase': phase, 'control_index': control,
                                'time_s': float(sim.data.time), 'metric': m,
                                'score': quality, 'snapshot': sim.snapshot(),
                                'peaks': sim.peaks.copy(), 'finite': sim.finite}
    if captured is None:
        raise RuntimeError('R39 unloaded two-foot transient was not reproduced')
    snapshot, peaks, finite = (captured.pop('snapshot'),
                               captured.pop('peaks'), captured.pop('finite'))
    rows = []
    for base_name, base in [('home', sim.home), ('first_pulse', first)]:
        for joint in range(sim.model.nu):
            for offset in (-.6, -.3, -.15, .15, .3, .6):
                target = base.copy()
                target[joint] = np.clip(target[joint] + offset,
                                        sim.lower[joint], sim.upper[joint])
                if abs(target[joint] - base[joint]) < .05:
                    continue
                for duration in (.4, .8):
                    result = trial(sim, snapshot, peaks, finite, target,
                                   duration, 2.)
                    rows.append({'base': base_name,
                                 'joint': sim.model.actuator(joint).name,
                                 'joint_index': joint, 'offset_rad': offset,
                                 'pulse_s': duration, 'result': result})
    ranked = sorted(rows, key=lambda r: r['result']['best_score'], reverse=True)
    args.output.mkdir(parents=True)
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene), 'reference_sha256': digest(args.reference),
              'source_sha256': digest(__file__), 'seed': args.seed,
              'prefix': prefix, 'captured': captured,
              'candidate_count': len(rows),
              'valid_count': sum(r['result']['valid'] for r in rows),
              'top': ranked[:20], 'all': rows}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('CAPTURE', captured['phase'], captured['control_index'],
          'up', round(captured['metric']['up_z'], 3),
          'feet_N', [round(x, 2) for x in captured['metric']['foot_normal_forces_n']],
          flush=True)
    for row in ranked[:10]:
        m = row['result']['best']
        print(row['base'], row['joint'], row['offset_rad'], row['pulse_s'],
              'valid', row['result']['valid'], 'up', round(m['up_z'], 3),
              'load', round(m['foot_load_fraction'], 3),
              'torso', m['torso_contact'], 'strict', m['stable'], flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
