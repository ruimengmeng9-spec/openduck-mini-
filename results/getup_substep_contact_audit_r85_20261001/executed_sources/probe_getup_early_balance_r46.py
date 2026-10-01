"""Probe the first 0.1 s of balance from an actually replayed near-upright fall.

This is a motor-response diagnosis, not a recovered-standing policy. The
physics scene, target slew, servo and collision audits are unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_loaded_balance_r45 import capture
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def probe(sim, peak, target):
    sim.restore(peak['snapshot'])
    sim.peaks, sim.finite = peak['peaks'].copy(), peak['finite']
    measurements = []
    valid = True
    for _ in range(round(.1 / DT)):
        sim.step_target(target)
        m = sim.measure()
        measurements.append(m)
        valid = sim.physical_valid()
        if not valid:
            break
    final = measurements[-1]
    score = (8 * final['up_z'] + 3 * final['height_m'] / .17
             + 2 * final['foot_load_fraction']
             + min(final['foot_up_alignment_to_home'])
             - 4 * final['angular_speed_rad_s']
             - 5 * final['torso_contact']) if valid else -100.
    return {'score': score, 'valid': valid, 'after_0p1s': final,
            'max_angular_speed_rad_s': max(m['angular_speed_rad_s']
                                          for m in measurements),
            'peaks': sim.peaks.copy()}


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
    baseline = []
    rows = []
    for name, base in [('home', sim.home),
                       ('current', peak['snapshot']['prev'])]:
        base = base.copy()
        baseline.append({'name': name, 'result': probe(sim, peak, base)})
        for joint in range(sim.model.nu):
            for offset in (-.6, -.3, -.15, .15, .3, .6):
                target = base.copy()
                target[joint] = np.clip(target[joint] + offset,
                                        sim.lower[joint], sim.upper[joint])
                if abs(target[joint] - base[joint]) < .05:
                    continue
                result = probe(sim, peak, target)
                rows.append({'base': name,
                             'joint': sim.model.actuator(joint).name,
                             'joint_index': joint,
                             'offset_rad': offset, 'result': result})
    ranked = sorted(rows, key=lambda x: -x['result']['score'])
    winner = ranked[0]
    replay_prefix, _, replay_peak, _ = capture(sim, reference, r42, r44)
    if replay_peak['hash'] != peak['hash']:
        raise RuntimeError('Actual-fall replay did not reproduce the branch')
    base = (sim.home if winner['base'] == 'home'
            else replay_peak['snapshot']['prev']).copy()
    j = winner['joint_index']
    base[j] = np.clip(base[j] + winner['offset_rad'], sim.lower[j], sim.upper[j])
    replay = probe(sim, replay_peak, base)
    if abs(replay['score'] - winner['result']['score']) > 1e-5:
        raise RuntimeError('Winner changed after actual-fall replay')
    args.output.mkdir(parents=True)
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene),
              'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'r42_checkpoint_sha256': digest(args.r42_checkpoint),
              'r44_checkpoint_sha256': digest(args.r44_checkpoint),
              'source_sha256': digest(__file__),
              'prefix': prefix,
              'peak': {k: v for k, v in peak.items()
                       if k not in ('snapshot', 'peaks', 'finite')},
              'candidate_count': len(rows),
              'valid_count': sum(row['result']['valid'] for row in rows),
              'baseline': baseline, 'winner': winner,
              'winner_full_fall_replay': replay,
              'top': ranked[:20], 'all': rows,
              'full_task_completed': False}
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    for row in ranked[:10]:
        m = row['result']['after_0p1s']
        print(row['base'], row['joint'], row['offset_rad'],
              'spin', round(m['angular_speed_rad_s'], 3),
              'up', round(m['up_z'], 3),
              'height', round(m['height_m'], 3),
              'load', round(m['foot_load_fraction'], 2), flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()

