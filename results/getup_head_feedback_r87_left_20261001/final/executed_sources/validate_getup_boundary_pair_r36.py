"""Held-out physical replay of one R36 paired-joint pulse.

Reports partial-tilt results separately from fully fallen starts. Never
promotes a partial-tilt success to the user's full get-up acceptance gate.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, POSES, digest
from diagnostics.probe_getup_partial_curriculum_r35 import prepare_partial
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def evaluate(sim, pose, fraction, seed, offsets, pulse_controls, duration_controls,
             record=False):
    if fraction == 1.:
        sim.prepare(pose, seed, True)
        initial = sim.measure()
        initial_valid = sim.physical_valid()
    else:
        initial, _, initial_valid, _ = prepare_partial(sim, pose, fraction, 5, seed)
    initial_hash = state_hash(sim)
    sim.clear_audit()
    target = sim.home.copy()
    for joint, offset in offsets:
        target[joint] = np.clip(sim.home[joint] + offset,
                                sim.lower[joint], sim.upper[joint])
    tail = 0
    longest = 0
    entry = False
    valid = initial_valid
    trace = []
    m = initial
    controls = 0
    if valid:
        for index in range(duration_controls):
            sim.step_target(target if index < pulse_controls else sim.home)
            controls += 1
            m = sim.measure()
            valid = sim.physical_valid()
            tail = tail + 1 if valid and m['stable'] else 0
            longest = max(longest, tail)
            entry |= bool(valid and m['stable'])
            if record:
                trace.append((sim.data.qpos.copy(), sim.data.qvel.copy(),
                              float(sim.data.time), int(m['stable'])))
            if not valid:
                break
    row = {'pose': pose, 'tilt_fraction': fraction, 'seed': seed,
           'initial_hash': initial_hash, 'initial': initial,
           'initial_physical_valid': initial_valid,
           'actual_fallen_start': bool(fraction == 1. and
                                       initial['up_z'] < .5 and initial['torso_contact']),
           'controls': controls, 'physical_valid': valid,
           'entry_reached': entry,
           'strict_tail_s': round(tail * DT, 6),
           'longest_strict_s': round(longest * DT, 6),
           'strict_1s': bool(valid and tail * DT >= 1.),
           'strict_30s': bool(valid and tail * DT >= 30.),
           'final': m, 'peaks': sim.peaks.copy()}
    return row, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pose', choices=POSES, default='supine')
    p.add_argument('--fractions', type=float, nargs='+', default=[.1, .15, .2, 1.])
    p.add_argument('--seeds', type=int, nargs='+', default=[1300001, 1300002, 1300003])
    p.add_argument('--duration-s', type=float, default=6.)
    args = p.parse_args()
    if args.duration_s < 1. or any(not 0. < f <= 1. for f in args.fractions):
        p.error('Positive duration and fractions in (0,1] required')
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source.read_text())['results'][args.pose]
    best = source['best']
    offsets = list(zip(best['joints'], best['offsets_rad']))
    pulse_controls = round(best['pulse_s'] / DT)
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    rows = []
    for fraction in args.fractions:
        recorded_success = recorded_failure = False
        for seed in args.seeds:
            row, trace = evaluate(sim, args.pose, fraction, seed, offsets,
                                  pulse_controls, round(args.duration_s / DT),
                                  record=True)
            rows.append(row)
            kind = 'success' if row['strict_1s'] else 'failure'
            if trace and ((kind == 'success' and not recorded_success) or
                          (kind == 'failure' and not recorded_failure)):
                name = f'{args.pose}_f{str(fraction).replace(".", "p")}_{kind}.npz'
                np.savez_compressed(args.output / name,
                                    qpos=np.stack([x[0] for x in trace]),
                                    qvel=np.stack([x[1] for x in trace]),
                                    time=np.array([x[2] for x in trace]),
                                    stable=np.array([x[3] for x in trace]))
                if kind == 'success':
                    recorded_success = True
                else:
                    recorded_failure = True
            print(args.pose, fraction, seed, 'actual_fall', row['actual_fallen_start'],
                  'valid', row['physical_valid'], 'tail', row['strict_tail_s'],
                  'final_up', round(row['final']['up_z'], 3), flush=True)
    summary = {}
    for fraction in args.fractions:
        trials = [r for r in rows if r['tilt_fraction'] == fraction]
        summary[str(fraction)] = {
            'trials': len(trials),
            'strict_1s': sum(r['strict_1s'] for r in trials),
            'strict_30s': sum(r['strict_30s'] for r in trials),
            'actual_fallen_starts': sum(r['actual_fallen_start'] for r in trials),
        }
    report = {'experiment': 'R36 held-out physical pair replay',
              'created_at': datetime.now().isoformat(),
              'source_sha256': digest(args.source), 'scene_sha256': digest(scene),
              'script_sha256': digest(__file__), 'simulation_only': True,
              'hardware_readiness': False, 'full_task_completed': False,
              'partial_tilt_success_not_full_fall_success': True,
              'pose': args.pose, 'joints_offsets_rad': offsets,
              'pulse_s': best['pulse_s'], 'duration_s': args.duration_s,
              'summary': summary, 'rows': rows}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
