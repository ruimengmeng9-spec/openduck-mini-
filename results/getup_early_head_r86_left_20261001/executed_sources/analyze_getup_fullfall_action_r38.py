"""Inspect deployed actor actions from actual fallen starts, without training.

The original scene, target slew, motor limits and strict physical audit remain
unchanged. This is a failure diagnostic, not a hardware deployment test.
"""
import argparse
from pathlib import Path
import json

import numpy as np
import onnxruntime as ort

from diagnostics.getup_fullfallen_contract_r32 import decode_action
from diagnostics.getup_fullfallen_env_r32 import observation
from diagnostics.getup_independent_native import DT, POSES, digest
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def replay(sim, session, pose, seed, seconds):
    sim.clear_audit()
    sim.prepare(pose, seed, True)
    initial = sim.measure()
    initial_valid = sim.physical_valid()
    if not initial_valid or initial['up_z'] >= .5 or not initial['torso_contact']:
        raise RuntimeError(f'Invalid actual fallen reset: {pose}, {seed}, {initial}, {sim.peaks}')
    sim.clear_audit()
    rows, qpos, qvel = [], [], []
    max_up, max_load = initial['up_z'], initial['foot_load_fraction']
    physical_valid, strict_tail = True, 0
    for decision in range(round(seconds / (5 * DT))):
        x = observation(sim)[None]
        action = session.run(None, {session.get_inputs()[0].name: x})[0].reshape(-1)
        target = decode_action(action, sim.lower, sim.upper)
        for _ in range(5):
            sim.step_target(target)
            m = sim.measure()
            physical_valid = sim.physical_valid()
            strict_tail = strict_tail + 1 if physical_valid and m['stable'] else 0
            max_up = max(max_up, m['up_z'])
            max_load = max(max_load, m['foot_load_fraction'])
            qpos.append(sim.data.qpos.copy())
            qvel.append(sim.data.qvel.copy())
            if not physical_valid:
                break
        rows.append({'time_s': round((decision + 1) * 5 * DT, 3),
                     'action': action.tolist(),
                     'target_delta_home_rad': (target - sim.home).tolist(),
                     'up_z': m['up_z'], 'height_m': m['height_m'],
                     'torso_contact': m['torso_contact'],
                     'foot_load_fraction': m['foot_load_fraction'],
                     'strict_standing': m['stable'],
                     'physical_valid': physical_valid})
        if not physical_valid:
            break
    summary = {'pose': pose, 'seed': seed, 'initial': initial,
               'decisions': len(rows), 'controls': len(qpos),
               'physical_valid': physical_valid,
               'strict_tail_s': strict_tail * DT,
               'max_up_z': max_up, 'max_foot_load_fraction': max_load,
               'max_abs_normalized_action': max(
                   max(abs(v) for v in row['action']) for row in rows),
               'max_abs_target_delta_home_rad': max(
                   max(abs(v) for v in row['target_delta_home_rad']) for row in rows),
               'final': m, 'peaks': sim.peaks.copy(), 'rows': rows}
    return summary, np.stack(qpos), np.stack(qvel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1300000)
    p.add_argument('--seconds', type=float, default=6.)
    a = p.parse_args()
    if a.seconds < 5 * DT or a.output.exists():
        p.error('Need at least one decision interval and a new output directory')
    scene = a.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = a.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(a.model), sess_options=opts,
                                   providers=['CPUExecutionProvider'])
    a.output.mkdir(parents=True)
    records = []
    for pose in POSES:
        row, qpos, qvel = replay(sim, session, pose, a.seed, a.seconds)
        np.savez_compressed(a.output / f'{pose}.npz', qpos=qpos, qvel=qvel)
        records.append(row)
        print(pose, 'max_up_z', round(row['max_up_z'], 3),
              'max_foot_load', round(row['max_foot_load_fraction'], 3),
              'physical_valid', row['physical_valid'], flush=True)
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene), 'model_sha256': digest(a.model),
              'source_sha256': digest(__file__), 'duration_s': a.seconds,
              'records': records}
    (a.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', a.output, flush=True)


if __name__ == '__main__':
    main()
