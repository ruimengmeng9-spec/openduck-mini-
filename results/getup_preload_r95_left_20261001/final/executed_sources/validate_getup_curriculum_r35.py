"""Held-out deterministic curriculum and actual-full-fall validation.

Partial tilts are diagnostic only. They never count as full-fall get-up.
No root edits occur after an episode begins, and original audited motor
constraints and the strict loaded-standing gate remain in force.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from diagnostics.getup_fullfallen_contract_r32 import decode_action
from diagnostics.getup_fullfallen_env_r32 import observation
from diagnostics.getup_independent_native import DT, POSES, digest
from diagnostics.probe_getup_partial_curriculum_r35 import prepare_partial
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


BOUNDARY = {'prone': .20, 'supine': .10,
            'left_side': .25, 'right_side': .25}


def test_case(sim, session, pose, fraction, seed, controls, record=False):
    if pose == 'standing':
        sim.clear_audit()
        sim.prepare('standing', seed, True)
        initial = sim.measure()
        initial_valid = sim.physical_valid()
    elif fraction == 1.:
        sim.clear_audit()
        sim.prepare(pose, seed, True)
        initial = sim.measure()
        initial_valid = sim.physical_valid()
    else:
        initial, _, initial_valid, _ = prepare_partial(sim, pose, fraction, 5, seed)
    start_hash = state_hash(sim)
    sim.clear_audit()
    tail = 0
    trace = []
    valid = initial_valid
    m = initial
    total_controls = 0
    if valid:
        for _ in range((controls + 4) // 5):
            obs = observation(sim)[None]
            action = session.run(None, {session.get_inputs()[0].name: obs})[0][0]
            target = decode_action(action, sim.lower, sim.upper)
            for _ in range(5):
                if total_controls == controls:
                    break
                sim.step_target(target)
                total_controls += 1
                m = sim.measure()
                valid = sim.physical_valid()
                tail = tail + 1 if valid and m['stable'] else 0
                if record:
                    trace.append((sim.data.qpos.copy(), sim.data.qvel.copy(),
                                  float(sim.data.time), int(m['stable'])))
                if not valid:
                    break
            if not valid:
                break
    result = {'pose': pose, 'tilt_fraction': fraction, 'seed': seed,
              'initial_hash': start_hash, 'initial_valid': initial_valid,
              'initial': initial, 'actual_fallen_start': bool(
                  pose != 'standing' and fraction == 1. and
                  initial['up_z'] < .5 and initial['torso_contact']),
              'controls': total_controls, 'valid': valid,
              'strict_tail_s': round(tail * DT, 6),
              'strict_1s': bool(valid and tail * DT >= 1.),
              'final': m, 'peaks': sim.peaks.copy()}
    return result, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[1300000, 1300001, 1300002])
    p.add_argument('--duration-s', type=float, default=6.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=opts,
                                   providers=['CPUExecutionProvider'])
    controls = round(args.duration_s / DT)
    rows = []
    for pose, fractions in [('standing', [0.])] + [
            (pose, [BOUNDARY[pose], 1.]) for pose in POSES]:
        for fraction in fractions:
            for seed in args.seeds:
                row, trace = test_case(sim, session, pose, fraction, seed,
                                       controls, record=(seed == args.seeds[0]))
                rows.append(row)
                if trace and seed == args.seeds[0] and not row['strict_1s']:
                    name = f'{pose}_{str(fraction).replace(".", "p")}_failure.npz'
                    np.savez_compressed(args.output / name,
                                        qpos=np.stack([x[0] for x in trace]),
                                        qvel=np.stack([x[1] for x in trace]),
                                        time=np.array([x[2] for x in trace]),
                                        stable=np.array([x[3] for x in trace]))
                print(pose, fraction, seed, 'initial_up',
                      round(row['initial']['up_z'], 3), 'strict', row['strict_1s'],
                      'final_up', round(row['final']['up_z'], 3), flush=True)
    summary = {}
    for pose in POSES:
        full = [r for r in rows if r['pose'] == pose and r['tilt_fraction'] == 1.]
        boundary = [r for r in rows if r['pose'] == pose and
                    r['tilt_fraction'] == BOUNDARY[pose]]
        summary[pose] = {'actual_full_fall_1s': sum(r['strict_1s'] for r in full),
                         'actual_full_fall_trials': len(full),
                         'partial_boundary_1s': sum(r['strict_1s'] for r in boundary),
                         'partial_boundary_trials': len(boundary)}
    standing = [r for r in rows if r['pose'] == 'standing']
    summary['standing'] = {'strict_1s': sum(r['strict_1s'] for r in standing),
                           'trials': len(standing)}
    report = {'experiment': 'R35 held-out deterministic diagnostic',
              'created_at': datetime.now().isoformat(),
              'model_sha256': digest(args.model), 'scene_sha256': digest(scene),
              'source_sha256': digest(__file__), 'simulation_only': True,
              'hardware_readiness': False, 'full_task_completed': False,
              'training_seed_namespace_disjoint': True,
              'short_1s_discovery_is_not_30s_acceptance': True,
              'summary': summary, 'rows': rows}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
