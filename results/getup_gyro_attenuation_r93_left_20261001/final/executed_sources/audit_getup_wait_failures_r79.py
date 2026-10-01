"""Offline R77 failure diagnosis; no new rollout, training or robot access.

Position-stage reconstruction supplies geometric contact proximity only.
No contact force, torque, solver state or physical success is invented from
qpos/qvel. Original recorded physical audit and strict flags are authoritative.
The reconstructed command follows the frozen R64/R27 50 Hz implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

from diagnostics.getup_independent_native import DT, SLEW, digest
from diagnostics.probe_getup_terminal_timing_r72 import timed_targets
from diagnostics.probe_getup_wait_library_r77 import change_wait
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS


def longest_run(flags):
    longest = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(run, longest)
    return longest


def first_run(flags, count):
    run = 0
    for index, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= count:
            return index - count + 1
    return None


def up_z(qpos):
    q = np.asarray(qpos)[:, 3:7]
    if not np.allclose(np.linalg.norm(q, axis=1), 1., atol=1e-6):
        raise RuntimeError('Invalid saved root orientation')
    return 1. - 2. * (q[:, 1] ** 2 + q[:, 2] ** 2)


def commands(targets, residuals, ids, home, lower, upper):
    desired = targets.copy()
    desired[:, ids] += residuals
    desired = np.clip(desired, lower, upper)
    previous = home.copy()
    applied = []
    for target in desired:
        previous = np.clip(target, previous - SLEW * DT, previous + SLEW * DT)
        applied.append(previous.copy())
    return desired, np.asarray(applied)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        trace = {key: z[key] for key in z.files}
    n = len(trace['time'])
    if not all(len(trace[key]) == n for key in ('qpos', 'qvel', 'phase', 'strict',
                                              'margin_m', 'imu_error', 'residual_rad')):
        raise RuntimeError('Trace arrays are not aligned')
    trace['root_up_z'] = up_z(trace['qpos'])
    trace['elapsed_s'] = (np.arange(n) + 1) * DT
    return trace


def geometry(model, data, qpos, qvel):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_fwdPosition(model, data)
    floor = model.geom('floor').id
    contacts = set()
    for c in data.contact:
        if floor in c.geom and c.dist <= .001:
            other = int(c.geom[1] if c.geom[0] == floor else c.geom[0])
            contacts.add(model.body(int(model.geom_bodyid[other])).name)
    return {'ground_proximity_bodies': sorted(contacts),
            'feet_geometrically_touching': [name in contacts for name in
                                           ('foot_assembly', 'foot_assembly_2')],
            'torso_geometrically_touching': bool(contacts & {'trunk_assembly', 'head_assembly'}),
            'root_up_z': float(up_z(qpos[None])[0]), 'root_height_m': float(qpos[2])}


def describe(trace, model, data, targets, ids, home, lower, upper, nominal=None):
    n = len(trace['time'])
    desired, applied = commands(targets[:n], trace['residual_rad'], ids, home, lower, upper)
    qa = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    tracking = np.abs(trace['qpos'][:, qa] - applied)
    tilt = np.linalg.norm(trace['imu_error'][:, :2], axis=1)
    divergence = first_run(tilt > .15, 10)
    result = {'maximum_root_up_z': float(trace['root_up_z'].max()),
              'upright_ge_0_95_longest_s': longest_run(trace['root_up_z'] >= .95) * DT,
              'recorded_strict_longest_s': longest_run(trace['strict']) * DT,
              'first_0_2s_reference_tilt_error_over_0_15': None if divergence is None else {
                  'elapsed_s': float(trace['elapsed_s'][divergence]),
                  'phase': int(trace['phase'][divergence])},
              'phases': []}
    if nominal is not None:
        comparable = min(n, len(nominal['time']))
        joint_to_nominal = np.abs(trace['qpos'][:comparable, qa] - nominal['qpos'][:comparable, qa])
    for phase in np.unique(trace['phase']):
        indices = np.flatnonzero(trace['phase'] == phase)
        mask = trace['phase'] == phase
        first, last = indices[0], indices[-1]
        row = {'phase': int(phase), 'start_elapsed_s': float(first * DT),
               'end_elapsed_s': float((last + 1) * DT),
               'start_up_z': float(trace['root_up_z'][first]),
               'end_up_z': float(trace['root_up_z'][last]),
               'max_up_z': float(trace['root_up_z'][mask].max()),
               'mean_imu_tilt_error': float(tilt[mask].mean()),
               'mean_tracking_error_rad': float(tracking[mask].mean()),
               'p95_tracking_error_rad': float(np.percentile(tracking[mask], 95)),
               'maximum_tracking_error_rad': float(tracking[mask].max()),
               'residual_cap_fraction': float(np.any(np.abs(trace['residual_rad'][mask]) >= .18 - 1e-8, axis=1).mean()),
               'slew_active_fraction': float(np.any(np.abs(desired[mask] - applied[mask]) > 1e-8, axis=1).mean()),
               'strict_fraction': float(np.mean(trace['strict'][mask])),
               'saved_support_margin_mean_m': float(trace['margin_m'][mask].mean()),
               'end_geometry': geometry(model, data, trace['qpos'][last], trace['qvel'][last])}
        if nominal is not None and last < comparable:
            row['mean_encoder_difference_from_nominal_rad'] = float(joint_to_nominal[mask[:comparable]].mean())
        result['phases'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--selector', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Need fresh diagnostic output')
    report = json.loads((args.library / 'results.json').read_text())
    scene = Path('/data/shijinsheng/open_duck/training/getup_decomposed_r4/model/scene.xml')
    if digest(scene) != report['scene_sha256'] or digest(args.checkpoint) != report['checkpoint_sha256']:
        raise RuntimeError('Input contract changed')
    with np.load(args.checkpoint, allow_pickle=False) as z:
        ck = {key: z[key] for key in z.files}
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qa = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    home = model.keyframe('home').qpos[qa]
    lower, upper = model.jnt_range[model.actuator_trnid[:, 0]].T
    ids = np.array([model.actuator(name).id for name in JOINTS])
    sim = SimpleNamespace(model=model, home=home, lower=lower, upper=upper)
    eligible = [r for r in report['results'] if r['nominal_success']]
    baseline = eligible[0]
    assert baseline['wait_s'] == 2.
    permanent = [int(seed) for seed, waits in report['successful_waits_by_training_seed'].items() if not waits]
    rescued = [r['seed'] for r in baseline['cases'] if not r['success'] and r['seed'] not in permanent]
    target_map = {r['wait_s']: timed_targets(sim, change_wait(ck, r['wait_s']), (.4, .6, .8), 2.)[0] for r in eligible}
    nominal = load(args.library / 'wait_100/nominal.npz')
    rows, hashes = [], {}
    for row in eligible:
        wait = row['wait_s']
        for entry in row['cases']:
            seed = entry['seed']
            if wait != 2. and seed not in permanent and seed not in rescued:
                continue
            path = args.library / f'wait_{round(wait / DT):03d}' / f'train_{seed}.npz'
            trace = load(path)
            if len(trace['time']) != entry['completed_steps']:
                raise RuntimeError('Saved trace disagrees with original recorded result')
            detail = describe(trace, model, data, target_map[wait], ids, home, lower, upper,
                              nominal if wait == 2. else None)
            rows.append({'seed': seed, 'wait_s_requested': wait,
                         'category': 'all_waits_failed' if seed in permanent else
                                     'rescued_by_other_wait' if seed in rescued else 'baseline_success',
                         'recorded_success': entry['success'], 'recorded_physical_valid': entry['valid'],
                         'recorded_strict_tail_s': entry['strict_tail_s'], **detail})
            hashes[str(path.relative_to(args.library))] = digest(path)
    fit_report = json.loads((args.selector / 'results.json').read_text())
    original_success = {r['seed']: r['success'] for r in baseline['cases']}
    comparisons = []
    for setting in fit_report['cv_settings']:
        chosen = setting['selected_columns']
        rescued_ids, regressed_ids = [], []
        for index, seed in enumerate(report['training_seeds']):
            row = eligible[chosen[index]]
            passed = next(r['success'] for r in row['cases'] if r['seed'] == seed)
            if passed and not original_success[seed]: rescued_ids.append(seed)
            if not passed and original_success[seed]: regressed_ids.append(seed)
        comparisons.append({**setting, 'rescued_seeds': rescued_ids, 'regressed_seeds': regressed_ids})
    result = {'diagnosis_only': True, 'simulation_only': True, 'hardware_readiness': False,
              'no_new_rollout_or_training': True, 'no_controller_or_physics_changes': True,
              'training_data_only': True, 'source_sha256': args.source_sha256,
              'scene_sha256': digest(scene), 'library_result_sha256': digest(args.library / 'results.json'),
              'selector_result_sha256': digest(args.selector / 'results.json'),
              'all_waits_failed_seeds': permanent, 'rescued_by_wait_seeds': rescued,
              'geometry_limit': 'offline post-control proximity only, not reconstructed contact forces or substep audit',
              'tracking_limit': 'reconstructed frozen clipped-and-slew-limited 50Hz targets, not hardware measurements',
              'diagnostic_tilt_threshold_not_acceptance_gate': '.15 reference XY upvector error for .2 s',
              'nominal': describe(nominal, model, data, target_map[2.], ids, home, lower, upper),
              'results': rows, 'selector_cv_paired_changes': comparisons, 'input_trace_hashes': hashes}
    args.output.mkdir(parents=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        if row['wait_s_requested'] == 2. and row['category'] != 'baseline_success':
            ends = {p['phase']: round(p['end_up_z'], 3) for p in row['phases']}
            print(json.dumps({'seed': row['seed'], 'category': row['category'],
                              'strict_longest': row['recorded_strict_longest_s'],
                              'upright_longest': row['upright_ge_0_95_longest_s'],
                              'divergence': row['first_0_2s_reference_tilt_error_over_0_15'],
                              'phase_end_up': ends}), flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
