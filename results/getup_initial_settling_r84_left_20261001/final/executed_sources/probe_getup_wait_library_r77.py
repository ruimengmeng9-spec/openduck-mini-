"""Training-only home-wait-duration library; no root edits during a rollout.

R76 retained identity timing and qualified 13/40, equal to baseline. R69
already showed that zero and 0.5 s waits sometimes recover, but its eight
cases and older feedback are not evidence for the current 24-case controller.
Freeze the current path and gains and test a finer wait library before routing
or further fitting. No held-out seed is used to select a wait.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import numpy as np

from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.getup_independent_native import DT, digest
from diagnostics.probe_getup_terminal_timing_r72 import timed_targets
from diagnostics.search_getup_reference_feedback_r64 import record_reference, rollout
from diagnostics.train_getup_prefix_pose_r73 import success

TRAIN = tuple(range(769000, 769008)) + tuple(range(773000, 773016))
WAITS = (2., 0., .1, .2, .3, .4, .5, .6, .75, 1., 1.25, 1.5, 1.75, 2.25, 2.5, 3., 4., 5.)


def change_wait(ck, wait_s):
    if not np.isfinite(wait_s) or not 0 <= wait_s <= 5:
        raise ValueError('Wait duration must be finite and in [0,5] seconds')
    phases = ck['prefix_phases']
    ix = np.flatnonzero(phases == 1)
    if len(ix) == 0 or np.any(np.diff(ix) != 1):
        raise ValueError('Need one contiguous constant home-wait phase')
    values = ck['prefix_targets'][ix]
    if not np.array_equal(values, np.tile(values[0], (len(ix), 1))):
        raise ValueError('Only identical wait commands may be repeated or removed')
    count = round(wait_s / DT)
    targets = np.concatenate([ck['prefix_targets'][:ix[0]],
                              np.tile(values[0], (count, 1)),
                              ck['prefix_targets'][ix[-1] + 1:]])
    changed_phases = np.concatenate([phases[:ix[0]], np.full(count, 1, dtype=phases.dtype),
                                     phases[ix[-1] + 1:]])
    return {**ck, 'prefix_targets': targets, 'prefix_phases': changed_phases}


def save_trace(path, trace):
    np.savez_compressed(path, time=[r[0] for r in trace], qpos=[r[1] for r in trace],
                        qvel=[r[2] for r in trace], phase=[r[3] for r in trace],
                        strict=[r[4] for r in trace], margin_m=[r[5] for r in trace],
                        imu_error=[r[6] for r in trace], residual_rad=[r[7] for r in trace])


def evaluate(job):
    wait, output = job
    sim, cases, ck, ids = base._CTX
    changed = change_wait(ck, wait)
    targets, phases = timed_targets(sim, changed, (.4, .6, .8), 2.)
    gains = np.concatenate([ck['prefix_gains'], ck['terminal_gains']])
    directory = Path(output) / f'wait_{round(wait / DT):03d}'
    directory.mkdir(exist_ok=False)
    row = {'wait_s': wait, 'target_steps': len(targets), 'cases': []}
    try:
        ref = record_reference(sim, *cases[None][:2], True, targets)
    except RuntimeError as exc:
        return {**row, 'nominal_success': False, 'successes': 0, 'rejected_reference': str(exc)}
    nominal, trace = rollout(sim, *cases[None][:2], True, targets, phases, ids, ref, gains, True)
    nominal['success'] = success(nominal, len(targets), 1.)
    save_trace(directory / 'nominal.npz', trace)
    row.update(nominal=nominal, nominal_success=nominal['success'], successes=0)
    # Preserve the mandatory nominal gate. A rejected wait is not 24 simulated failures.
    if not nominal['success']:
        return row
    for seed in TRAIN:
        source, peaks, initial, initial_hash = cases[seed]
        value, trace = rollout(sim, source, peaks, True, targets, phases, ids, ref, gains, True)
        value['success'] = success(value, len(targets), 1.)
        save_trace(directory / f'train_{seed}.npz', trace)
        row['cases'].append({'seed': seed, 'initial_fallen': True, 'initial': initial,
                             'initial_state_sha256': initial_hash, **value})
        row['successes'] += int(value['success'])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.workers <= 6:
        parser.error('Need a fresh output and 1 through 6 CPU workers')
    root = Path('/data/shijinsheng/open_duck')
    scene = root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    with np.load(args.checkpoint, allow_pickle=False) as data:
        ck = {key: data[key] for key in data.files}
    for key in ('best_wait_delta', 'best_time_factors', 'best_pose_delta'):
        if key in ck and np.any(ck[key] != (1 if key == 'best_time_factors' else 0)):
            parser.error('Need frozen identity baseline; do not combine interventions')
    if len(np.flatnonzero(ck['prefix_phases'] == 1)) != round(2. / DT):
        parser.error('Expected current baseline wait of exactly two seconds')
    args.output.mkdir(parents=True)
    dependencies = ('train_getup_wait_pose_r74.py', 'probe_getup_terminal_timing_r72.py',
                    'search_getup_reference_feedback_r64.py', 'train_getup_prefix_pose_r73.py',
                    'validate_getup_fullpath_r27.py', 'train_getup_fullpath_r27.py',
                    'getup_independent_native.py', 'audit_getup_load_support.py',
                    'getup_aligned_extension.py', 'getup_load_search.py')
    executed = args.output / 'executed_sources'
    executed.mkdir()
    for name in (*dependencies, Path(__file__).name):
        shutil.copy2(Path(__file__).parent / name, executed / name)
    contract = {'simulation_only': True, 'hardware_readiness': False,
                'full_task_completed': False, 'training_only': True, 'pose': 'left_side',
                'method': 'constant home-wait duration library',
                'hypothesis': 'different contact configurations may require different home-wait durations',
                'training_seeds': list(TRAIN), 'heldout_seeds_used': [],
                'wait_grid_s': list(WAITS), 'control_dt_s': DT, 'workers': args.workers,
                'nominal_gate_required': True, 'short_training_gate_s': 1.,
                'no_midpath_state_reset': True,
                'frozen': ['early_targets_and_timing', 'wait_target', 'postwait_commands',
                           'feedback_gains', 'physics', 'motor_limits', 'strict_gate'],
                'source_sha256': digest(__file__), 'checkpoint_sha256': digest(args.checkpoint),
                'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
                'dependency_hashes': {name: digest(executed / name) for name in dependencies}}
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=base.init_worker,
                             initargs=(str(scene), str(stand), ck, TRAIN)) as pool:
        for row in pool.map(evaluate, [(wait, str(args.output)) for wait in WAITS], chunksize=1):
            rows.append(row)
            (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
            print('WAIT', row['wait_s'], 'NOMINAL', row['nominal_success'],
                  'TRAIN', row['successes'], '/24', flush=True)
    eligible = [row for row in rows if row['nominal_success']]
    if not rows[0]['nominal_success']:
        raise RuntimeError('Frozen baseline nominal failed; no candidate may be selected')
    best = max(eligible, key=lambda row: (row['successes'], sum(r['score'] for r in row['cases'])))
    oracle = {seed: [row['wait_s'] for row in eligible
                     if next(r for r in row['cases'] if r['seed'] == seed)['success']] for seed in TRAIN}
    report = {**contract, 'baseline_successes': rows[0]['successes'],
              'selected_wait_s': best['wait_s'], 'selected_training_successes': best['successes'],
              'library_training_oracle_successes': sum(bool(waits) for waits in oracle.values()),
              'oracle_is_not_a_policy_or_validation': True,
              'successful_waits_by_training_seed': oracle, 'results': rows}
    np.savez_compressed(args.output / 'selected.npz', **change_wait(ck, best['wait_s']),
                        selected_wait_s=best['wait_s'])
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('PROBE_COMPLETE', best['successes'], '/24 ORACLE',
          report['library_training_oracle_successes'], '/24', flush=True)


if __name__ == '__main__':
    main()
