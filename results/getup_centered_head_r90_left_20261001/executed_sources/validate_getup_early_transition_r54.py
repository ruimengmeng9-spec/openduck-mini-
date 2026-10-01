"""Full-perturbation test with immediate takeover after the R27 recovery path."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, DURATIONS
from diagnostics.search_getup_support_margin_r49 import foot_support
from diagnostics.train_getup_fullpath_r27 import rollout, state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def capture_early(sim, reference, seed, perturb=True):
    prefix, _ = rollout(sim, 'left_side', reference['targets'],
                        reference['durations_s'], seed, perturb, 0.)
    if not prefix['initial_fallen']:
        raise RuntimeError('initial state was not fallen')
    if not prefix['valid']:
        raise RuntimeError('recovery prefix failed physical audit')
    first = sim.home.copy()
    j = sim.model.actuator('right_hip_pitch').id
    first[j] = np.clip(first[j] - .6, sim.lower[j], sim.upper[j])
    for _ in range(round(.4 / DT)):
        sim.step_target(first)
        if not sim.physical_valid():
            raise RuntimeError('first pulse failed physical audit')
    for control in range(11):
        sim.step_target(sim.home)
        if not sim.physical_valid():
            raise RuntimeError('return pulse failed physical audit')
    return prefix, {'phase': 'early_return', 'control_index': control,
                    'metric': sim.measure(), 'snapshot': sim.snapshot(),
                    'peaks': sim.peaks.copy(), 'finite': sim.finite,
                    'state_hash': state_hash(sim)}


def evaluate(job):
    root, reference_path, checkpoint_path, seed, hold_s, perturb = job
    root = Path(root)
    scene = root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    reference = np.load(reference_path, allow_pickle=False)
    offsets = np.load(checkpoint_path, allow_pickle=False)['best_offsets']
    try:
        prefix, captured = capture_early(sim, reference, seed, perturb)
    except Exception as exc:
        return {'seed': seed, 'success': False, 'physical_valid': False,
                'stage': 'capture', 'error': str(exc)}
    transition = captured['metric']
    sim.restore(captured['snapshot'])
    sim.peaks, sim.finite = captured['peaks'].copy(), captured['finite']
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    valid = True
    for duration, row in zip(DURATIONS, offsets):
        target = sim.home.copy()
        target[ids] = np.clip(sim.home[ids] + row,
                              sim.lower[ids], sim.upper[ids])
        for _ in range(round(duration / DT)):
            sim.step_target(target)
            valid = sim.physical_valid()
            if not valid:
                break
        if not valid:
            break
    strict = []
    min_margin = np.inf
    if valid:
        for _ in range(round(hold_s / DT)):
            sim.step_target(sim.home)
            valid = sim.physical_valid()
            strict.append(bool(sim.measure()['stable']))
            min_margin = min(min_margin, foot_support(sim)['margin_m'])
            if not valid:
                break
    tail = 0
    for passed in reversed(strict):
        if not passed:
            break
        tail += 1
    final = sim.measure()
    success = bool(valid and len(strict) == round(hold_s / DT)
                   and tail * DT >= 30. - DT)
    return {'seed': seed, 'success': success, 'physical_valid': valid,
            'stage': 'complete', 'transition': transition,
            'capture_state_hash': captured['state_hash'],
            'strict_tail_s': tail * DT, 'executed_hold_s': len(strict) * DT,
            'minimum_support_margin_m_during_hold': float(min_margin),
            'final': final, 'final_support': foot_support(sim),
            'peaks': sim.peaks.copy()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--seed-base', type=int, default=754000)
    p.add_argument('--hold-seconds', type=float, default=35.)
    p.add_argument('--include-nominal', action='store_true')
    args = p.parse_args()
    if args.output.exists() or args.trials < 1 or args.hold_seconds < 30.:
        p.error('Need new output, trials >=1, hold >=30 s')
    jobs = []
    if args.include_nominal:
        jobs.append((str(args.root), str(args.reference), str(args.checkpoint),
                     0, args.hold_seconds, False))
    jobs.extend((str(args.root), str(args.reference), str(args.checkpoint),
                 args.seed_base + i, args.hold_seconds, True)
                for i in range(args.trials))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, jobs, chunksize=1))
    perturbed = rows[1:] if args.include_nominal else rows
    successes = sum(row['success'] for row in perturbed)
    report = {'simulation_only': True, 'hardware_readiness': False,
              'orientation': 'left_side',
              'transition': 'immediate_after_r27_recovery_then_return_control_10',
              'nominal': rows[0] if args.include_nominal else None,
              'trials': args.trials, 'successes': successes,
              'required_successes': 18 if args.trials == 20 else None,
              'passed': bool(args.trials == 20 and successes >= 18),
              'perturbations': {'orientation_rad': .05,
                                'joint_position_rad': .02,
                                'generalized_velocity': .01},
              'hold_seconds': args.hold_seconds,
              'required_strict_tail_seconds': 30.,
              'scene_sha256': digest(args.root / 'training/getup_decomposed_r4/model/scene.xml'),
              'stand_sha256': digest(args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'),
              'reference_sha256': digest(args.reference),
              'checkpoint_sha256': digest(args.checkpoint),
              'source_sha256': digest(__file__), 'results': rows,
              'full_task_completed': False}
    args.output.mkdir(parents=True)
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in (
        'orientation', 'successes', 'trials', 'passed')}, indent=2), flush=True)
    for row in rows:
        print('SEED', row['seed'], 'SUCCESS', int(row['success']),
              'STAGE', row['stage'], 'TAIL', row.get('strict_tail_s', 0.),
              'UP0', round(row.get('transition', {}).get('up_z', -9.), 3),
              flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
