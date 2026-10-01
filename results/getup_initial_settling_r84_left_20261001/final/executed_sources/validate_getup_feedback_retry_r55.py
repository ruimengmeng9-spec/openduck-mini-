"""State-gated retry controller for perturbed left-side recovery.

Each retry continues from the physical state reached by the previous attempt.
There are no root-state edits, external forces, or altered motor/contact limits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, DURATIONS
from diagnostics.search_getup_support_margin_r49 import foot_support
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def execute_reference(sim, targets, durations):
    for target, duration in zip(targets, durations):
        start = sim.prev.copy()
        n = max(1, round(float(duration) / DT))
        for i in range(n):
            a = min((i + 1) / (.7 * n), 1.)
            a = a * a * (3. - 2. * a)
            sim.step_target((1. - a) * start + a * target)
            if not sim.physical_valid():
                return False
    return True


def execute_bridge(sim, offsets, ids):
    first = sim.home.copy()
    j = sim.model.actuator('right_hip_pitch').id
    first[j] = np.clip(first[j] - .6, sim.lower[j], sim.upper[j])
    for _ in range(round(.4 / DT)):
        sim.step_target(first)
        if not sim.physical_valid():
            return False
    for _ in range(11):
        sim.step_target(sim.home)
        if not sim.physical_valid():
            return False
    for duration, row in zip(DURATIONS, offsets):
        target = sim.home.copy()
        target[ids] = np.clip(sim.home[ids] + row,
                              sim.lower[ids], sim.upper[ids])
        for _ in range(round(duration / DT)):
            sim.step_target(target)
            if not sim.physical_valid():
                return False
    return True


def hold_and_gate(sim, seconds):
    strict = []
    for _ in range(round(seconds / DT)):
        sim.step_target(sim.home)
        if not sim.physical_valid():
            return False, 0.
        strict.append(bool(sim.measure()['stable']))
    tail = 0
    for passed in reversed(strict):
        if not passed:
            break
        tail += 1
    return True, tail * DT


def evaluate(job):
    root, reference_path, checkpoint_path, seed, retries, final_hold_s = job
    root = Path(root)
    scene = root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    reference = np.load(reference_path, allow_pickle=False)
    offsets = np.load(checkpoint_path, allow_pickle=False)['best_offsets']
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    sim.prepare('left_side', seed, True)
    initial = sim.measure()
    sim.clear_audit()
    attempts = []
    reached = False
    valid = True
    for attempt in range(1, retries + 1):
        valid = execute_reference(sim, reference['targets'],
                                  reference['durations_s'])
        if valid:
            valid = execute_bridge(sim, offsets, ids)
        tail = 0.
        if valid:
            valid, tail = hold_and_gate(sim, 2.)
        m = sim.measure()
        support = foot_support(sim)
        attempts.append({'attempt': attempt, 'valid': valid,
                         'strict_tail_s': tail, 'metric': m,
                         'support': support})
        if valid and tail >= 1.:
            reached = True
            break
        if not valid:
            break
    final_tail = 0.
    if reached and valid:
        valid, final_tail = hold_and_gate(sim, final_hold_s)
    final = sim.measure()
    success = bool(initial['up_z'] < .5 and initial['torso_contact']
                   and reached and valid and final_tail >= final_hold_s - DT)
    return {'seed': seed, 'success': success, 'physical_valid': valid,
            'initial': initial, 'attempts': attempts,
            'attempts_used': len(attempts),
            'entry_reached': reached, 'strict_final_tail_s': final_tail,
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
    p.add_argument('--seed-base', type=int, default=755000)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--final-hold-seconds', type=float, default=30.)
    args = p.parse_args()
    if (args.output.exists() or args.trials < 1 or args.retries < 1
            or args.final_hold_seconds < 30.):
        p.error('Need new output, positive trials/retries, final hold >=30 s')
    jobs = [(str(args.root), str(args.reference), str(args.checkpoint),
             args.seed_base + i, args.retries, args.final_hold_seconds)
            for i in range(args.trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, jobs, chunksize=1))
    successes = sum(row['success'] for row in rows)
    report = {'simulation_only': True, 'hardware_readiness': False,
              'orientation': 'left_side',
              'method': 'strict-state-gated recovery retries',
              'trials': args.trials, 'successes': successes,
              'required_successes': 18 if args.trials == 20 else None,
              'passed': bool(args.trials == 20 and successes >= 18),
              'maximum_retries': args.retries,
              'final_hold_seconds': args.final_hold_seconds,
              'perturbations': {'orientation_rad': .05,
                                'joint_position_rad': .02,
                                'generalized_velocity': .01},
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
              'ATTEMPTS', row['attempts_used'],
              'ENTRY', int(row['entry_reached']),
              'TAIL', row['strict_final_tail_s'], flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
