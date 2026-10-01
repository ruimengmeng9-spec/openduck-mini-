"""Long-hold validation for the R59 hard-case-refined transition."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_hardcase_refine_r59 import HELDOUT_SEEDS
from diagnostics.search_getup_robust_transition_r57 import perturb_snapshot
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, DURATIONS, capture
from diagnostics.search_getup_support_margin_r49 import foot_support
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def validate_one(job):
    scene, stand, source, peaks, finite, offsets, seed, hold_s, required_s = job
    sim = StrictSim(scene, stand)
    initial = perturb_snapshot(sim, source, seed)
    sim.restore(initial)
    sim.peaks, sim.finite = peaks.copy(), finite
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
    hold_strict, margins = [], []
    if valid:
        for _ in range(round(hold_s / DT)):
            sim.step_target(sim.home)
            valid = sim.physical_valid()
            m = sim.measure()
            hold_strict.append(bool(m['stable']))
            margins.append(foot_support(sim)['margin_m'])
            if not valid:
                break
    tail = 0
    for passed in reversed(hold_strict):
        if not passed:
            break
        tail += 1
    final = sim.measure()
    final_support = foot_support(sim)
    success = bool(valid and len(hold_strict) == round(hold_s / DT)
                   and tail * DT >= required_s - DT)
    return {'seed': seed, 'physical_valid': valid,
            'executed_hold_s': len(hold_strict) * DT,
            'strict_tail_s': tail * DT,
            'minimum_hold_margin_m': float(min(margins)) if margins else None,
            'final': final, 'final_support': final_support,
            'success': success}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--hold-seconds', type=float, default=35.)
    p.add_argument('--required-strict-seconds', type=float, default=30.)
    args = p.parse_args()
    if (args.output.exists() or args.workers < 1
            or args.required_strict_seconds < 30.
            or args.hold_seconds < args.required_strict_seconds):
        p.error('Need new output, workers, and >=30 s strict requirement')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    offsets = np.load(args.checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, captured = capture(sim, reference, 0)
    snapshot = captured['snapshot']
    jobs = [(str(scene), str(stand), snapshot, captured['peaks'],
             captured['finite'], offsets, seed, args.hold_seconds,
             args.required_strict_seconds) for seed in HELDOUT_SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(validate_one, jobs, chunksize=1))
    successes = sum(row['success'] for row in rows)
    args.output.mkdir(parents=True)
    report = {'simulation_only': True, 'hardware_readiness': False,
              'stage': 'captured transition long hold; not full recovery',
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'checkpoint_sha256': digest(args.checkpoint),
              'source_sha256': digest(__file__), 'prefix': prefix,
              'captured_state_hash': captured['state_hash'],
              'heldout_seeds': list(HELDOUT_SEEDS),
              'requested_hold_s': args.hold_seconds,
              'required_strict_s': args.required_strict_seconds,
              'successes': successes, 'trials': len(rows),
              'success_rate': successes / len(rows), 'results': rows,
              'full_task_completed': False}
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    for row in rows:
        print('SEED', row['seed'], 'SUCCESS', int(row['success']),
              'TAIL', round(row['strict_tail_s'], 2),
              'VALID', int(row['physical_valid']), flush=True)
    print('LONG_HOLD', successes, '/', len(rows), flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
